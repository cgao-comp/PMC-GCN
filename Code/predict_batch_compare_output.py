# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:34:11 2021

@author: wzhangcd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import numpy as np
import pandas as pd
from time import time

import sys
sys.path.append('./lib/')
from pkl_process import *
from lib.utils import load_graphdata_channel_my, compute_val_loss_sttn, \
    masked_mape_np, re_normalization, max_min_normalization, re_max_min_normalization, \
    load_features, generate_torch_datasets

from time import time
import shutil
import argparse
import configparser
from tensorboardX import SummaryWriter
import os

# from ST_Transformer_new import STTransformer # STTN model with linear layer to get positional embedding
# from Model.ST_Transformer_new_sinembedding import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_LapGCN import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_LapGCN_and_Trans_Contrast_loss import STTransformer_sinembedding
from Model.ST_Transformer_new_sinembedding_with_LapGCN_and_Trans import STTransformer_sinembedding  # 测试两个输出的相似程度
# from Model.ST_Transformer_new_sinembedding_with_LapGCN_and_Trans_Contrast_loss_NoLearn import STTransformer_sinembedding

#STTN model with sin()/cos() to get positional embedding, the same as "Attention is all your need"

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, max_val, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(device)
            labels = labels.to(device)
            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs.permute(0, 2, 1, 3))

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        # input = re_normalization(input, _mean, _std)
        # re_normalization
        # input = input * max_val

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        prediction = prediction * max_val
        data_target_tensor = data_target_tensor * max_val

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)

def predict_main(params_filename, global_step, data_loader, data_target_tensor, max_val, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    # 若是双通道模式，选择不同的模型的参数，用于对应的模型进行测试
    # params_filename = os.path.join(params_path, '1epoch_%s.params' % global_step)
    # print('load weight from:', params_filename)
    # 只有一个net
    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    net.load_state_dict(torch.load(params_filename))
    # LH 解决Error(s) in loading state_dict for STTransformer: Missing key(s) in state_dict: "Transformer.encoder.layers.0
    # 未解决， 是因为模型不匹配的原因，训练是使用new——siembendding 预测时也要使用该模型
    # best_model = torch.load(params_filename)
    # net.load_state_dict(best_model)

    # 读取网络的参数，获取到可学习矩阵
    # learnable_1 = net.Transformer.encoder.layers[0].ES_GCN.gcn.gcn2.alpha_matrix
    # learnable_2 = net.Transformer.encoder.layers[1].ES_GCN.gcn.gcn2.alpha_matrix
    # learnable_3 = net.Transformer.encoder.layers[2].ES_GCN.gcn.gcn2.alpha_matrix
    # # type_leanable = learnable_1.dtype
    # # print(type_leanable)
    # learnable_1 = learnable_1.cpu().detach().numpy()
    # # alpha = alpha.detach().numpy()
    # data1 = pd.DataFrame(learnable_1)
    # data1.to_csv("learnable_1.csv", header=None, index=0)
    #
    # learnable_2 = learnable_2.cpu().detach().numpy()
    # data2 = pd.DataFrame(learnable_2)
    # data2.to_csv("learnable_2.csv", header=None, index=0)
    #
    # learnable_3 = learnable_3.cpu().detach().numpy()
    # data3 = pd.DataFrame(learnable_3)
    # data3.to_csv("learnable_3.csv", header=None, index=0)
    #
    # data_all = pd.DataFrame((data1 + data2 + data3)/3)
    # data_all.to_csv("learnable_sum.csv", header=None, index=0)


    # type_leanable = learnable_1.dtype
    # print(learnable_1)
    # print(learnable_2)
    # print(learnable_3)
    # print(type_leanable)
    # LH

    predict_and_save_results_my(net, data_loader, data_target_tensor, global_step, max_val, params_path, type)



if __name__=='__main__':
    
    ## Best Epoch during Training
    best_epoch = 796


    ## Same Setting as train_my.py  Experiment/SH_embed_size64_without_week_day_hour_wth_contrast_loss
    # params_path = './Experiment/SH_embed_size64_without_week_day_hour_wth_contrast_loss' ## Path for saving network parameters
    # params_path = './Experiment/SH_embed_size64_without_week_day_hour_wth_contrast_loss_R-drop_No_Learn' ## Path for saving network parameters
    params_path = './Experiment/SH_embed_size64_without_week_day_hour_with_LapGCN_and_Trans' ## Path for saving network parameters


    # params_path = './Experiment/SH_embed_size64_without_week_day_hour_wth_contrast_loss_R-drop'  ## Path for saving network parameters
    # params_path = './Experiment/CQ_embed_size64_without_week_day_hour_wth_contrast_loss_R-drop'  ## Path for saving network parameters
    # params_path = './Experiment/HZ_embed_size64_without_week_day_hour_wth_contrast_loss_R-drop'  ## Path for saving network parameters

    # ######双通道
    # params_path = './Experiment/SH_with_Two_channel'  ## Path for saving network parameters
    # params_path = './Experiment/HZ_with_Two_channel'  ## Path for saving network parameters


    print('params_path:', params_path)

    # filename = './PEMSD7/V_25_r2_d1_w2_astcgn.npz' ## Data generated by prepareData.py
    # num_of_hours, num_of_days, num_of_weeks = 2, 1, 2 ## The same setting as prepareData.py
    ### Training Hyparameter
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = DEVICE
    batch_size = 16
    # learning_rate = 0.01
    learning_rate = 0.004
    epochs = 800

    feat_name = "./Data/SH/SH_flow.csv"
    # feat_name = "./Data/CQ/CQ_flow.csv"
    # feat_name = "./Data/HZ/HZ_flow.csv"
    # 加载特征矩阵X
    feat = load_features(feat_name)
    seq_len = 12
    pre_len = 4
    split_ratio = 0.8
    normalize = True
    time_len = None
    shuffle = True
    test_dataset, train_dataset, train_x_tensor, train_target_tensor, test_x_tesor, test_target_tensor, max_val = \
        generate_torch_datasets(feat, seq_len, pre_len, time_len, split_ratio, normalize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    
    ### Generate Data Loader
    # train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel_my(
    #     filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size)
    
    ### Adjacency Matrix Import
    adj_mx = pd.read_csv('./Data/SH/SH_adj.csv', header = None)
    # adj_mx = pd.read_csv('./Data/CQ/CQ_adj.csv', header = None)
    # adj_mx = pd.read_csv('./Data/HZ/HZ_adj.csv', header = None)
    # adj_mx = import_pkl('/home/wzhangcd@HKUST/Commonpkg/adj_mx_tran_89.pkl')
    adj_mx = np.array(adj_mx)
    A = adj_mx
    A = torch.Tensor(A)

    ### Training Hyparameter
    in_channels = 1  # Channels of input
    embed_size = 64  # Dimension of hidden embedding features
    # time_num = 288 一天的间隔数
    time_num = 96  # SH数据
    # time_num = 62    # CQ数据
    # time_num = 69    # HZ数据
    num_layers = 3  # Number of ST Block
    T_dim = 12  # Input length, should be the same as prepareData.py
    output_T_dim = 4  # Output Expected length
    heads = 4  # Number of Heads in MultiHeadAttention
    cheb_K = 2  # Order for Chebyshev Polynomials (Eq 2)
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0

    ### Construct Network
    net = STTransformer_sinembedding(
        A,
        in_channels, 
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads,
        cheb_K,
        device,
        forward_expansion,
        dropout
    )

    net.to(device)
    start_time = time()
    predict_main(params_path, best_epoch, test_loader, test_target_tensor, max_val, 'test')
    end_time = time()
    print("time cost:%.2fs" % (end_time-start_time))
