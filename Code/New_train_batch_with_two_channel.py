# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import math
import numpy as np
import pandas as pd

import sys

sys.path.append('./lib/')
from pkl_process import *
from lib.utils import load_graphdata_channel_my, compute_val_loss_sttn, generate_torch_datasets, load_features

from time import time
import shutil
import argparse
import configparser
from tensorboardX import SummaryWriter
import os

# from Model.ST_Transformer_new import STTransformer # STTN model with linear layer to get positional embedding
# from Model.ST_Transformer_new_sinembedding import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_LapGCN import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_LapGCN_and_Trans import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_LapGCN_and_Trans_Contrast_loss import STTransformer_sinembedding
from Model.ST_Transformer_new_sinembedding_with_Two_channel_model import STTransformer_sinembedding
# from Model.ST_Transformer_new_sinembedding_with_Two_channel_model_robust_experiment import STTransformer_sinembedding

# STTN model with sin()/cos() to get positional embedding, the same as "Attention is all your need"

# %%

if __name__ == '__main__':

    # params_path = './Experiment/PEMS04_with_Two_channel'  ## Path for saving network parameters
    # params_path = './Experiment/PEMS08_with_Two_channel'  ## Path for saving network parameters
    params_path = './Experiment/SH_with_Two_channel'  ## Path for saving network parameters
    # params_path = './Experiment/CQ_with_Two_channel'  ## Path for saving network parameters

    # params_path = './Experiment/HZ_with_Two_channel'
    print('params_path:', params_path)
    # filename = './PEMSD7/V_25_r2_d1_w2_astcgn.npz' ## Data generated by prepareData.py
    # filename = './Data/SH/SH_flow_r2_d1_w2_astcgn.npz'  ## Data generated by prepareData.py
    ### Training Hyparameter
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = device
    print("device:{}".format(device))
    batch_size = 16
    learning_rate = 0.004    # 0.004（SH，HZ, CQ数据集）
    # learning_rate = 0.008    # PEMS0408 最优0.008 epoch=300 submodel2(不加Fourier)
    # learning_rate = 0.01    # PEMS0408 最优0.008 epoch=300 submodel2
    # epochs = 300
    epochs = 800
    # LH
    feat_name = "./Data/SH/SH_flow.csv"
    # feat_name = "./Data/CQ/CQ_flow.csv"
    # feat_name = "./Data/HZ/HZ_flow.csv"
    # feat_name = "./Data/PEMS04/PEMS04_flow.csv"
    # feat_name = "./Data/PEMS08/PEMS08_flow.csv"
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

    # LH
    # kl loss
    def compute_kl_loss(p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()

        loss = (p_loss + q_loss) / 2
        return loss


    ### Adjacency Matrix Import
    adj_mx = pd.read_csv('./Data/SH/SH_adj.csv', header=None)
    # adj_mx = pd.read_csv('./Data/CQ/CQ_adj.csv', header=None)
    # adj_mx = pd.read_csv('./Data/HZ/HZ_adj.csv', header=None)
    # adj_mx = pd.read_csv('./Data/PEMS04/PEMS04_adj.csv', header=None)
    # adj_mx = pd.read_csv('./Data/PEMS08/adj_PEMS08.csv', header=None)
    adj_mx = np.array(adj_mx)
    A = adj_mx
    A = torch.Tensor(A)

    ### Training Hyparameter
    in_channels = 1  # Channels of input
    embed_size = 64  # Dimension of hidden embedding features
    # time_num = 288 # 一天的间隔数 PEM04 and PEM08
    time_num = 72 # SH  72?需要再跑一次实验  原96
    # time_num = 62   # CQ
    # time_num = 69  # HZ

    num_layers = 3  # Number of ST Block
    T_dim = 12  # Input length, should be the same as prepareData.py
    output_T_dim = 4  # Output Expected length

    heads = 4  # Number of Heads in MultiHeadAttention AAAI论文为4

    cheb_K = 2  # Order for Chebyshe    v Polynomials (Eq 2)
    forward_expansion = 4  # Dimension of Feed Forward Network: embed_size --> embed_size * forward_expansion --> embed_size
    dropout = 0.2

    ### Construct Network

    net1 = STTransformer_sinembedding(
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
        forward_expansion, dropout=dropout)

    net2 = STTransformer_sinembedding(
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
        forward_expansion, dropout=dropout)

    net1.to(device)
    net2.to(device)

    ### Training Process
    #### Load the parameter we have already learnt if start_epoch does not equal to 0
    start_epoch = 0
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    #### Loss Function Setting
    criterion = nn.MSELoss().to(device)

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    #
    # 传入多个模型参数
    optimizer = torch.optim.Adam([
        {"params": net1.parameters(), "lr": learning_rate, "weight_decay": 0.001},
        {"params": net2.parameters(), "lr": learning_rate, "weight_decay": 0.001},
    ])
    # 优化器传入多个模型参数
    # opt = torch.optim.Adam([
    #     {'params': model_1.parameters(), 'lr': 0.001, },
    #     {'params': model_2.parameters()},
    # ])

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    # LH
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01)
    # criterion = nn.L1Loss().to('cuda:0')

    #### Training Log Set and Print Network, Optimizer
    sw = SummaryWriter(logdir=params_path, flush_secs=5)
    print("Network1:", net1)
    # print("Network2:", net2)
    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    start_time = time()
    kl_loss_csv_data = []

    alpha = 5

    #### Load parameters from files
    if start_epoch > 0:
        params_filename1 = os.path.join(params_path, '1epoch_%s.params' % start_epoch)
        params_filename2 = os.path.join(params_path, '2epoch_%s.params' % start_epoch)
        net1.load_state_dict(torch.load(params_filename1))
        net2.load_state_dict(torch.load(params_filename2))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename1)
        print('load weight from: ', params_filename2)

    #### train model
    for epoch in range(start_epoch, epochs):
        training_loss_all = 0.0
        kl_loss_all = 0
        ##### Parameter Saving
        # params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
        params_filename1 = os.path.join(params_path, '1epoch_%s.params' % epoch)
        params_filename2 = os.path.join(params_path, '2epoch_%s.params' % epoch)
        ##### Evaluate on Validation Set
        # No Value
        # val_loss = compute_val_loss_sttn(net, val_loader, criterion, sw, epoch)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_epoch = epoch
            # torch.save(net.state_dict(), params_filename)
            # print('save parameters to file: %s' % params_filename)

        net1.train()  # ensure dropout layers are in train mode
        net2.train()
        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # outputs = net(encoder_inputs.permute(0, 2, 1, 3))
            # loss = criterion(outputs, labels)
            # 输出的值有两个，两个子模型都有dropout，计算两个值的loss，以及标签和预测值的loss
            outputs_dropout1 = net1(encoder_inputs.permute(0, 2, 1, 3))
            # print("output1:" , outputs_dropout1)
            # 先回复原值，再进行loss
            outputs_dropout1 = outputs_dropout1 * max_val
            labels = labels * max_val
            loss_dropout1 = criterion(outputs_dropout1, labels)
            loss= criterion(outputs_dropout1, labels)
            # 消融实验，去正则化约束
            outputs_dropout2 = net2(encoder_inputs.permute(0, 2, 1, 3))
            # print("output2:" , outputs_dropout2)
            # 先回复原值，再进行loss

            outputs_dropout2 = outputs_dropout2 * max_val
            loss_dropout2 = criterion(outputs_dropout2, labels)
            loss_result = criterion(outputs_dropout1, outputs_dropout2)
            ce_loss = 0.5*(loss_dropout1 + loss_dropout2)
            kl_loss = compute_kl_loss(outputs_dropout1, outputs_dropout2)

            # print("kl_loss {}".format(kl_loss.item()))

            kl_loss_all += kl_loss
            # loss = (loss_dropout1 + loss_dropout2) + alpha*loss_result
            loss = ce_loss + alpha * kl_loss

            # print(ce_loss.item())
            # print(loss.item())
            # exit()
            loss.backward()
            optimizer.step()    # L2正则化的优化器
            training_loss = loss.item()
            training_loss_all += training_loss
            # # save parameters to file : params filename
            # if training_loss < best_val_loss:
            #     best_val_loss = training_loss
            #     best_epoch = epoch
            #     torch.save(net.state_dict(), params_filename)
            #     print('save parameters to file: %s' % params_filename)
            # LH

            global_step += 1
            sw.add_scalar('training_loss', training_loss, global_step)
            # LH
            print('global step: %s, training loss: %.4f, KL_loss: %.4f, time: %.2fs' % (global_step, training_loss, kl_loss, time() - start_time))
            # print('global step: %s, training loss: %.4f, time: %.2fs' % (global_step, training_loss, time() - start_time))
            # LH
            # if global_step % 38 == 0:
            #     print('global step: %s, training loss: %.4f, time: %.2fs' % (
            #     global_step, training_loss_all, time() - start_time))
        print('global step: %s, training loss: %.4f, KL_loss_all: %.4f, time: %.2fs' % (global_step, training_loss_all,kl_loss_all, time() - start_time))
        # print('global step: %s, training loss: %.4f, time: %.2fs' % (global_step, training_loss_all, time() - start_time))
        kl_loss_csv_data.append(kl_loss_all.__float__());
        if training_loss_all < best_val_loss:
            best_val_loss = training_loss_all
            best_epoch = epoch
            # torch.save(net.state_dict(), params_filename)
            torch.save(net1.state_dict(), params_filename1)
            torch.save(net2.state_dict(), params_filename2)
            # print('save parameters to file: %s' % params_filename)
            print('save parameters to file: %s' % params_filename1)
            print('save parameters to file: %s' % params_filename2)

        # prediction test
        # 计算test的loss值，以及一些指标，这样就不需要存储训练参数，取得其中的最小值或者平均值

    print('best epoch:', best_epoch)
    print("best training_loss_all", best_val_loss)
    # np.savetxt("kl_loss.csv",kl_loss_csv_data, delimiter=",")











