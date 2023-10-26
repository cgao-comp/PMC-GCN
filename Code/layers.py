import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from lib.utils import scaled_Laplacian, cheb_polynomial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, in_features, out_features, adj, K, bias=True):
#     def __init__(self, nfeat, nhid, nclass, dropout, K):
#     def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.DEVICE = 'cuda:0'
        # LH
        # self.DEVICE = 'cpu'
        # LH
        self.K = K
        adj = np.array(adj.cpu())
        L_tilde = scaled_Laplacian(adj)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.DEVICE) for i in cheb_polynomial(L_tilde, K)]
        
        
        self.in_channels = in_features
        self.out_channels = out_features
        
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (B, N, C] --> (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
#         x = x.permute(0, 1, 3, 2)
        batch_size, num_of_vertices, in_channels = x.shape

           

        graph_signal = x

        output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

        for k in range(self.K):

            T_k = self.cheb_polynomials[k]  # (N,N)

            theta_k = self.Theta[k]  # (in_channel, out_channel)

            rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1) # （b, F_in, N) * (N, N) --> (b, F_in, N) --> (b, N, F_in)

            output = output + rhs.matmul(theta_k) # (b, N, F_in) * (F_in, F_out) --> (b, N, F_out) 


        
        result = F.relu(output)

        return result


class GCN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, adj, device):
        super(GCN_layer, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        # LH 增加可学习训练参数矩阵α
        self.alpha_matrix = nn.Parameter(torch.eye(adj.shape[0]), requires_grad=True)
        # LH
        self.reset_parameters()

    def reset_parameters(self):
        # 这里需要注意一下alpha_matrix的参数值是否需要进行初始化

        nn.init.normal_(self.alpha_matrix, 0.5, 0.5)

        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, embedding)
        inputs = inputs.permute(1, 0, 2)
        # inputs = inputs.transpose(1, 0, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))

        # LH
        # 因为使用cpu训练，不需要把alpha_matrix 放入GPU中
        # device = torch.device('cuda')
        # self.alpha_matrix = self.alpha_matrix * torch.eye(num_nodes).to(device)
        new_alpha_matrix = self.alpha_matrix * torch.eye(self._num_nodes).to(self.device)
        new_laplacian = self.laplacian + new_alpha_matrix
        # LH

        # AX (num_nodes, batch_size * seq_len)
        ax = new_laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs


class GCN_layer_No_Learn(nn.Module):
    def __init__(self, input_dim, output_dim, adj, device):
        super(GCN_layer_No_Learn, self).__init__()
        self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        self.weights = nn.Parameter(torch.FloatTensor(self._input_dim, self._output_dim))
        # LH 增加可学习训练参数矩阵α
        # self.alpha_matrix = nn.Parameter(torch.eye(adj.shape[0]), requires_grad=True)
        # LH
        self.reset_parameters()

    def reset_parameters(self):
        # 这里需要注意一下alpha_matrix的参数值是否需要进行初始化，需要查阅资料确定一下

        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('tanh'))

    def forward(self, inputs):
        # (batch_size, seq_len, num_nodes)
        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, embedding)
        inputs = inputs.permute(1, 0, 2)
        # inputs = inputs.transpose(1, 0, 2)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))

        # LH
        # 因为使用cpu训练，不需要把alpha_matrix 放入GPU中
        # device = torch.device('cuda')
        # self.alpha_matrix = self.alpha_matrix * torch.eye(num_nodes).to(device)
        # new_alpha_matrix = self.alpha_matrix * torch.eye(self._num_nodes).to(self.device)
        # new_laplacian = self.laplacian + new_alpha_matrix
        # LH

        # AX (num_nodes, batch_size * seq_len)
        # ax = new_laplacian @ inputs
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.weights)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs