import torch.nn as nn
import torch.nn.functional as F
from layers import cheb_conv, GCN_layer, GCN_layer_No_Learn
import numpy as np


class chev_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, cheb_K, dropout):
        super(GCN, self).__init__()

        self.gc1 = cheb_conv(nfeat, nhid, adj, cheb_K)
        self.gc2 = cheb_conv(nhid, nclass, adj, cheb_K)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)


class GCN(nn.Module):
    def __init__(self, adj, input_dim, output_dim, device):
        super(GCN, self).__init__()
        # self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        # 两层GCN
        self.gcn1 = GCN_layer(self._input_dim, self._input_dim * 2, adj, device)
        self.gcn2 = GCN_layer(self._input_dim * 2, self._output_dim, adj, device)

    def forward(self, inputs, adj_laplacian):
        output_first = self.gcn1(inputs)

        # 添加激活函数
        output_second = self.gcn2(output_first)

        return output_second


class GCN_No_Learn(nn.Module):
    def __init__(self, adj, input_dim, output_dim, device):
        super(GCN_No_Learn, self).__init__()
        # self.register_buffer('laplacian', calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self._num_nodes = adj.shape[0]
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.device = device
        # 两层GCN
        self.gcn1 = GCN_layer_No_Learn(self._input_dim, self._input_dim * 2, adj, device)
        self.gcn2 = GCN_layer_No_Learn(self._input_dim * 2, self._output_dim, adj, device)

    def forward(self, inputs, adj_laplacian):
        output_first = self.gcn1(inputs)

        output_second = self.gcn2(output_first)

        return output_second