import math
import torch as th

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.nn as nn

class AdaptiveDiffusionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(AdaptiveDiffusionLayer, self).__init__()
        # t is the neighborhood radius; it is learnable [cite: 504]
        self.t = nn.Parameter(th.FloatTensor(1).fill_(1.0)) 
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # Approximation of Heat Kernel: exp(-t(I - adj)) [cite: 495]
        # For a hackathon, 1st or 2nd order Taylor expansion is efficient:
        # P_diff = (1 - t) * I + t * adj
        identity = th.eye(adj.size(0)).to(adj.device)
        adj_diff = (1 - self.t) * identity + self.t * adj
        
        support = th.spmm(x, self.weight)
        return th.spmm(adj_diff, support)

# Update your GCN class to use the new layer
class ADC_GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ADC_GCN, self).__init__()
        self.gc1 = AdaptiveDiffusionLayer(nfeat, nhid)
        self.gc2 = AdaptiveDiffusionLayer(nhid, nclass)
        self.dropout = dropout


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = th.spmm(infeatn, self.weight)
        output = th.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = th.relu(x)
        x = th.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        return x
