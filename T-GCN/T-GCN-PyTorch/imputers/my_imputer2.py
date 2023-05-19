import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Sequential, GCNConv

from torch_geometric.utils import add_self_loops, degree, get_laplacian, to_dense_adj
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class FPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__()  
        self.lin = Linear(in_channels, out_channels)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, x_hat, edge_index, edge_weight, M):
        # Step 1: Add self-loops to the adjacency matrix.
        

        # Step 3: Reseting the known values.
        x_tilde = x * (M) + x_hat * (~M)

        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]


        
        # Step 4: Propagate the features.
        out = self.propagate(edge_index, x=x_tilde, edge_weight=edge_weight, norm=norm)

        # apply linear transformation
        out = self.lin(out) + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class FPGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FPGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, out_channels)  
        self.conv1 = FPLayer(in_channels, hidden_channels)
        self.conv2 = FPLayer(hidden_channels, hidden_channels)
        self.conv_last = FPLayer(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

                    

    def forward(self, edge_index, edge_weight, x, M):
        x_hat = torch.zeros_like(x)

        x_hat = self.conv1(x, x_hat, edge_index, edge_weight, M)
        x_hat = x_hat.relu_()
        x_hat = self.conv2(x, x_hat, edge_index, edge_weight, M)
        x_hat = x_hat.relu_()
        x_hat = self.conv_last(x, x_hat, edge_index, edge_weight, M)
        x_hat = x_hat.relu_()
        x_hat = self.fc(x_hat)


        return x_hat
    