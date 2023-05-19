import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class NBSLoss(nn.Module):
    def __init__(self, gamma = 1e-3) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, inp, target, mask, poly):
        mse = F.mse_loss(inp[mask], target[mask], reduction='sum')
        regularizer = torch.sum((poly @ inp) ** 2)
        return mse + self.gamma * regularizer
        
class NBS_Approx(nn.Module):
    # input W (n_nodes, n_features)
    # output Y (n_nodes, n_features)
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        (n_nodes, n_features) = x.shape
        x = x.flatten()
        x = self.lin1(x)
        x = torch.relu(x)

        x = self.lin2(x)
        x = x.reshape(n_nodes, n_features)
        return x


def NBS_approx(adj, X, feature_mask):
    L = torch.diag(torch.sum(adj, axis=1)) - adj
    gamma = 1e-3

    n_nodes, n_features = X.shape
    inp_dim = n_features * n_nodes
    n_hidden = n_features // 2

    model = NBS_Approx(inp_dim, n_hidden, inp_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = NBSLoss(gamma=gamma)
    print("Start training")
    print("X shape", X.shape)
    for _ in range(100):
        optimizer.zero_grad()
        out = model(X)
        print("out shape", out.shape)
        l = loss(out, X, feature_mask, L)
        l.backward()
        optimizer.step()
    return out.detach()
