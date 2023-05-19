import torch
import numpy as np


def least_square(x, mask_, adj):
    adj = adj.float()
    # eigendecomposition
    L = torch.diag(torch.sum(adj, dim=1)) - adj
    # symmetric normalization
    D = torch.diag(torch.pow(torch.sum(adj, dim=1), -0.5))
    L = D @ L @ D
    eigval, eigvec = torch.linalg.eigh(L)

    # B
    k = 2
    B = eigvec[mask_, :k].float()

    # reconstruction
    alpha = torch.linalg.lstsq(B, x[mask_])[0].float()
    x_reconstructed = eigvec[:, :k] @ alpha
    return x_reconstructed


def sequential_least_square(x, mask, adj):
    x = x.clone()
    for i in range(x.shape[1]):
        x[:, i] = least_square(x[:, i], mask[:, i], adj)
    return x





