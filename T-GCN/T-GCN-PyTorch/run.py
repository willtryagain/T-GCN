import argparse
import multiprocessing
import time

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import torch.nn as nn


from imputers.feature_propgate import feature_propagation
from imputers.my_imputer2 import FPGCN 
from compare_FP import train_model, test_model
torch.manual_seed(0)

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}


def get_product_graph(mat, T):
    N = mat.shape[0]
    mat_T = mat[:, :T].clone()
    nodes = torch.arange(N) 

    edge_index_T = []
    for i in range(T):
        cur_edge_index = edge_index + i * N
        edge_index_T.append(cur_edge_index)

    for i in range(T-1):
        cur_edge_index = torch.stack([nodes + i * N, nodes + (i + 1) * N], dim=0)
        edge_index_T.append(cur_edge_index)
        cur_edge_index = torch.stack([nodes + (i + 1) * N, nodes + i * N], dim=0)
        edge_index_T.append(cur_edge_index)
    
    edge_index_T = torch.cat(edge_index_T, dim=1)
    edge_weight_T = torch.ones(edge_index_T.shape[1])
    mat_T = mat_T.T.flatten().unsqueeze(dim=1)

    return edge_index_T, edge_weight_T, mat_T



def create_mask(shape, k):
    np.random.seed(0)

    total = shape[0] * shape[1]
    num_zeros = int(total * k / 100)
    indices = np.random.choice(total, num_zeros, replace=False)
    mask = np.ones((shape[0], shape[1]), dtype=bool)
    mask.ravel()[indices] = 0
    return torch.from_numpy(mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="shenzhen")
    parser.add_argument("--T", type=int, default=0)
    args = parser.parse_args()

    data = args.data
    mat = pd.read_csv(DATA_PATHS[data]["feat"]).to_numpy() 
    adj = pd.read_csv(DATA_PATHS[data]["adj"], header=None).to_numpy()
    mat = torch.tensor(mat).float()
    adj = torch.tensor(adj)
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    # edge_weight = torch.ones(edge_index.shape[1])
    edge_weight = adj[edge_index[0], edge_index[1]].float()
    if args.T > 0:
        mat = mat.T[:, :args.T]
    else: 
        args.T = mat.shape[1]
    results = {
        "fp": [],
        "product fp_gcn": [],
        "product fp": [],
    }

    loss_fn = nn.MSELoss()
    lr = 0.9
    rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, .8, .9]
    mat_og = mat.clone()

    edge_index_T, edge_weight_T, mat_T = get_product_graph(mat, args.T)

    for missing_rate in  rates:
        start = time.time()
        mask = create_mask(mat.shape, missing_rate * 100)
        X = torch.zeros_like(mat)
        X[mask] = mat[mask].clone()

        X_reconstructed = feature_propagation(edge_index, edge_weight, X, mask, 40)
        # save inputs in file
     
        norm_mse = torch.norm(X_reconstructed - mat) / torch.norm(mat)
        results["fp"].append(norm_mse)

        mask_T = mask.T.flatten().unsqueeze(dim=1)
        mask_T_og = mask_T.clone()
        X = torch.zeros_like(mat_T)
        X[mask_T] = mat_T[mask_T].clone()
        X_reconstructed = feature_propagation(edge_index_T, edge_weight_T, X, mask_T, 40)
        norm_mse = torch.norm(X_reconstructed - mat_T) / torch.norm(mat_T)
        results["product fp"].append(norm_mse)

        model = FPGCN(1, 1, 1)
        mask_T = mask.T.flatten().unsqueeze(dim=1)
        train_model(model, X, edge_index_T, edge_weight_T, mask_T, epochs=1000, lr=lr, loss_fn=loss_fn)
        X_pred = test_model(model, X, edge_index_T, edge_weight_T, mask_T)

        norm_mse = torch.norm(X_pred - mat_T) / torch.norm(mat_T)   
        results["product fp_gcn"].append(norm_mse)

        end = time.time()
        print("time", end - start)
        print("missing rate", missing_rate)
        # assert torch.allclose(mat, mat_og)
        # assert torch.allclose(mat.T.flatten().unsqueeze(dim=1), mat_T)
        # assert torch.allclose(mask_T, mask_T_og)

    for key in results:
        print(key, results[key])
    plt.title(args.data)
    for key in results:
        plt.plot(rates, results[key], label=key)
        plt.xlabel("missing rate")
        plt.ylabel("Normalized RMSE")

    plt.legend()
    plt.savefig(args.data + ".png")
