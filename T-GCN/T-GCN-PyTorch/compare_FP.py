import argparse
import multiprocessing
import subprocess

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
import time
import matplotlib.pyplot as plt


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
torch.manual_seed(0)


DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}


stats = {
    "FP": 0,
    "NBS": 0,
    "FPGCN": 0,
}

class SpectralLoss(nn.Module):
    def __init__(self, gamma, filter) -> None:
        super().__init__()
        self.gamma = gamma
        self.filter = filter

    def forward(self, inp, target, mask):
        # print(self.filter.dtype, inp.dtype, target.dtype)
        mse = F.mse_loss(inp[mask], target[mask])
        # 
        regularizer = torch.sum((self.filter @ inp) ** 2)
        return mse + self.gamma * regularizer


def train_model(model, X, edge_index, edge_weight, mask, epochs=200, lr=1, loss_fn=nn.MSELoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = loss_fn
    model.train()
    patience = 10
    wait = 0
    min_loss = float("inf")

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(edge_index, edge_weight, X, mask)
        # loss = criterion(out, X, mask)
        loss = criterion(out[mask], X[mask])
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
            wait = 0
        else:
            wait += 1
            if wait > patience:
                # print("Early stopping", epoch)
                break
        # print("Epoch: ", epoch, "Loss: ", loss.item())

def test_model(model, X, edge_index, edge_weight, mask):
    model.eval()
    out = model(edge_index, edge_weight, X, mask).detach()
    out[mask] = X[mask]
    return out
    

def sequential_FPGCN(model, X, edge_index, edge_weight, mask, laplacian, epochs=200, lr=1, gamma=1e-2):
    model = model(1, 1, 1)
    # laplacian from edge_index

    X_hat = X.clone()
    X_hat[~mask] = 0
    # loss_fn = SpectralLoss(gamma, laplacian)

    for i in range(X.shape[1]):
        train_model(model, X[:, i].unsqueeze(1), edge_index, edge_weight, mask[:, i].unsqueeze(1), epochs, lr)
    
    for i in range(X.shape[1]):
        out = test_model(model, X[:, i].unsqueeze(1), edge_index, edge_weight, mask[:, i].unsqueeze(1)).detach()
        X_hat[:, i] = out[:, 0]
    return X_hat


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="shenzhen")
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--num_cols", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(0)

    mat = pd.read_csv(DATA_PATHS[args.data]["feat"]).to_numpy() 
    adj = pd.read_csv(DATA_PATHS[args.data]["adj"], header=None).to_numpy()
    mat = torch.tensor(mat).float()
    adj = torch.tensor(adj)
    laplacian = torch.diag(torch.sum(adj, dim=1)) - adj
    laplacian = laplacian.float()
    deg = torch.diag(torch.sum(adj, dim=1)).float()
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    deg_inv_sqrt = deg_inv_sqrt.float()
    laplacian = deg_inv_sqrt @ laplacian @ deg_inv_sqrt
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    edge_weight = torch.ones(edge_index.shape[1])
    for i in range(edge_index.shape[1]):
        edge_weight[i] = adj[edge_index[0][i]][edge_index[1][i]]
    mat = mat.T
    if args.num_cols > 0:
        mat = mat[:, :args.num_cols]

    overall_results = {
        "gcn1": [],
        "gcn2": [],
        "fp": [],
    }

    MIN_MISSING_RATE = 0.8
    MAX_MISSING_RATE = 0.8

    for missing_rate in np.arange(MIN_MISSING_RATE, MAX_MISSING_RATE + 0.1, 0.1):
    
        mask = torch.bernoulli(torch.Tensor([1 - missing_rate]).repeat(mat.shape)).bool()

        start = time.time()
        X_reconstructed = feature_propagation(edge_index, edge_weight, mat, mask, 40)
        end = time.time()
        norm_mse = torch.norm(X_reconstructed - mat) / torch.norm(mat)
        overall_results["fp"].append(norm_mse)
        print(norm_mse)
        print("FP time: ", end - start)

        lr = 1e-1

        start = time.time()
        model = FPGCN
        X_reconstructed = sequential_FPGCN(model, mat, edge_index, edge_weight, mask, laplacian, 1000, lr)
        end = time.time()
        norm_mse = torch.norm(X_reconstructed - mat) / torch.norm(mat)
        overall_results["gcn2"].append(norm_mse)
        print(norm_mse)
        print("FPGCN2 time: ", end - start)
        s = "FPGCN2 time: " + str(end - start) + " norm_mse: " + str(norm_mse)
        subprocess.call('echo ' + s, shell=True)


        # num_processes = multiprocessing.cpu_count()
        # mat_split = np.array_split(mat, num_processes, axis=1)
        # mask_split = np.array_split(mask, num_processes, axis=1)

    


        # pool = multiprocessing.Pool(processes=num_processes)
        # results = pool.starmap(sequential_FPGCN, [(mat_split[i], edge_index, mask_split[i], 200, 0.9) for i in range(num_processes)])
        # pool.close()
        # pool.join()
        # end = time.time()
        # print("FPGCN time: ", end - start)
        # X_reconstructed = torch.cat(results, dim=1)
        # start = time.time()


            


    plt.title(args.data)
    # plt.plot(np.arange(0.1, 1, 0.1), overall_results["gcn1"], label="GCN1")
    plt.plot(np.arange(MIN_MISSING_RATE, MAX_MISSING_RATE + 0.1, 0.1), overall_results["gcn2"], label="GCN2")
    plt.plot(np.arange(MIN_MISSING_RATE, MAX_MISSING_RATE + 0.1, 0.1), overall_results["fp"], label="Feature Propagation")
    plt.xlabel("missing rate")
    plt.ylabel("Normalized RMSE")

    plt.legend()
    plt.savefig(args.data + ".png")
    stats = str(stats)
    # save stats
    np.save(args.data + ".npy", stats)
    subprocess.call('echo ' + stats, shell=True)

    












