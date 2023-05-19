import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add


import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt


class FPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, mask_reverse=False):
        super().__init__()  
        self.lin = Linear(in_channels, out_channels)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.mask_reverse = mask_reverse
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, M):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm=norm) * M + x * (~M)
        if self.mask_reverse:
            out = self.propagate(edge_index, x=out, norm=norm) * (~M) + x * M
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
    

class FPGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, reverse=True):
        super(FPGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = FPLayer(in_channels, hidden_channels, reverse)
        self.conv2 = FPLayer(hidden_channels, hidden_channels, reverse)

    def forward(self, x, edge_index, M):
        x = self.conv1(x, edge_index, M)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, M)
        return x
    
def run_model(model, X, edge_index, mask, epochs=200, lr=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X, edge_index, mask)
        loss = criterion(out[mask], X[mask])
        loss.backward()
        optimizer.step()
    model.eval()
    out = model(X, edge_index, mask)
    out[mask] = X[mask]
    return out

class FeaturePropagation(torch.nn.Module):
    def __init__(self, num_iterations: int):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations

    def propagate(self, x: Tensor, edge_index: Adj, mask: Tensor) -> Tensor:
        # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
        # value at convergence
        out = x
        if mask is not None:
            out = torch.zeros_like(x)
            out[mask] = x[mask]

        n_nodes = x.shape[0]
        adj = self.get_propagation_matrix(out, edge_index, n_nodes)
        for _ in range(self.num_iterations):
            # Diffuse current features
            out = torch.sparse.mm(adj, out)
            # Reset original known features
            out[mask] = x[mask]

        return out

    def get_propagation_matrix(self, x, edge_index, n_nodes):
        # Initialize all edge weights to ones if the graph is unweighted)
        edge_index, edge_weight = get_symmetrically_normalized_adjacency(edge_index, n_nodes=n_nodes)
        adj = torch.sparse.FloatTensor(edge_index, values=edge_weight, size=(n_nodes, n_nodes)).to(edge_index.device)

        return adj
        

def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)


def get_symmetrically_normalized_adjacency(edge_index, n_nodes):
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, DAD



def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, missing_rate=0.99, adj=None
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]

    assert len(train_data.shape) == 2, "The input data should be a 2D matrix (time_len, n_nodes)"

    n_nodes = train_data.shape[1]
    F = train_data.shape[0]
    mask = torch.bernoulli(torch.Tensor([1 - missing_rate]).repeat(n_nodes, F)).bool()


    # FP 
    # get the edge index from the adjacency matrix
    adj = torch.from_numpy(adj)
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    train_data = train_data.T
    train_data = torch.from_numpy(train_data)
    X_reconstructed = []
    for i in range(train_data.shape[0]):
        propagation_model = FPGCN(1, 1, 1)
        pred = run_model(propagation_model, train_data[i].unsqueeze(1), edge_index, mask[i, :].unsqueeze(), epochs=200, lr=0.9)
        print(pred.shape)
        X_reconstructed.append(pred.squeeze(1))

    train_data = torch.stack(X_reconstructed)


    # train_data = feature_propagation(edge_index, train_data, mask, num_iterations=40)
    # # to numpy
    train_data = train_data.cpu().detach().numpy()
    train_data = train_data.T





    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, adj=None
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
        adj=adj,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
