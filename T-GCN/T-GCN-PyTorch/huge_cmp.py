import pandas as pd
import numpy as np
from imputers.my_imputer2 import FPGCN 
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
    


df = pd.read_hdf('pems_bay/pems_bay.h5')
# to numpy 
data = df.to_numpy()
data.shape

vals = df.columns.values
# construct a map from index to sensor id
sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(vals)}
adj_df= pd.read_csv('pems_bay/distances_bay.csv')

# obtain edges
edges = []
weights = []
for row in adj_df.values:
    if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
        continue
    edges.append((sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]))
    weights.append(row[2])

edges = np.array(edges)
weights = np.array(weights)
edges.shape



def create_mask(shape, k):
    np.random.seed(0)

    total = shape[0] * shape[1]
    num_zeros = int(total * k / 100)
    indices = np.random.choice(total, num_zeros, replace=False)
    mask = np.ones((shape[0], shape[1]), dtype=bool)
    mask.ravel()[indices] = 0
    return torch.from_numpy(mask)


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


mat = torch.tensor(data).float().T
edge_index = torch.tensor(edges).long().T
edge_weight = torch.tensor(weights).float()
# invert and set 0 to 0
edge_weight = 1 / edge_weight
edge_weight[edge_weight == float("inf")] = torch.max(edge_weight[edge_weight != float("inf")]) + 1

edge_index_T, edge_weight_T, mat_T = get_product_graph(mat, mat.shape[1])

# construct a mask
missing_rate= 25
mask = create_mask(mat.shape, missing_rate)
X = torch.zeros_like(mat)
X[mask] = mat[mask].clone()
X_reconstructed = feature_propagation(edge_index, edge_weight, X, mask, 40)
mse = torch.nn.functional.mse_loss(X_reconstructed[~mask], mat[~mask])
print(mse)

mask_T = mask.T.flatten().unsqueeze(dim=1)
X = torch.zeros_like(mat_T)
X[mask_T] = mat_T[mask_T].clone()
X_reconstructed = feature_propagation(edge_index_T, edge_weight_T, X, mask_T, 40)
mse = torch.nn.functional.mse_loss(X_reconstructed[~mask_T], mat_T[~mask_T])
print(mse)


loss_fn = nn.MSELoss()
lr = 0.9
model = FPGCN(1, 1, 1)
train_model(model, X, edge_index_T, edge_weight_T, mask_T, epochs=1000, lr=lr, loss_fn=loss_fn)
X_pred = test_model(model, X, edge_index_T, edge_weight_T, mask_T)
mse = torch.nn.functional.mse_loss(X_pred[~mask], mat[~mask])
print(mse)