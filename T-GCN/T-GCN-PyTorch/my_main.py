from models import MyGCN
import numpy as np
import torch


if __name__ == "__main__":
    adj = np.random.rand(10, 10)
    model = MyGCN(adj, 10, 10)
    inputs = torch.randn(64, 10, 10)
    outputs = model(inputs)