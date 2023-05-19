import torch
import numpy as np
from scipy.optimize import minimize

def NBS(adj, X, feature_mask):
    L = torch.diag(torch.sum(adj, axis=1)) - adj
    L = L.numpy()

    gamma = 1e-3

    X_reconstructed = torch.zeros_like(X)
    for i in range(feature_mask.shape[1]):
        mask = feature_mask[:, i].cpu().numpy()
        x0 = X[:, i].cpu().numpy()
        S = torch.eye(x0.shape[0])[mask].float()
        S = S.numpy()
        fun = lambda x: np.linalg.norm(S @ x - S @ x0) ** 2 + gamma * np.linalg.norm(L @ x) ** 2
        res = minimize(fun, x0, method='BFGS', options={'maxiter': 200})
        X_reconstructed[:, i] = torch.tensor(res.x)

    X_reconstructed[feature_mask] = X[feature_mask]
    return X_reconstructed

