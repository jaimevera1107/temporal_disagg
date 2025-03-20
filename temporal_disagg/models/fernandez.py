import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase

class Fernandez:
    def estimate(self, y_l, X, C):
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        Delta = np.eye(n) - np.diag(np.ones(n - 1), -1)
        Sigma_F = np.linalg.inv(Delta.T @ Delta)
        Q = C @ Sigma_F @ C.T
        inv_Q = np.linalg.inv(Q)
        beta = solve(X.T @ C.T @ inv_Q @ C @ X, X.T @ C.T @ inv_Q @ y_l).reshape(-1, 1)
        p = X @ beta
        D = Sigma_F @ C.T @ inv_Q
        u_l = y_l - C @ p
        return (p + D @ u_l).flatten()