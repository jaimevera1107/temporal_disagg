import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase

class Fast:
    def estimate(self, y_l, X, C):
        rho = 0.9
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho
        Sigma_F = pinv(H.T @ H)
        Q = C @ Sigma_F @ C.T
        inv_Q = pinv(Q)
        beta = solve(X.T @ C.T @ inv_Q @ C @ X, X.T @ C.T @ inv_Q @ y_l)
        p = X @ beta
        D = Sigma_F @ C.T @ inv_Q
        u_l = y_l - C @ p
        return (p + D @ u_l).flatten()