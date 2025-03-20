import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase
from ..optimization import RhoOptimizer

class DynamicChowLin:
    def estimate(self, y_l, X, C):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="maxlog")
        n = X.shape[0]
        Sigma_CL = (1 / (1 - rho_opt**2)) * toeplitz(np.ravel(rho_opt ** np.arange(n)))
        Q = C @ Sigma_CL @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_CL @ C.T @ inv_Q
        u_l = y_l - C @ p
        return (p + D @ u_l).flatten()
    
class DynamicLitterman:
    def estimate(self, y_l, X, C):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="minrss")
        n = len(X)
        H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho_opt
        Sigma_L = pinv(H.T @ H)
        Q = C @ Sigma_L @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_L @ C.T @ inv_Q
        u_l = y_l - C @ p
        
        return (p + D @ u_l).flatten()