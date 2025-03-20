import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase
from ..optimization import RhoOptimizer

class ChowLin:
    def estimate(self, y_l, X, C, rho=0.5):
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho = np.clip(rho, -0.9, 0.99)
        Sigma_CL = (1 / (1 - rho**2)) * toeplitz((rho ** np.arange(n)).ravel())
        Q = C @ Sigma_CL @ C.T
        inv_Q = pinv(Q)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_CL @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l
    
class ChowLinFixed:
    def estimate(self, y_l, X, C, rho=0.9):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho = np.clip(rho, -0.9, 0.99)
        n = len(X)
        Sigma_CL = (1 / (1 - rho**2)) * toeplitz(np.ravel(rho ** np.arange(n)))
        Q = C @ Sigma_CL @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_CL @ C.T @ inv_Q
        u_l = y_l - C @ p
        return (p + D @ u_l).flatten()

class ChowLinOpt:
    def estimate(self, y_l, X, C):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="maxlog")
        return ChowLin().estimate(y_l, X, C, rho_opt)


class ChowLinEcotrim:
    def estimate(self, y_l, X, C, rho=0.75):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        n = X.shape[0]
        rho = np.clip(rho, -0.9, 0.99)
        R = toeplitz(rho ** np.arange(n))
        Q = C @ R @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = R @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l

class ChowLinQuilis:
    def estimate(self, y_l, X, C, rho=0.15):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        n = X.shape[0]
        rho = np.clip(rho, -0.9, 0.99)
        epsilon = 1e-6
        R = (1 / (1 - (rho + epsilon)**2)) * toeplitz(rho ** np.arange(n))
        Q = C @ R @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = R @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l