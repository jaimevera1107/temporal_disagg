import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase
from ..optimization import RhoOptimizer

class Litterman:
    def estimate(self, y_l, X, C, rho=0.5):
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho = np.clip(rho, -0.9, 0.99)
        H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho
        Sigma_L = pinv(H.T @ H)
        Q = C @ Sigma_L @ C.T
        inv_Q = pinv(Q)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_L @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l
    
class LittermanOpt:
    def estimate(self, y_l, X, C):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="minrss")
        return Litterman().estimate(y_l, X, C, rho_opt)