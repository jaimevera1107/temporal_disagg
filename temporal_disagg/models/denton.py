import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase

class Denton:
    def estimate(self, y_l, X, C, h=1):
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        D = np.eye(n) - np.diag(np.ones(n - 1), -1)
        D_h = np.linalg.matrix_power(D, h) if h > 0 else np.eye(n)
        Sigma_D = pinv(D_h.T @ D_h)
        D_matrix = Sigma_D @ C.T @ pinv(C @ Sigma_D @ C.T)
        u_l = y_l - C @ X
        return X + D_matrix @ u_l