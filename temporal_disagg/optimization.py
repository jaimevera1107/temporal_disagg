import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from .base import TempDisBase
from .conversion_matrix import CovMatrixBuilder

class RhoOptimizer:
    def rho_optimization(self, y_l, X, C, method="maxlog", min_rho = -1.999, max_rho = 1.999):
        X_l = np.atleast_2d(C @ X)
        pm = CovMatrixBuilder().power_matrix_calculation(X.shape[0])
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)

        def objective(rho):
            if not (min_rho < rho < max_rho):
                return np.inf
            Q = CovMatrixBuilder().q_calculation(rho, pm)
            vcov = C @ Q @ C.T
            inv_vcov = pinv(vcov + np.eye(vcov.shape[0]) * 1e-8)
            XTX = X_l.T @ inv_vcov @ X_l
            if XTX.shape[0] != XTX.shape[1]:
                return np.inf
            beta = pinv(XTX) @ X_l.T @ inv_vcov @ y_l
            u_l = y_l - X_l @ beta
            if method == "maxlog":
                return -(-0.5 * (np.log(np.abs(np.linalg.det(vcov)) + 1e-8) + u_l.T @ inv_vcov @ u_l))
            elif method == "minrss":
                return u_l.T @ inv_vcov @ u_l
            else:
                raise ValueError("Invalid method for rho calculation")

        opt_result = minimize_scalar(objective, bounds=(min_rho, max_rho), method="bounded")
        return opt_result.x