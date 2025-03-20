import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed

class CovMatrixBuilder:
    def power_matrix_calculation(self, n):
        return np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    def q_calculation(self, rho, pm, min_rho_boundarie=0, max_rho_boundarie=1):
        epsilon = 1e-6
        rho = np.clip(rho, min_rho_boundarie, max_rho_boundarie)
        factor = 1 / (1 - rho**2 + epsilon)
        return factor * (rho ** pm)

    def q_lit_calculation(self, X, rho=0):
        n = X.shape[0]
        epsilon = 1e-8
        H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho
        D = np.eye(n) - np.diag(np.ones(n - 1), -1)
        Q_Lit = D.T @ H.T @ H @ D
        try:
            Q_Lit_inv = np.linalg.inv(Q_Lit + np.eye(n) * epsilon)
        except np.linalg.LinAlgError:
            Q_Lit_inv = np.linalg.pinv(Q_Lit)
        return Q_Lit_inv
