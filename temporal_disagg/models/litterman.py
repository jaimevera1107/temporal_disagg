import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase
from ..optimization import RhoOptimizer


class Litterman:
    """
    Litterman method for temporal disaggregation.

    Incorporates a fixed smoothing parameter (rho) into the estimation of the
    high-frequency series using a state-space like formulation.
    """

    def estimate(self, y_l, X, C, rho=0.5):
        """
        Estimates the high-frequency series using the Litterman method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            rho (float): Smoothing parameter controlling temporal correlation.

        Returns:
            np.ndarray: High-frequency estimate.
        """
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
    """
    Litterman method with automatic optimization of the smoothing parameter.

    Selects the rho value that minimizes the residual sum of squares,
    and then applies the standard Litterman estimation.
    """

    def estimate(self, y_l, X, C):
        """
        Estimates the high-frequency series using the optimized Litterman method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="minrss")
        return Litterman().estimate(y_l, X, C, rho_opt)
