import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase


class Fernandez:
    """
    Fernandez method for temporal disaggregation.

    Uses an autoregressive-like structure to extrapolate high-frequency estimates
    from a low-frequency target while minimizing the second-order differences.
    """

    def estimate(self, y_l, X, C):
        """
        Estimates the high-frequency series using the Fernandez method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.

        Returns:
            np.ndarray: High-frequency estimate.
        """
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
