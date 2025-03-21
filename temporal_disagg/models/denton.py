import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase


class Denton:
    """
    Classic Denton method for temporal disaggregation.

    Minimizes the volatility of period-to-period changes in the adjusted
    high-frequency series while ensuring consistency with the low-frequency aggregates.
    """

    def estimate(self, y_l, X, C, h=1):
        """
        Estimates the high-frequency series using the Denton method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            h (int): Degree of differencing for volatility penalty (default: 1).

        Returns:
            np.ndarray: High-frequency estimate that preserves low-frequency totals.
        """
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        D = np.eye(n) - np.diag(np.ones(n - 1), -1)
        D_h = np.linalg.matrix_power(D, h) if h > 0 else np.eye(n)
        Sigma_D = pinv(D_h.T @ D_h)
        D_matrix = Sigma_D @ C.T @ pinv(C @ Sigma_D @ C.T)
        u_l = y_l - C @ X
        return X + D_matrix @ u_l
