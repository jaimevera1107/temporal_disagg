import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase


class DentonCholette:
    """
    Denton-Cholette method for temporal disaggregation.

    This method applies a smoothing transformation to the residuals between
    the target aggregate and the extrapolated high-frequency series, minimizing
    the volatility of changes while preserving the low-frequency constraints.
    """

    def estimate(self, y_l, X, C, h=1):
        """
        Estimates the high-frequency series using the Denton-Cholette method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            h (int): Degree of differencing used for penalization (default: 1).

        Returns:
            np.ndarray: High-frequency estimate adjusted for smoothness and coherence.
        """
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        D = np.eye(n) - np.diag(np.ones(n - 1), -1)
        D_h = np.linalg.matrix_power(D, h) if h > 0 else np.eye(n)
        Sigma_D = pinv(D_h.T @ D_h)
        D_matrix = Sigma_D @ C.T @ pinv(C @ Sigma_D @ C.T)
        u_l = y_l - C @ X
        adjusted_u_l = D_matrix @ u_l

        return X + adjusted_u_l
