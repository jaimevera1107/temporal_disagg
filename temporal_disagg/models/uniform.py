import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase


class Uniform:
    """
    Uniform method for temporal disaggregation.

    Applies a uniform weighting structure without imposing temporal correlation,
    adjusting the residuals proportionally across high-frequency periods.
    """

    def estimate(self, y_l, X, C):
        """
        Estimates the high-frequency series using a uniform disaggregation approach.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        Sigma_U = np.eye(n)
        D_matrix = Sigma_U @ C.T @ pinv(C @ Sigma_U @ C.T)
        u_l = y_l - C @ X
        adjusted_u_l = D_matrix @ u_l

        return X + adjusted_u_l
