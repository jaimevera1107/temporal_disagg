import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase


class OLS:
    """
    Ordinary Least Squares method for temporal disaggregation.

    This method applies standard OLS estimation using the aggregated indicators
    and low-frequency targets without incorporating temporal correlation or smoothing.
    """

    def estimate(self, y_l, X, C):
        """
        Estimates the high-frequency series using an OLS-based approach.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        X_l = np.atleast_2d(C @ X)
        beta = pinv(X_l.T @ X_l) @ X_l.T @ y_l
        return (X @ beta).flatten()
