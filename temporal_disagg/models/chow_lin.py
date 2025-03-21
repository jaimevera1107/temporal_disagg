import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase
from ..optimization import RhoOptimizer


class ChowLin:
    """
    Chow-Lin method for temporal disaggregation using a fixed rho value.
    """

    def estimate(self, y_l, X, C, rho=0.5):
        """
        Estimates the high-frequency series using the Chow-Lin method.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            rho (float): Autocorrelation parameter.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        n = len(X)
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho = np.clip(rho, -0.9, 0.99)
        Sigma_CL = (1 / (1 - rho**2)) * toeplitz((rho ** np.arange(n)).ravel())
        Q = C @ Sigma_CL @ C.T
        inv_Q = pinv(Q)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_CL @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l


class ChowLinFixed:
    """
    Chow-Lin method using a fixed rho value and increased numerical stability.
    """

    def estimate(self, y_l, X, C, rho=0.9):
        """
        Estimates the high-frequency series using a fixed rho.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            rho (float): Fixed autocorrelation parameter.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho = np.clip(rho, -0.9, 0.99)
        n = len(X)
        Sigma_CL = (1 / (1 - rho**2)) * toeplitz(np.ravel(rho ** np.arange(n)))
        Q = C @ Sigma_CL @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = Sigma_CL @ C.T @ inv_Q
        u_l = y_l - C @ p
        return (p + D @ u_l).flatten()


class ChowLinOpt:
    """
    Chow-Lin method with automatic optimization of the rho parameter.
    """

    def estimate(self, y_l, X, C):
        """
        Optimizes rho using log-likelihood and applies Chow-Lin disaggregation.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        rho_opt = RhoOptimizer().rho_optimization(y_l, X, C, method="maxlog")
        return ChowLin().estimate(y_l, X, C, rho_opt)


class ChowLinEcotrim:
    """
    Chow-Lin variant based on Ecotrim correlation structure.
    """

    def estimate(self, y_l, X, C, rho=0.75):
        """
        Estimates high-frequency series using the Ecotrim variant.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            rho (float): Autocorrelation parameter.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        n = X.shape[0]
        rho = np.clip(rho, -0.9, 0.99)
        R = toeplitz(rho ** np.arange(n))
        Q = C @ R @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = R @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l


class ChowLinQuilis:
    """
    Chow-Lin variant based on Quilis formulation with regularization.
    """

    def estimate(self, y_l, X, C, rho=0.15):
        """
        Estimates high-frequency series using the Quilis variant.

        Parameters:
            y_l (np.ndarray): Low-frequency target series.
            X (np.ndarray): High-frequency indicator series.
            C (np.ndarray): Conversion matrix.
            rho (float): Autocorrelation parameter.

        Returns:
            np.ndarray: High-frequency estimate.
        """
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        n = X.shape[0]
        rho = np.clip(rho, -0.9, 0.99)
        epsilon = 1e-6
        R = (1 / (1 - (rho + epsilon)**2)) * toeplitz(rho ** np.arange(n))
        Q = C @ R @ C.T
        inv_Q = pinv(Q + np.eye(Q.shape[0]) * 1e-8)
        beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
        p = X @ beta
        D = R @ C.T @ inv_Q
        u_l = y_l - C @ p
        return p + D @ u_l
