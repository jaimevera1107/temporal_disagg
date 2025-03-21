import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed


class CovMatrixBuilder:
    """
    Class responsible for constructing covariance and penalty matrices used
    in temporal disaggregation methods such as Chow-Lin and Litterman.
    """

    def power_matrix_calculation(self, n):
        """
        Computes the power matrix (distance matrix) used to build correlation structures.

        Parameters:
            n (int): Size of the matrix.

        Returns:
            np.ndarray: Matrix of absolute differences between index positions.
        """
        return np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    def q_calculation(self, rho, pm, min_rho_boundarie=0, max_rho_boundarie=1):
        """
        Calculates the Q matrix used in Chow-Lin models.

        Parameters:
            rho (float): Autocorrelation parameter.
            pm (np.ndarray): Power matrix (distance matrix).
            min_rho_boundarie (float): Minimum bound for rho.
            max_rho_boundarie (float): Maximum bound for rho.

        Returns:
            np.ndarray: Covariance matrix Q.
        """
        epsilon = 1e-6
        rho = np.clip(rho, min_rho_boundarie, max_rho_boundarie)
        factor = 1 / (1 - rho**2 + epsilon)
        return factor * (rho ** pm)

    def q_lit_calculation(self, X, rho=0):
        """
        Calculates the Q matrix used in Litterman-based models.

        Parameters:
            X (np.ndarray): High-frequency indicator matrix.
            rho (float): Smoothing parameter (typically between 0 and 1).

        Returns:
            np.ndarray: Inverted Q matrix used for smoothing.
        """
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
