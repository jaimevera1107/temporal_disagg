import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed
from ..base import TempDisBase

class OLS:
    def estimate(self, y_l, X, C):
        y_l, X, C = TempDisBase().preprocess_inputs(y_l, X, C)
        X_l = np.atleast_2d(C @ X)
        beta = pinv(X_l.T @ X_l) @ X_l.T @ y_l
        return (X @ beta).flatten()