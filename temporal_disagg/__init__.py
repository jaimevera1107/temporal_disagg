"""
TemporalDisagg: A Python package for temporal disaggregation of time series.

Provides multiple econometric and statistical methods to estimate high-frequency series
from low-frequency aggregates, including Chow-Lin, Denton, Litterman, OLS-based models,
retropolation techniques, and ensemble forecasting tools.
"""

import pandas as pd
import numpy as np
from scipy.linalg import toeplitz, pinv, solve
from scipy.optimize import minimize_scalar, minimize
from joblib import Parallel, delayed

from .preprocessing import TimeSeriesPreprocessor
from .base import TempDisBase
from .estimation import TempDisModel
from .conversion_matrix import CovMatrixBuilder
from .optimization import RhoOptimizer
from .aggregation import TemporalAggregation
from .retropolation import Retropolation

from .models.chow_lin import ChowLin, ChowLinFixed, ChowLinOpt, ChowLinEcotrim, ChowLinQuilis
from .models.denton import Denton
from .models.denton_cholette import DentonCholette
from .models.dynamic_models import DynamicChowLin, DynamicLitterman
from .models.fast import Fast
from .models.fernandez import Fernandez
from .models.litterman import Litterman, LittermanOpt
from .models.ols import OLS
from .models.uniform import Uniform

__all__ = [
    "TimeSeriesPreprocessor",
    "TempDisBase",
    "TempDisModel",
    "CovMatrixBuilder",
    "RhoOptimizer",
    "TemporalAggregation",
    "Retropolation",
    "ChowLin",
    "ChowLinFixed",
    "ChowLinOpt",
    "ChowLinEcotrim",
    "ChowLinQuilis",
    "Denton",
    "DentonCholette",
    "DynamicChowLin",
    "DynamicLitterman",
    "Fast",
    "Fernandez",
    "Litterman",
    "LittermanOpt",
    "OLS",
    "Uniform"
]
