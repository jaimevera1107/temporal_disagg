"""
Models for temporal disaggregation.

This submodule includes various statistical and econometric methods to estimate
high-frequency series from low-frequency aggregates. Each model implements the 
`estimate()` method and follows a standardized interface.
"""

from .chow_lin import ChowLin, ChowLinFixed, ChowLinOpt, ChowLinEcotrim, ChowLinQuilis
from .denton import Denton
from .denton_cholette import DentonCholette
from .dynamic_models import DynamicChowLin, DynamicLitterman
from .fast import Fast
from .fernandez import Fernandez
from .litterman import Litterman, LittermanOpt
from .ols import OLS
from .uniform import Uniform

__all__ = [
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
