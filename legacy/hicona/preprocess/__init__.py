"""
``hicona.processing``
=====================

This module contains all the functions required to convert a raw pixel
table into a sparsified one ready to be analyzed.

"""

_DEFAULT_FLOWS: str = "flows.json"
_FUNCS_MODULES: list[str] = [
    "hicona.preprocess._filt",
    "hicona.preprocess._norm",
    "hicona.preprocess._spar",
]

from ._abcs import FiltOperation, NormOperation, SparOperation
from ._filt import *
from ._flow import Flow
from ._norm import *
from ._spar import *
