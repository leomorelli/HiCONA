"""
``hicona.processing``
=====================

This module contains all the functions required to pass from a raw pixel
table to a sparsified one ready to be analyzed.

All filtering and normalization functions share a common blueprint:
- a pixel table object must always be provided as first argument
- other arguments might be present (but not always)
- an iterator of processed pixel chunks is returned
- default arguments should be JSON types

"""

_DEFAULT_FLOWS: str = "flows.json"
_FUNCS_MODULES: list[str] = ["hicona.preprocess._filt", "hicona.preprocess._norm"]

from ._abcs import FiltOperation, NormOperation
from ._filt import *
from ._norm import *
from ._flow import Flow
