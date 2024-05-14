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

from ._flow import Flow
from ._filt import *
from ._norm import *
