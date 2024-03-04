"""
``hicona.processing``
=====================

This module contains all the functions required to pass from a raw pixel
table to a sparsified one ready to be analyzed.

All filtering and normalization functions share a common blueprint:
- a pixel table object must always be provided as first argument
- other arguments might be present (but not always)
- an iterator of processed pixel chunks is returned
- defult arguments should be JSON types
"""

from .processing_flow import *
