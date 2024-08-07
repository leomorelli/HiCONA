"""
Hicona
======

A Python3 package for the network analysis of Hi-C data.

"""

__version__ = "alpha"


from ._cooler import HiconaCooler
from ._graph import HiconaGraph  # TODO: Make optional
from ._table import HiconaTable
from . import analysis
from . import preprocess
from ._core import subsample
