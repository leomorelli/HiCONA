"""Placeholder"""

from collections.abc import Iterable
import importlib.resources as imp_res
from statistics import median

from numpy import log2
import pandas as pd
from scipy import integrate

from . import table_filter_funs as ffuns
from .hicona_table import HiconaTable
from .utils.numeric import round_half_up
from .utils.hdf5_ops import resize_table, write_chunk


__all__ = ["get_processing_filters"]


# TODO: Move into a module?
def apply_filter(table, filter_fun, **kwargs):
    """Apply a filtering function to a table."""

    filt_table_size = 0
    for chunk in filter_fun(table, **kwargs):
        write_chunk(table.store, table.pixels_uri, chunk, filt_table_size)
        filt_table_size += len(chunk)
    resize_table(table.store, table.pixels_uri, filt_table_size)


class _FiltersManager:
    """Pixels table filters scheduler.

    Stores a list of filters to apply to a pixel table.
    Filters are applied in the order they are provided.
    A table filtering function is a function which takes as input a table
    object and some other keywords, then yields filtered chunks of the table.
    """

    def __init__(self):
        self._filters = []
        self._default = [f for f in dir(ffuns) if callable(getattr(ffuns, f))]

    @property
    def filters(self):
        """Currently scheduled filters with respective parameters."""
        return self._filters

    def available_filters(self):
        """Summary of the filter functions implemented by HiCONA."""

        sep_line = "-" * 79 + "\n"
        out = sep_line + "Available Filters\n" + sep_line
        for func in self._default:
            out += f"{func}:\n{getattr(ffuns, func).__doc__}\n"
        out += sep_line
        print(out.strip())

    def add_filter(self, filter_fun, fun_kwargs):
        """Add a filter to the workflow."""

        if isinstance(filter_fun, str):
            try:
                filter_fun = getattr(ffun, filter_fun)
            except AttributeError:
                raise ValueError(f"{filter_fun} is not a filter function.")

        self._filters.append([filter_fun, fun_kwargs])

    def remove_filter(self, filter_fun):
        """Remove all scheduled instances of the provided filter."""

        if callable(filter_fun):
            filter_fun = filter_fun.__name__

        kept = [f for f in self._filters if f[0].__name__ != filter_fun]
        self._filters = kept

    def reset_filters(self):
        """Reset all scheduled filters."""

        self._filters = []


class _TableProcessor:
    """Placeholder"""

    def __init__(self, table, pre_filters, norm_method, post_filters):
        self._table = table
        self._pre_filters = pre_filters
        self._norm_method = norm_method
        self._post_filters = post_filters

    def _filter_table(self, pre_norm=True):
        """Apply filters to a table."""

        filters = self._pre_filters if pre_norm else self._post_filters
        for filter_fun, fun_kwargs in filters:
            apply_filter(self._table, filter_fun, fun_kwargs)

    def _normalize(self):
        """Placeholder"""

        # Select appropriate normalization function
        norm_fun = None
        match self._norm_method:
            case "hicona":
                norm_fun = hicona_norm
            case "ice":
                norm_fun = ice_norm

        # Do the actual normalization

    def _sparsify(self):
        pass

    def start(self):
        """Begin actual table processing."""

        self._filter_table(pre_norm=True)
        self._normalize()
        self._filter_table(pre_norm=False)
        self._sparsify()
