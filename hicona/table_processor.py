"""Placeholder"""

from collections.abc import Iterable
from statistics import median

from numpy import log2
import pandas as pd
from scipy import integrate

from .hicona_table import HiconaTable
from .utils.numeric import round_half_up
from .utils.hdf5ops import resize_tab, write_data


class _TableProcessor:
    """Placeholder"""

    def __init__(self, table, queries, norm_method):
        self._table = table
        self._queries = queries
        self._norm_method = norm_method

    def _filter_table(self, queries):
        """Remove pixels from the table not matching the filters."""

        final_size = 0
        store, pixels = self._table.get_pixel_uris()

        for chunk in table.get_chunks():
            for query in queries:
                chunk.query(query, inplace=True)

            write_data(store, pixels, chunk, final_size)
            final_size += len(chunk)

        resize_tab(store, pixels, final_size)

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
        # TODO: separate the queries
        self._filter_table()
        self._normalize()
        self._filter_table()
        self._sparsify()
