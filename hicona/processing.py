"""Placeholder"""

from collections.abc import Iterable
from statistics import median

from numpy import log2
import pandas as pd
from scipy import integrate

from .hicona_table import HiconaTable
from .utils.numeric import round_half_up
from .utils.hdf5ops import resize_tab, write_data


class _PixelsFilter:
    """Individual filter to apply to a pixel table."""

    def __init__(self, infos, value, query):
        self._infos = infos
        self._value = value
        self._query = query

    @property
    def infos(self):
        """Brief description of the filter."""
        return self._infos

    @property
    def value(self):
        """Value to set the filter to."""
        return self._value

    @value.setter
    def value(self, new):
        self._value = new

    @property
    def query(self):
        """Query string corresponding to the filter."""
        if self._value == None:
            return None
        elif isinstance(self._value, bool):
            return self._query if self._value else None
        else:
            return self._query.format(self._value)


class ProcessingFilters:
    """Object to select the filters to apply to a pixel table.

    Object meant to simplify the choice of filter to apply to a pixel table.
    Each attribute of the class is an individual filter; to see all available
    filters, and the current value they are set to, simply print the object.
    To see a filter description access its infos attribute.
    To change a filter value access its value attribute.

    Not meant for direct initialization.

    Parameters
    ----------
    filters: dict[str, dict]
        Dictionary of dictionaries representing the individual filters.
    """

    def __init__(self, filters: dict[str, dict]):
        for k, v in filters.items():
            setattr(self, k, _PixelsFilter(**v))

    def __str__(self):
        out = "Summary of the filters:"
        for k, v in self.__dict__.items():
            out += f"\n {k}: {v.value}"
        return out

    def reset_filters(self):
        """Set all filters value to None (e.i. skip that filter)."""

        for v in self.__dict__.values():
            v.value = None

    def get_queries(self) -> list[dict]:
        """Return the queries to use with ```pd.DataFrame.query```."""

        queries = []
        for k, v in self.__dict__.items():
            if v.query:
                query = {"name": k, "query": v.query, "value": v.value}
                queries.append(query)
        return queries


def get_processing_filters(method=None) -> ProcessingFilters:
    """Return an object to use to specify the filters to apply.

    Parameters
    ----------
    method: str, optional
        If provided, return the default filters for that method, else
        return an empty object to manually populate. Default is None.

    Returns
    -------
    An instance of the ProcessingFilters class.
    """

    # TODO: all the logic to fetch the right dictionary.
    
    filters_dict = pass
    filters_obj = ProcessingFilters(filters_dict)

    return filters_obj


class _TableProcessor:
    """Placeholder"""

    def __init__(self, table, pre_queries, norm_method, post_queries):
        self._table = table
        self._pre_queries = pre_queries
        self._norm_method = norm_method
        self._post_queries = post_queries

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
