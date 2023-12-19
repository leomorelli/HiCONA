"""Placeholder
Placeholder
"""

import h5py
from pandas import DataFrame

from .chrom_table import ChromTable


class ChromTablesIterator:
    """Iterator object of chromosome-level tables and respective information.

    For each table group specified in a list of URI strings, return a
    :py:class:`ChromTable` object whose data attribute corresponds to all
    tables in the group, while the preprocessing_params contains all the
    parameters used for processing plus the chromosome id.

    Parameters
    ----------
    store : str
        Path to the cool/mcool file.
    root : str
        URI string to resolution of interest.
    uris : list
        List of URI strings to the table groups of interest.
    """

    def __init__(self, store, uris):
        self._store = store
        self._uri_list = uris

        self._uri_index = 0
        self._max_uri = len(uris)

    def __len__(self):
        return self._max_uri

    def __iter__(self):
        return self

    def __next__(self):
        if self._uri_index >= self._max_uri:
            raise StopIteration

        curr_uri = self._uri_list[self._uri_index]
        self._uri_index += 1

        with h5py.File(self._store, mode="r") as h5_handle:
            table_grp = h5_handle[curr_uri]

        return ChromTable(self._store, curr_uri)
