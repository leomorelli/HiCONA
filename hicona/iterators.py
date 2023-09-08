"""Placeholder
Placeholder
"""

from cooler.util import open_hdf5
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

    def __init__(self, store, root, uris):
        self.store = store
        self.root = root
        self.uri_list = uris

        self.uri_index = 0
        self.max_uri = len(uris)

    def __len__(self):
        return self.max_uri

    def __iter__(self):
        return self

    def __next__(self):
        if self.uri_index >= self.max_uri:
            raise StopIteration

        curr_uri = self.uri_list[self.uri_index]
        self.uri_index += 1

        with open_hdf5(self.store, mode="r") as h5_handle:
            main_grp = h5_handle[self.root + "/chrom_tables"]
            table_grp = main_grp[curr_uri]

            attr_dict = dict(table_grp.parent.attrs.items())
            attr_dict["chromosome"] = curr_uri.split("/")[-1]
            table = DataFrame({f: table_grp[f] for f in table_grp.keys()})
            table = ChromTable(table, attr_dict)

        return table
