"""Placeholder
Placeholder
"""

from cooler.util import open_hdf5
from pandas import DataFrame


# TODO: Use Abstract base class?
class ChunkBordersIterator:
    """Iterator object of pixel chunk borders of consistent size.

    Return tuples of two integers to use to slice a pixel table
    and retrieve a chunk of the desired size (or as big as possible).

    Parameters
    ----------
    bounds : tuple
        Upper and lower indexes bounds for table selection
    chunk_size : int
        Number of pixels to span for each chunk
    """

    def __init__(self, bounds, chunk_size):
        # Define number of iterations
        min_off, max_off = bounds
        self.curr_chunk = 0
        self.num_chunks = -(-(max_off - min_off) // chunk_size)

        # Define list of break-points
        self.borders = [min_off + k * chunk_size for k in range(self.num_chunks)]
        if self.borders[-1] < max_off:
            self.borders.append(max_off)

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_chunk < self.num_chunks:
            chunk = self.borders[self.curr_chunk : self.curr_chunk + 2]
            self.curr_chunk += 1
            return chunk
        raise StopIteration


class ChromTablesIterator:
    """Iterator object of chromosome-level tables and respective information.

    For each table group specified in a list of URI strings, return a tuple
    in the form (chromosome table, information dictionary), where the
    chromosome table is a :py:class:`DataFrame` obtained using all tables in
    the group as columns, while the information dictionary contains all the
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

        return (table, attr_dict)
