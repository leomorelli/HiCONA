"""Placeholder
"""
from cooler.util import open_hdf5
from pandas import DataFrame


class ChunkBordersIterator:
    """Iterator object of pixel chunk for a specified chromosome.

    Return tuples of two integers to use to slice the full pixel table and
    only retrieve a chunk of the desired size for the chromosome of interest.

    Parameters
    ----------
    chrom_id: str
        String identifier of the chromosome of interest
    chunk_size: int
        Number of pixels to span for each chunk
    ids_dict: dict
        Mapping of chromosome string identifiers into integer identifiers
        (self._chromids in a standard cooler object)
    h5_group: :py:class:`h5py.Group`
        Open handle to the root of a cooler file

    #TODO add examples
    """

    def __init__(self, store, root, extent, chunk_size):
        # Retrieve positions of boundary pixels
        with open_hdf5(store, mode="r") as h5_handle:
            h5_grp = h5_handle[root]
            min_off = h5_grp["indexes/bin1_offset"][extent[0]]
            max_off = h5_grp["indexes/bin1_offset"][extent[1]]

        # Define number of iterations
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
            self.curr_chunk += 1
            return self.borders[self.curr_chunk : self.curr_chunk + 2]
        raise StopIteration


class ChromTablesIterator:
    """Placeholder

    Placeholder
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
        if self.uri_index < self.max_uri:
            curr_uri = self.uri_list[self.uri_index]
            self.uri_index += 1

            with open_hdf5(self.store, mode="r") as h5_handle:
                main_grp = h5_handle[self.root + "/chrom_tables"]
                table_grp = main_grp[curr_uri]

                attr_dict = dict(table_grp.parent.attrs.items())
                attr_dict["chromosome"] = curr_uri.split("/")[-1]

                table = DataFrame({f: table_grp[f] for f in table_grp.keys()})

            return (table, attr_dict)

        raise StopIteration
