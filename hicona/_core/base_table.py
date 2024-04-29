"""Module containing the Table class and related utilities."""

import json
from typing import Iterable

import cooler
import pandas as pd

from hicona._core import uris
from hicona._dtypes import PdChunks
from hicona._ops import hdf5
from hicona._core import genomic
from hicona.preprocess import Flow


FULL_TABLE = "full_table"


class TableIntervals:
    """Class to handle chunk fetching from pixel tables.

    Chunk fetching is handled by storing indexes for each fixed size chunk.
    When iterating through the table, each chunk is fetched and indexed.
    When the number of fetched rows covers the full chunk size (or if no more
    chunks are available), the chunk is returned and the process is repeated.

    The class also implements operations such as subsetting and boolean logic
    among indexes to facilitate the selection of complex intervals.

    Parameters
    ----------
    table : Table
        The table object to which the intervals belong.
    indexes : list[pd.Index], optional
        List of indexes for each chunk. If not provided, a full index is
        created for each chunk (all pixels are iterated). Default is None.
    interval_str : str, optional
        String representation of the intervals, that is a description of the
        operations used to generate the indexes. If not provided, it is set
        to `full_table`. Default is None.
    """

    def __init__(
        self,
        table: "Table",
        indexes: list[pd.Index] | None = None,
        interval_str: str | None = None,
    ):

        if bool(indexes) != bool(interval_str):
            raise ValueError(
                "Both indexes and interval_str must be provided or neither."
            )

        self._table = table
        self._indexes = indexes or self._initial_index()
        self._interval_str = interval_str or FULL_TABLE
        self._size: int = 0  # Random initialization value

        self.update_size()

    def __str__(self) -> str:
        return self._interval_str

    def __or__(self, other: "TableIntervals") -> "TableIntervals":

        iterator = zip(self._indexes, other.get_indexes())
        new_intervals = [i.join(j, how="outer") for i, j in iterator]
        new_repr = f"({self._interval_str} | {other._interval_str})"

        return TableIntervals(self._table, new_intervals, new_repr)

    def __and__(self, other: "TableIntervals") -> "TableIntervals":

        iterator = zip(self._indexes, other.get_indexes())
        new_intervals = [i.join(j, how="inner") for i, j in iterator]
        new_repr = f"({self._interval_str} & {other._interval_str})"

        return TableIntervals(self._table, new_intervals, new_repr)

    def _initial_index(self) -> list[pd.Index]:
        """Return an index of the complete table split into chunks."""

        table_size = hdf5.get_table_size(self._table.uris)
        chunk_size = self._table.chunk_size

        num_full_chunks, partial_chunk_size = divmod(table_size, chunk_size)
        full_chunk_ind = pd.Index(range(0, chunk_size), dtype="int32")
        part_chunk_ind = pd.Index(range(0, partial_chunk_size), dtype="int32")

        return [full_chunk_ind] * num_full_chunks + [part_chunk_ind]

    @property
    def size(self) -> int:
        """Return the size of the complete table or the subset."""
        return self._size

    def update_size(self) -> None:
        """Update the size of the complete table or the subset."""
        self._size = sum(len(c) for c in self._indexes)

    def subset(self, region: str, both: bool = True) -> "TableIntervals":
        """Subset a full genomic table to a region of interest."""

        # NOTE: currently not allowing the subset of subsets because it
        #       is not clear how to handle the interval_str in that case.
        #       Might be implemented in the future if needed.
        if self._interval_str != FULL_TABLE:
            raise ValueError("Cannot subset a subset. Use boolean operators instead.")

        new_repr = f"{region}({'+' if both else '-'})"

        gen_region = genomic.GenomicRegion(region)
        gen_region.snap_to_bin(self._table.bin_size)
        query_str = gen_region.to_query(both)

        pd_chunks = self._table.chunks(annotated=True)
        new_indexes = [c.query(query_str).index for c in pd_chunks]

        return TableIntervals(self._table, new_indexes, new_repr)

    def get_indexes(self) -> Iterable[pd.Index]:
        """Return the indexes of the table or the subset."""
        for index in self._indexes:
            yield index


class Table:
    """Base class for all table types in Hicona."""

    def __init__(
        self,
        uri_path: uris.Uris,
        intervals: TableIntervals | None = None,
        bin_size: int | None = None,
        chunk_size: int = 10_000_000,
    ):

        def reconstruct_flow(uri_path: uris.Uris) -> Flow:
            """Reconstruct the flow.Flow object from the store."""

            # TODO: check whether the table is valid and skip if not
            tab_attrs = hdf5.get_attrs(uri_path)
            flow_json = json.loads(tab_attrs["process_info"])
            return Flow.from_json(flow_json)

        def get_bin_size(uri_path: uris.Uris) -> int:
            """Return the bin size of the cooler."""

            parent_cool = cooler.Cooler(uri_path.cooler_uri())
            return parent_cool.binsize

        self._uris = uri_path
        self._flow = reconstruct_flow(uri_path)
        self._bin_size = bin_size or get_bin_size(uri_path)
        self._chunk_size = chunk_size
        self._intervals = intervals or TableIntervals(self)

    @property
    def bin_size(self) -> int:
        """Resolution of the original cooler (size of the bins in bp)."""
        return self._bin_size

    @property
    def chunk_size(self) -> int:
        """Size of the chunks to retrieve during iteration."""
        return self._chunk_size

    @property
    def flow(self) -> Flow:
        """flow.Flow object containing all processing information."""
        return self._flow

    @property
    def uris(self) -> uris.Uris:
        """uris.Uris object containing all table uris."""
        return self._uris

    @property
    def size(self) -> int:
        """Total number of pixels in the table."""
        return self._intervals.size

    def chunks(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
        query: str | None = None,
    ) -> PdChunks:
        """Returns an iterator of table chunks (as pandas DataFrames).

        Parameters
        ----------
        columns : Iterable[str] or None, optional
            If provided, only fetch the specified columns. Default is None.
        annotated: bool, optional
            Whether to annotate with the bin information. Default is False.
        query : str or None, optional
            If provided, only fetch the pixels that satisfy the query.
            Query must be a valid pandas query string. Default is None.

        Returns
        -------
        An iterator of table chunks (as pandas DataFrames).
        """

        def prepare_chunk(
            chunk: pd.DataFrame,
            bins=None,
            columns=None,
            query=None,
        ) -> pd.DataFrame:
            """Prepare the chunk for output with annotation of filtering."""

            if bins is not None:
                chunk = cooler.annotate(chunk, bins)
            if query is not None:
                chunk = chunk.query(query)
            if columns is not None:
                chunk = chunk[columns]

            return chunk

        # Fetch bins if needed for annotation
        cool = cooler.Cooler(self.uris.cooler_uri())
        bins = cool.bins()[:] if annotated else None

        # Growing list to concat to create the full chunk
        chunk_parts: list[pd.DataFrame] = []

        # Iterate all indexes
        indexes = self._intervals.get_indexes()
        for num, index in enumerate(indexes):

            # Skip chunk if no pixels from it need to be kept
            if len(index) == 0:
                continue

            # Fetch and index the chunk
            bounds = slice(num * self.chunk_size, (num + 1) * self.chunk_size)
            chunk = hdf5.fetch_chunk(self.uris, bounds)
            chunk = chunk.iloc[index]

            # Add kept pixels to the growing list
            chunk_parts.append(chunk)

            # Yield the chunk if it is complete
            # Save extra pixels for the next chunk
            new_chunks_size = sum(len(c) for c in chunk_parts)
            if new_chunks_size >= self.chunk_size:

                out_chunk = pd.concat(chunk_parts)

                chunk_parts = [out_chunk.iloc[self.chunk_size :]]
                out_chunk = out_chunk.iloc[: self.chunk_size]

                yield prepare_chunk(out_chunk, bins, columns, query)

        # Yield the remaining pixels as an incomplete chunk
        if len(chunk_parts) > 0:
            yield prepare_chunk(pd.concat(chunk_parts), bins, columns, query)

    def dataframe(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
        query: str | None = None,
    ) -> pd.DataFrame:
        """Return all table chunks in a single pandas DataFrame.

        Obtain the full table by concatenating all chunks into a single
        pandas DataFrame.

        Parameters
        ----------
        columns : Iterable[str], optional
            If provided, only fetch the specified columns. Default is None.
        annotated: bool, optional
            Whether to annotate with the bin information. Default is False.
        query : str, optional
            If provided, only fetch the pixels that satisfy the query.
            Query must be a valid pandas query string. Default is None.

        Returns
        -------
        A pandas DataFrame with all table pixels.

        Notes
        -----
        This operation can be rather expensive in terms of memory, especially
        for large or non subsetted tables.
        """

        iterator = self.chunks(columns, annotated, query)
        return pd.concat(iterator).reset_index(drop=True)

    def reset_index(self) -> None:
        """Regenerate the indexes (useful after table resizing)."""

        self._intervals = TableIntervals(self)
        # TODO: check because this might not work as expected
        # due to the way the table is resized in the hdf5 file
