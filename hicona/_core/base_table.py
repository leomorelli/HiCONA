"""Module containing the Table class and related utilities."""

import json
from pathlib import Path
from typing import Iterable

import cooler
import numpy as np
import polars as pl

from hicona._core import uris, annotation, genomic
from hicona._dtypes import DfChunks
from hicona._ops import hdf5
from hicona.preprocess import Flow


FULL_TABLE = "full_table"

__all__ = ["Table", "TableIndex"]


class TableIndex:
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
        index: pl.Series | None = None,
        interval: str | None = None,
    ):

        both_provided = index is not None and interval is not None
        none_provided = index is None and interval is None
        if not (both_provided or none_provided):
            raise ValueError("Both index and interval must be provided or neither.")

        if index is None:
            table_size = hdf5.get_table_size(table.uris)
            index = pl.Series("index", np.repeat(True, table_size))

        self._table: "Table" = table
        self._interval: str = interval or FULL_TABLE
        self._index: pl.Series = index
        self._size: int = int(self._index.sum())

    def __str__(self) -> str:
        return self._interval

    def __or__(self, other: "TableIndex") -> "TableIndex":
        index = self._index | other[:]
        interval = f"({self} | {other})"
        return TableIndex(self._table, index, interval)

    def __and__(self, other: "TableIndex") -> "TableIndex":
        index = self._index & other[:]
        interval = f"({self} & {other})"
        return TableIndex(self._table, index, interval)

    def __getitem__(self, key) -> pl.Series:
        return self._index[key]

    def __len__(self) -> int:
        return self._index.len()

    @property
    def interval(self) -> str:
        """Return the string representation of the intervals."""
        return self._interval

    def subset(self, region: str, both: bool = True) -> "TableIndex":
        """Subset a full genomic table to a region of interest."""

        # NOTE: currently not allowing the subset of subsets because it
        #       is not clear how to handle the interval_str in that case.
        #       Might be implemented in the future if needed.
        if self._interval != FULL_TABLE:
            raise ValueError("Cannot subset a subset. Use boolean operators instead.")

        interval = f"{region}({'+' if both else '-'})"

        gen_region = genomic.GenomicRegion(region)
        gen_region.snap_to_bin(self._table.bin_size)

        pd_chunks = self._table.chunks(annotated=True)
        exp = gen_region.to_query(both)
        new_indexes = [c.with_columns(exp.alias("index"))["index"] for c in pd_chunks]

        return TableIndex(self._table, pl.concat(new_indexes), interval)


class Table:
    """Base class for all table types in Hicona."""

    def __init__(
        self,
        uri_path: uris.Uris,
        intervals: TableIndex | None = None,
        bin_size: int | None = None,
        chunk_size: int = 10_000_000,
    ):

        def reconstruct_flow(uri_path: uris.Uris) -> Flow:
            """Reconstruct the flow.Flow object from the store."""

            tab_attrs = hdf5.get_attrs(uri_path)
            flow_name = Path(uri_path.hdf5_uris()[1]).name
            flow_json = json.loads(tab_attrs["flow"])
            return Flow.from_json(flow_name, flow_json)

        def get_bin_size(uri_path: uris.Uris) -> int:
            """Return the bin size of the cooler."""

            parent_cool = cooler.Cooler(uri_path.cooler_uri())
            return parent_cool.binsize

        if not uri_path.is_valid():
            raise ValueError(f"Table '{uri_path.hdf5_uris()[1]}' does not exist.")

        self._uris = uri_path
        self._flow = reconstruct_flow(uri_path)
        self._bin_size = bin_size or get_bin_size(uri_path)
        self._chunk_size = chunk_size
        self._index = intervals or TableIndex(self)

    def reset_index(self) -> None:
        """Reset the index of the table to the full table."""
        self._index = TableIndex(self)

    @property
    def bin_size(self) -> int:
        """Resolution of the original cooler (size of the bins in bp).

        Returns
        -------
        int
            Bin size in base pairs.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.bin_size
        10000
        """
        return self._bin_size

    @property
    def chunk_size(self) -> int:
        """Number of bins per chunk to retrieve during iteration.

        Returns
        -------
        int
            Number of bins per chunk.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.chunk_size
        10000000  # default value
        """
        return self._chunk_size

    @property
    def flow(self) -> Flow:
        """Flow object containing all preprocessing information.

        Returns
        -------
        Flow
            Workflow used during table creation.

        See Also
        --------
        hicona.preprocess.Flow : Used to define a preprocessing workflow.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.flow
        <hicona.preprocess._flow.Flow object at 0x73e704c9d090>
        """
        return self._flow

    @property
    def uris(self) -> uris.Uris:
        """Uris object containing all table uris.

        Returns
        -------
        Uris
            Object specifying the path to the table in the cooler file.

        See Also
        --------
        hicona._core.uris.Uris : Used to handle all uris in Hicona.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.uris
        <hicona._core.uris.Uris object at 0x73e704c9d090>
        """
        # TODO: fix docs after moving uris
        return self._uris

    @property
    def size(self) -> int:
        """Total number of pixels in the table.

        Returns
        -------
        int
            Number of pixels in the table.

        Examples
        --------
        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.size
        656880
        """
        return int(self._index[:].sum())

    def chunks(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
        query: pl.Expr | None = None,
    ) -> DfChunks:
        """Return an iterable of table chunks (as pandas.DataFrames).

        Read table chunks from memory only when needed and return them as
        ``pandas.DataFrame`` objects. Optionally, the bins can be annotated.

        Chunk size is determined by the ``chunk_size`` attribute of the table,
        unless a query is provided; in that case the chunk size is equal or
        smaller than the ``chunk_size`` attribute.

        .. note::
            This is the preferred way to access table data, since loading it
            fully into memory can be quite slow and memory intensive. If the
            full table is absolutely needed, use the ``dataframe`` method
            instead (which equates to calling this method and concatenating).

        Parameters
        ----------
        columns : iterable of str or None, optional
            If provided, only fetch the specified columns. Default is 'None'.
        annotated: bool, optional
            Whether to annotate with the bin information. Performed prior to
            column selection to allow for complex queries. Default is 'False'.
        query : str or None, optional
            If provided, only fetch the pixels that satisfy the query.
            Query must be a valid pandas query string, since internally it is
            passed to pandas.DataFrame.query. Default is 'None'.

        Returns
        -------
        Iterable of pandas.DataFrame
            Processed table chunks.

        See Also
        --------
        hicona.HiconaTable.dataframe : Used to fetch the full table as a DataFrame.
        pandas.DataFrame.query : Used to filter the table based on a query string.

        Examples
        --------
        Iterate over the first chunk of a table:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> for chunk in table.chunks():
        ...     print(chunk.head())
        ...     break
        # TODO: remake this example

        Only fetch the count values for the first chunk of the table:

        >>> for chunk in table.chunks(columns=["count"]):
        ...     print(chunk.head())
        ...     break
              count
        0        1
        1        1
        2        1
        3        1
        4        1

        Iterate over the first chunk of a table with bin annotations:

        >>> for chunk in table.chunks(annotated=True):
        ...     print(chunk.head())
        ...     break
              chrom1  start1    end1 HMM_annot1  ...  bin1_id bin2_id  count  norm
        0       chr1   10000   20000         Tx  ...        1    2720      1   1.0
        1       chr1   10000   20000         Tx  ...        1    5476      1   1.0
        2       chr1   10000   20000         Tx  ...        1    5802      1   1.0
        3       chr1   10000   20000         Tx  ...        1   18419      1   1.0
        4       chr1   30000   40000       Void  ...        3     890      1   1.0

        Iterate over the first chunk of a table with a query:

        >>> for chunk in table.chunks(query="score < 0.1"):
        ...     print(chunk.head())
        ...     break
        # TODO: remake this example

        """

        def prepare_chunk(
            chunk: pl.DataFrame,
            columns: Iterable[str] | None,
            bins: pl.DataFrame | None,
            query: pl.Expr | None,
        ) -> pl.DataFrame:
            """Prepare the chunk for output with annotation of filtering."""

            # TODO: Add back support for pandas query strings
            if bins is not None:
                chunk = annotation.annotate(chunk, bins)
            if query is not None:
                chunk = chunk.filter(query)
            if columns is not None:
                chunk = chunk.select(columns)

            return chunk

        # Fetch bins if needed for annotation
        cool = cooler.Cooler(self.uris.cooler_uri())
        bins = pl.DataFrame(cool.bins()[:]) if annotated else None
        prep_kwargs = {"columns": columns, "bins": bins, "query": query}

        buffer: pl.DataFrame = pl.DataFrame()
        index_pos: int = 0

        while index_pos < len(self._index):

            # Add next chunk to the buffer
            bounds = slice(index_pos, index_pos + self._chunk_size)
            index_chunk = self._index[bounds]
            pixel_chunk = hdf5.fetch_chunk(self._uris, bounds)

            buffer = pl.concat([buffer, pixel_chunk.filter(index_chunk)])
            index_pos += self._chunk_size

            if buffer.height >= self._chunk_size:
                yield prepare_chunk(buffer.slice(0, self._chunk_size), **prep_kwargs)
                buffer = buffer.slice(self._chunk_size)

        if buffer.height > 0:
            yield prepare_chunk(buffer, **prep_kwargs)

    def dataframe(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
        query: pl.Expr | None = None,
    ) -> pl.DataFrame:
        """Return all table chunks in a single pandas DataFrame.

        Obtain the full table by concatenating all chunks into a single
        ``pandas.DataFrame`` object. The table can be optionally annotated and
        filtered.

        .. warning::
            This operation can be rather expensive in terms of memory,
            especially for large or non subsetted tables. If possible, use
            the ``chunks`` method to iterate over the table in chunks instead.

        Parameters
        ----------
        columns : iterable of str, optional
            If provided, only fetch the specified columns. Default is 'None'.
        annotated: bool, optional
            Whether to annotate with the bin information. Performed prior to
            column selection to allow for complex queries. Default is 'False'.
        query : str, optional
            If provided, only fetch the pixels that satisfy the query.
            Query must be a valid pandas query string, since internally it is
            passed to pandas.DataFrame.query. Default is 'None'.

        Returns
        -------
        A pandas.DataFrame with all table pixels.

        See Also
        --------
        hicona.HiconaTable.chunks : Used to fetch the table in chunks.
        pandas.DataFrame.query : Used to filter the table based on a query string.

        Examples
        --------
        Load a small table in memory at once:

        >>> handle = HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> table.dataframe()  # Usually tables are much larger than this
        # TODO: remake this example

        Only fetch the bin ids in the table:

        >>> table.dataframe(columns=["bin1_id", "bin2_id"])
                bin1_id  bin2_id
        0             1     2720
        1             1     5476
        2             1     5802
        3             1    18419
        4             3      890
        ...         ...      ...
        449637      520     4700
        449638      520     4701
        449639      520     4702
        449640      520     4704
        449641      520     4706
        <BLANKLINE>
        [449642 rows x 2 columns]

        Fetch the table with bin annotations:

        >>> table.dataframe(annotated=True)
               chrom1   start1     end1 HMM_annot1  ...  bin1_id bin2_id  count  norm
        0        chr1    10000    20000         Tx  ...        1    2720      1   1.0
        1        chr1    10000    20000         Tx  ...        1    5476      1   1.0
        2        chr1    10000    20000         Tx  ...        1    5802      1   1.0
        3        chr1    10000    20000         Tx  ...        1   18419      1   1.0
        4        chr1    30000    40000       Void  ...        3     890      1   1.0
        ...       ...      ...      ...        ...  ...      ...     ...    ...   ...
        449637   chr1  5200000  5210000        Het  ...      520    4700      1   1.0
        449638   chr1  5200000  5210000        Het  ...      520    4701      1   1.0
        449639   chr1  5200000  5210000        Het  ...      520    4702      1   1.0
        449640   chr1  5200000  5210000        Het  ...      520    4704      3   2.0
        449641   chr1  5200000  5210000        Het  ...      520    4706      1   1.0
        <BLANKLINE>
        [449642 rows x 16 columns]

        Only fetch the rows where the ``score`` is smaller than 0.1:

        >>> table.dataframe(query="score < 0.1")
        # TODO: remake this example
        """

        iterator = self.chunks(columns, annotated, query)
        return pl.concat(iterator)
