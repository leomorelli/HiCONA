"""Module containing the Table class and related utilities."""

import json
from typing import Iterable

import cooler
import pandas as pd

from hicona._core import uris
from hicona._dtypes import PdChunks, OptionalAxes
from hicona._ops import hdf5
from hicona._core import genomic
from hicona.preprocess import Flow
from hicona.analysis import _plotting as plotting  # TODO: Make better


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

            tab_attrs = hdf5.get_attrs(uri_path)
            flow_json = json.loads(tab_attrs["flow"])
            return Flow.from_dict(*flow_json.popitem())

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
        self._intervals = intervals or TableIntervals(self)

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
        return self._intervals.size

    def chunks(
        self,
        columns: Iterable[str] | None = None,
        annotated: bool = False,
        query: str | None = None,
    ) -> PdChunks:
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
           alpha_max  alpha_min  bin1_id  bin2_id  count  norm
        0     0.4219     0.4037        1     2720      1   1.0
        1     0.4219     0.3737        1     5476      1   1.0
        2     0.4219     0.3814        1     5802      1   1.0
        3     0.4219     0.3816        1    18419      1   1.0
        4     0.4219     0.4155        3      890      1   1.0

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

        >>> for chunk in table.chunks(query="alpha_min < 0.1"):
        ...     print(chunk.head())
        ...     break
              alpha_max  alpha_min  bin1_id  bin2_id  count      norm
        0        0.0839     0.0787       94      130     35  2.772590
        1        0.0900     0.0895       94      131     26  2.632268
        2        0.1039     0.0947       95      126     33  2.514573
        3        0.0776     0.0739       98      123     48  2.807355
        4        0.1031     0.0963       98      126     38  2.523562

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
                alpha_max  alpha_min  bin1_id  bin2_id  count  norm
        0          0.4219     0.4037        1     2720      1   1.0
        1          0.4219     0.3737        1     5476      1   1.0
        2          0.4219     0.3814        1     5802      1   1.0
        3          0.4219     0.3816        1    18419      1   1.0
        4          0.4219     0.4155        3      890      1   1.0
        ...           ...        ...      ...      ...    ...   ...
        449637     0.4574     0.4220      520     4700      1   1.0
        449638     0.4574     0.3963      520     4701      1   1.0
        449639     0.4574     0.3909      520     4702      1   1.0
        449640     0.2091     0.1761      520     4704      3   2.0
        449641     0.4574     0.4107      520     4706      1   1.0
        <BLANKLINE>
        [449642 rows x 6 columns]

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

        Only fetch the rows where the ``alpha_min`` is smaller than 0.1:

        >>> table.dataframe(query="alpha_min < 0.1")
              alpha_max  alpha_min  bin1_id  bin2_id  count      norm
        0        0.0839     0.0787       94      130     35  2.772590
        1        0.0900     0.0895       94      131     26  2.632268
        2        0.1039     0.0947       95      126     33  2.514573
        3        0.0776     0.0739       98      123     48  2.807355
        4        0.1031     0.0963       98      126     38  2.523562
        ...         ...        ...      ...      ...    ...       ...
        1173     0.1313     0.0734      517     1311      8  3.169925
        1174     0.1668     0.0990      517     1339      6  2.807355
        1175     0.1158     0.0844      517     1359      7  3.000000
        1176     0.0990     0.0969      517     1427      6  2.807355
        1177     0.1937     0.0747      519     1323      7  3.000000
        <BLANKLINE>
        [1178 rows x 6 columns]
        """

        iterator = self.chunks(columns, annotated, query)
        return pd.concat(iterator).reset_index(drop=True)

    def reset_index(self) -> None:
        """Refresh indexes to pixel position in the full table.

        Update the indexes to the pixels to consider from the full table in
        memory. This is used to refresh the object when the table changes on
        disk. In general, should not be needed by the user.
        """

        self._intervals = TableIntervals(self)
        # TODO: check because this might not work as expected
        # due to the way the table is resized in the hdf5 file

        # TODO: Maybe recreate the table from scratch and remove this method
