"""Placeholder for the parquet tables module."""

import tempfile
import pathlib
import json
from collections.abc import Iterable
import math
import time

import polars as pl
import pyarrow.parquet as pq
import cooler


from hicona._ops import genomic, hdf5, uris, logging, chunked
from hicona._dtypes import Operation, DfChunks
import hicona.preprocess as prep

LOGGER = logging.get_console_logger("hicona_table")


class TmpStorage:
    """Class to handle storage in temporary directories."""

    def __init__(self, prefix: str = ""):
        self._tmp_store = tempfile.TemporaryDirectory(prefix="hicona_" + prefix)

    def __del__(self):
        if hasattr(self, "_tmp_store"):
            self._tmp_store.cleanup()

    @property
    def tmp_store(self) -> pathlib.Path:
        """Return the temporary folder in which the table is stored."""
        return pathlib.Path(self._tmp_store.name)


class Table(TmpStorage):
    """Abstract base class for any table with a temporary parquet storage."""

    def __init__(self, path: uris.Uris, prefix: str = ""):
        super().__init__(prefix)
        self._path = path

    @property
    def path(self) -> uris.Uris:
        """Return the URIs object associated with the table."""
        return self._path

    def dataframe(self) -> pl.LazyFrame:
        """Return the table as a dataframe."""
        return pl.scan_parquet(self.tmp_store)

    def chunks(self) -> "DfChunks":
        """Return the table as a generator of chunks."""

        # TODO: Make chunk size consistent
        # Faster to iterate empty ones rather than rechunking currently
        # TODO: There probably is a better way using arrow

        parquet = pq.ParquetDataset(self.tmp_store)
        for batch in parquet.fragments:
            batch_df = pl.from_arrow(batch.to_table())
            assert isinstance(batch_df, pl.DataFrame)

            if batch_df.height > 0:
                yield batch_df

    def _save_chunks(self, chunks: Iterable[pl.DataFrame]):
        """Save the chunks from a h5df file to the parquet store."""
        for i, chunk in enumerate(chunks):
            chunk.write_parquet(self.tmp_store / f"chunk_{str(i).zfill(4)}.parquet")


class PixelTable(Table):
    """Table of pixel data."""

    def __init__(
        self,
        path: uris.Uris,
        bins: "BinTable",
        chunk_size: int = 10_000_000,
    ):
        super().__init__(path=path, prefix="pixel_table_")
        self._bins = bins
        self._flow: prep.Flow = prep.Flow()
        self._chunk_size = chunk_size

    @classmethod
    def from_cooler(cls, cooler_path: uris.Uris, chunk_size: int) -> "PixelTable":
        """Create a new table from a cooler file."""

        bins = BinTable.from_cooler(cooler_path)

        instance = cls(cooler_path, bins, chunk_size)
        LOGGER.info("Copying chunks from %s", cooler_path.cooler_uri())
        instance._save_chunks(hdf5.iter_chunks(cooler_path, instance._chunk_size))

        flow_json: str = hdf5.get_attrs(cooler_path).get("flow", "{}")
        instance._flow = prep.Flow.from_json(json.loads(flow_json), cooler_path.path)
        return instance

    @classmethod
    def from_operation(cls, table: "PixelTable", operation: Operation) -> "PixelTable":
        """Create a new table by applying an operation to an existing table."""

        instance = PixelTable(table.path, table.bins)
        LOGGER.info("Applying operation %s", operation.name)

        start = time.time()
        for i, batch in enumerate(operation.run(table)):
            assert isinstance(batch, pl.DataFrame)
            batch.write_parquet(instance.tmp_store / f"chunk_{str(i).zfill(6)}.parquet")
        LOGGER.info("Operation took %.4f seconds", time.time() - start)

        instance._flow = table.flow.ops_add(operation)
        return instance

    @property
    def chunk_size(self) -> int:
        """Return the chunk size of the table."""
        return self._chunk_size

    @property
    def bins(self) -> "BinTable":
        """Return the bin table associated with this pixel table."""
        return self._bins

    @property
    def chroms(self) -> "ChromTable":
        """Return the chrom table associated with this pixel table."""
        return self.bins.chroms

    @property
    def bin_size(self) -> int:
        """Return the bin size of the table."""
        return self.bins.bin_size

    @property
    def flow(self) -> prep.Flow:
        """Return the flow object associated with the table.

        The flow object is created by running eval on the operations in the
        history, therefore the functions must be available in the global scope.
        """
        return self._flow

    def apply(self, ops: Operation | prep.Flow) -> "PixelTable":
        """Return a new table with the given operations applied."""

        if not isinstance(ops, prep.Flow):
            ops = prep.Flow([ops])

        new_table = self
        for op in ops.operations:
            new_table = PixelTable.from_operation(new_table, op)

        return new_table

    def get(self, operation: Operation) -> DfChunks:
        """Return a dataframe with the operation applied."""
        return operation.run(self)

    def get_size(self):
        """Return the number of pixels in the table."""
        # TODO: Probably a better way to do this
        return sum(chunk.shape[0] for chunk in self.chunks())

    def get_bins_with_stats(self, weight_col: str = "count") -> pl.DataFrame:
        """Return the bins with the operation applied."""

        node_stats = chunked.get_node_stats(self.chunks(), weight_col).sort("bin_id")
        bins_table = (
            self.bins.dataframe()
            .with_row_index("bin_id")
            .with_columns(pl.col("bin_id").cast(pl.Int64))
            .collect()
        )
        return (
            bins_table.join(node_stats, on="bin_id", how="left")
            .drop_nulls()
            .with_columns(pl.col("chrom").cast(pl.Utf8))
            .sort(["chrom", "start", "end"])
            .with_columns(pl.col("chrom").cast(pl.Categorical))
        )

    def bins_with_annot(self, col: str) -> "DfChunks":
        """Get a dataframe with the columns `bin_id` and `chrom` for join purposes."""

        chrom_ann = (
            self.bins.dataframe()
            .select(col)
            .with_row_index("bin_id")
            .with_columns(pl.col("bin_id").cast(pl.Int64))
            .collect()
        )

        # TODO: CategoricalRemappingWarning is raised in this area, fix
        for chunk in self.chunks():
            yield (
                chunk.join(chrom_ann, left_on="bin1_id", right_on="bin_id", how="left")
                .join(chrom_ann, left_on="bin2_id", right_on="bin_id", how="left")
                .rename({col: f"{col}1", f"{col}_right": f"{col}2"})
            )

    def table_with_annot(self, col: str) -> pl.LazyFrame:
        """Return a table with the annotation column."""

        chrom_ann = (
            self.bins.dataframe()
            .select(col)
            .with_row_index("bin_id")
            .with_columns(pl.col("bin_id").cast(pl.Int64))
            .collect()
            .lazy()
        )

        return (
            pl.scan_parquet(self.tmp_store)
            .join(chrom_ann, left_on="bin1_id", right_on="bin_id", how="left")
            .join(chrom_ann, left_on="bin2_id", right_on="bin_id", how="left")
            .rename({col: f"{col}1", f"{col}_right": f"{col}2"})
        )


class BinTable(Table):
    """Table of bin data."""

    def __init__(self, path: uris.Uris, chroms: "ChromTable"):
        super().__init__(path=path, prefix="bin_table_")
        self._chroms = chroms

        handle = cooler.Cooler(path.cooler_uri())
        self._bin_size: int = int(handle.info["bin-size"])

    @classmethod
    def from_cooler(cls, cooler_path: uris.Uris) -> "BinTable":
        """Create a new table from a cooler file."""

        chroms = ChromTable.from_cooler(cooler_path)
        path = cooler_path.add_path("bins")

        cooler_handle = cooler.Cooler(cooler_path.cooler_uri())
        chunks = (pl.DataFrame(cooler_handle.bins()[:]),)

        instance = cls(path, chroms)
        instance._save_chunks(chunks)

        return instance

    @property
    def bin_size(self) -> int:
        """Return the bin size of the table."""
        return self._bin_size

    @property
    def chroms(self) -> "ChromTable":
        """Return the chrom table associated with this bin table."""
        return self._chroms

    def get_region_bounds(self, genomic_region: str) -> tuple[int, int]:
        """Return the bin bounds of a given genomic region."""

        region = genomic.GenomicRegion(genomic_region)

        # base_offset is the total number of bins in the previous chromosomes
        # upper_bound is the total number of bins in the current chromosome
        chrom_offsets: pl.DataFrame = (
            self._chroms.pixel_counts(self.bin_size)
            .with_columns(
                pl.col("num_bins").shift(1).fill_null(0).cum_sum().alias("offset")
            )
            .filter(pl.col("name") == region.chrom)
            .collect()
        )

        base_offset: int = chrom_offsets.get_column("offset")[0]
        upper_bound: int = chrom_offsets.get_column("num_bins")[0]

        offset = region.end / self.bin_size if region.end else upper_bound

        lower = base_offset + (region.start or 0) // self.bin_size
        upper = base_offset + math.ceil(offset)

        return lower, upper


class ChromTable(Table):
    """Table of chromosome data."""

    def __init__(self, path: uris.Uris):
        super().__init__(path=path, prefix="chrom_table_")

    @classmethod
    def from_cooler(cls, cooler_path: uris.Uris) -> "ChromTable":
        """Create a new table from a cooler file."""

        path = cooler_path.add_path("chroms")

        cooler_handle = cooler.Cooler(cooler_path.cooler_uri())
        chunks = (pl.DataFrame(cooler_handle.chroms()[:]),)

        instance = cls(path)
        instance._save_chunks(chunks)

        return instance

    def pixel_counts(self, bin_size: int) -> pl.LazyFrame:
        """Frame with the number of pixels per chromosome at the given bin size."""
        return self.dataframe().with_columns(
            (pl.col("length") / bin_size).ceil().cast(pl.Int64).alias("num_bins")
        )
