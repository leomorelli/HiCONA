"""
Graph representation of a portion of 3D chromatin conformation data.

Class which utilizes the graph_tool.Graph class to represent a portion of a
Hi-C data-like file. The graph object has the bins as vertices and the pixels
as edges. All bin annotation columns are stored as vertex properties, as well
as the pixel count.

Rather than directly inheriting from graph_tool.Graph, the graph object is
stored as an attribute of the class. This avoids mixing the new methods and
attributes with the methods and attributes from graph_tool.Graph. 
This is relevant since most of the graph_tool.Graph methods will likely not be
needed by the user, therefore this way keep the namespace cleaner.

"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import graph_tool as gt  # type: ignore

from .._utils.chunked_ops import (
    rechunk,
    convert,
    to_iterable,
    format_stream,
    cast_dtypes,
)
from .table_ops import subset_bin_region, subset_pixel_region


if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        PlChunks,
        DataFrame,
        DfChunks,
        DfStream,
        PlStream,
        Bool,
        DfDtype,
    )
    from .table import HiconaTable
    from .cooler import HiconaCooler


__all__ = ["HiconaGraph"]


def _iter_row_chunks(graph: gt.Graph, chunk_size: int) -> "PlChunks":
    """Iterate over the rows of a graph in chunks."""

    buffer_size: tuple[int, int] = (chunk_size, (2 + len(list(graph.ep.keys()))))
    colnames: list[str] = ["node1_id", "node2_id"] + list(graph.ep.keys())
    buffer: np.ndarray = np.empty(buffer_size, dtype=int)

    index: int = 0
    for row in graph.iter_edges(eprops=graph.ep.values()):
        buffer[index, :] = row
        index += 1

        if index == chunk_size:
            yield pl.DataFrame(dict(zip(colnames, buffer.T)))
            index = 0

    if index:
        yield pl.DataFrame(dict(zip(colnames, buffer[:index].T))).drop_nulls()


class HiconaGraph:
    """Graph representation of a portion of 3D chromatin conformation data."""

    def __init__(
        self,
        *,
        bins: "DataFrame" | "DfStream",
        pixels: "DataFrame" | "DfStream",
        info: dict[str, Any],
        default_link: int = 1,
    ):

        bins, pixels = to_iterable(bins, pixels)
        bins = (
            pl.concat(convert(bins, "polars"))
            .with_row_index("node_id")
            .with_columns(pl.col("node_id").cast(pl.Int32))
        )

        # TODO: Maybe review this for the sake of memory efficiency
        # AFAIK, properties must be created in one go, no chunking allowed.
        # Theoretically, one could create an empty property and update it steb by step,
        # though I do not know whether doing that through ep.count.fa is intended.
        # Moreover, it would probably be slower using edge accessors given the loop.
        # For this reason all pixels properties are stored in a df and added at the end.
        eprops: list[pl.DataFrame] = []

        graph = gt.Graph(bins.height, directed=False)
        for chunk in convert(pixels, "polars"):

            eprops.append(chunk.select(pl.all().exclude("bin1_id", "bin2_id")))

            chunk = (
                chunk.join(bins, left_on="bin1_id", right_on="bin_id", how="left")
                .rename({"node_id": "node1_id"})
                .join(bins, left_on="bin2_id", right_on="bin_id", how="left")
                .rename({"node_id": "node2_id"})
                .select(["node1_id", "node2_id"])
            )

            graph.add_edge_list(chunk.to_numpy())

        eprops_df = pl.concat(eprops)
        for col in eprops_df.columns:
            graph.edge_properties[col] = graph.new_edge_property(
                "int",  # TODO: this is tmp, should be inferred
                eprops_df[col].to_numpy(),
            )

        # TODO: Probably need to find a better way to do this, seems slow
        # Add missing genomic links to avoid isolated nodes
        num_edges: int = graph.num_edges()

        for i in range(bins.height - 1):
            graph.edge(i, i + 1, add_missing=True)
        graph.ep.count.fa[num_edges:] = default_link  # Unsure about safety
        graph.edge_properties["genomic"] = graph.new_edge_property(
            "bool", [False] * num_edges + [True] * (graph.num_edges() - num_edges)
        )

        self._graph: gt.Graph = graph
        self._bins: pl.DataFrame = bins
        self._info: dict[str, Any] = info  # TODO: decide what goes in info

    @property
    def graph(self) -> gt.Graph:
        """graph_tool.Graph instance associated with the HiconaGraph."""
        return self._graph

    @property
    def info(self) -> dict[str, Any]:
        """Dictionary with additional information about the graph."""
        return self._info

    def get_bins(
        self,
        region: str | None = None,
        *,
        bare: Bool = False,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Return the bins as a dataframe."""

        chunks: "PlStream" = (self._bins,)

        if bare:
            bare_cols = ["node_id", "chrom", "start", "end"]
            chunks = (chunk.select(bare_cols) for chunk in chunks)

        if region:
            chunks = subset_bin_region(
                chunks,
                region=region,
                df_dtype="polars",
                as_chunks=True,
                chunk_size=chunk_size,
            )

        chunks = rechunk(chunks, chunk_size)
        chunks = cast_dtypes(chunks, {"chrom": str})
        return format_stream(chunks, df_dtype, as_chunks)

    def get_pixels(
        self,
        region: str | None = None,
        *,
        keep_genomic: bool = False,
        df_dtype: DfDtype = "polars",
        as_chunks: Bool = True,
        chunk_size: int = 10_000_000,
    ) -> "DataFrame" | "DfChunks":
        """Get the pixels table with all pixel properties."""

        def pix_chunks(graph: gt.Graph, bins: pl.DataFrame) -> "PlChunks":

            conv_table = bins.select(["node_id", "bin_id"])
            for pix_df in _iter_row_chunks(graph, 1_000_000):
                keep_cols: tuple[str, ...] = tuple(graph.ep.keys())

                if not keep_genomic:
                    pix_df = pix_df.filter(pl.col("genomic") == 0).drop("genomic")
                    keep_cols = tuple(col for col in keep_cols if col != "genomic")

                pix_df = (
                    pix_df.with_columns(
                        pl.col("node1_id").cast(pl.Int32),
                        pl.col("node2_id").cast(pl.Int32),  # TODO: rm?
                    )
                    .join(
                        conv_table, left_on="node1_id", right_on="node_id", how="left"
                    )
                    .drop("node1_id")
                    .rename({"bin_id": "bin1_id"})
                    .join(
                        conv_table, left_on="node2_id", right_on="node_id", how="left"
                    )
                    .drop("node2_id")
                    .rename({"bin_id": "bin2_id"})
                    .select(("bin1_id", "bin2_id") + keep_cols)
                )

                yield pix_df

        chunks: "DfChunks" = rechunk(pix_chunks(self._graph, self._bins), 10_000_000)

        if region:
            chunks = subset_pixel_region(
                chunks,
                region=region,
                ref_bins=self._bins,
                df_dtype="polars",
                as_chunks=True,
                chunk_size=chunk_size,
            )

        return format_stream(chunks, df_dtype, as_chunks)

    @classmethod
    def from_table(
        cls,
        table: "HiconaTable",
        *,
        region: str | None = None,
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a HiconaTable instance."""

        pixels: "PlChunks" = table.get_pixels(region, df_dtype="polars", as_chunks=True)
        bins: "PlChunks" = table.get_bins(region, df_dtype="polars", as_chunks=True)
        info: dict[str, Any] = table.info  # TODO: decide what goes in info

        return cls(bins=bins, pixels=pixels, info=info)

    @classmethod
    def from_cooler(
        cls,
        handle: "HiconaCooler",
        *,
        region: str | None = None,
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a HiconaCooler instance."""

        pixels: "PlChunks" = handle.get_pixels(
            region,
            df_dtype="polars",
            as_chunks=True,
        )
        bins: "PlChunks" = handle.get_bins(
            region,
            df_dtype="polars",
            as_chunks=True,
        )
        info: dict[str, Any] = {"region": region}  # TODO: decide what goes in info

        return cls(bins=bins, pixels=pixels, info=info)

    def plot_matrix(self):
        """Plot the genomic region data as a contact matrix."""

    def compute_clustering(self):
        """Compute hierarchical clustering of the bins based on the pixels."""
