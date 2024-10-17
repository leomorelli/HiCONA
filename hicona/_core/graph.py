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

from typing import TYPE_CHECKING, Generator

import numpy as np
import polars as pl
import graph_tool as gt  # type: ignore

from .._utils.chunked_ops import convert, to_iterable
from .._utils.dtype_conversion import NP_TO_GT, GT_TO_NP
from .table import BinTable, PixelTable


if TYPE_CHECKING:
    from .._utils.df_dtypes import PlChunks, DataFrame, DfStream, Bool
    from .cooler import HiconaCooler


__all__ = ["HiconaGraph"]


def _iter_item_chunks(
    iterable: Generator[np.ndarray],
    colnames: list[str],
    coltypes: list[str],
    chunk_size: int,
) -> Generator[pl.DataFrame]:

    buffer: list[np.ndarray | None] = [None] * chunk_size
    index: int = 0
    for item in iterable:
        buffer[index] = item
        index += 1

        if index == chunk_size:
            generator = zip(colnames, coltypes, np.array(buffer).T)
            yield pl.DataFrame({n: v.astype(t) for n, t, v in generator})

            buffer = [None] * chunk_size
            index = 0

    if index:
        buffer = [b for b in buffer if b is not None]
        generator = zip(colnames, coltypes, np.array(buffer).T)
        yield pl.DataFrame({n: v.astype(t) for n, t, v in generator})


def _iter_bin_chunks(graph: gt.Graph, chunk_size: int) -> "PlChunks":
    """Iterate over the bins of a graph in chunks."""

    # Get names and types of the vertex properties
    colnames: list[str] = list(graph.vp.keys())
    coltypes: list[str] = [GT_TO_NP[graph.vp[c].value_type()] for c in colnames]

    # The first column in the iterator is always the node_id
    colnames = ["node_id"] + colnames
    coltypes = ["int32"] + coltypes

    generator: Generator[np.ndarray] = graph.iter_vertices(vprops=graph.vp.values())
    return _iter_item_chunks(generator, colnames, coltypes, chunk_size)


def _iter_pix_chunks(graph: gt.Graph, chunk_size: int) -> "PlChunks":
    """Iterate over the rows of a graph in chunks."""

    # Get names and types of the edge properties
    colnames: list[str] = list(graph.ep.keys())
    coltypes: list[str] = [GT_TO_NP[graph.ep[c].value_type()] for c in colnames]

    # The first two columns in the iterator are always the node1_id and node2_id
    colnames = ["node1_id", "node2_id"] + colnames
    coltypes = ["int32", "int32"] + coltypes

    generator: Generator[np.ndarray] = graph.iter_edges(eprops=graph.ep.values())
    return _iter_item_chunks(generator, colnames, coltypes, chunk_size)


def _add_df_as_vp(graph: gt.Graph, df: pl.DataFrame):
    """Add vertex properties to a graph from a DataFrame."""
    for col in df.columns:
        np_col: np.ndarray = df[col].to_numpy()
        np_dtype: str = NP_TO_GT.get(np_col.dtype.name, "object")
        graph.vp[col] = graph.new_vertex_property(np_dtype, np_col)


def _add_df_as_ep(graph: gt.Graph, df: pl.DataFrame):
    """Add edge properties to a graph from a DataFrame."""
    for col in df.columns:
        np_col: np.ndarray = df[col].to_numpy()
        np_dtype: str = NP_TO_GT.get(np_col.dtype.name, "object")
        graph.ep[col] = graph.new_edge_property(np_dtype, np_col)


def _bin_to_node_id(data_df: pl.DataFrame, conv_df: pl.DataFrame) -> pl.DataFrame:
    """Change bin1_id and bin2_id to node1_id and node2_id."""

    conv_df = conv_df.select(["bin_id", "node_id"])  # Safety measure
    init_cols = [c for c in data_df.columns if c not in ["bin1_id", "bin2_id"]]

    return (
        data_df.join(conv_df, left_on="bin1_id", right_on="bin_id", how="left")
        .rename({"node_id": "node1_id"})
        .join(conv_df, left_on="bin2_id", right_on="bin_id", how="left")
        .rename({"node_id": "node2_id"})
        .select(["node1_id", "node2_id"] + init_cols)
    )


def _node_to_bin_id(data_df: pl.DataFrame, conv_df: pl.DataFrame) -> pl.DataFrame:
    """Change node1_id and node2_id to bin1_id and bin2_id."""

    conv_df = conv_df.select(["node_id", "bin_id"])  # Safety measure
    init_cols = [c for c in data_df.columns if c not in ["node1_id", "node2_id"]]

    return (
        data_df.join(conv_df, left_on="node1_id", right_on="node_id", how="left")
        .rename({"bin_id": "bin1_id"})
        .join(conv_df, left_on="node2_id", right_on="node_id", how="left")
        .rename({"bin_id": "bin2_id"})
        .select(["bin1_id", "bin2_id"] + init_cols)
    )


def _add_genomic_link(graph: gt.Graph, link_val: int):
    """Add missing genomic links to avoid isolated nodes."""
    # TODO: Probably need to find a better way to do this, seems slow

    num_edges: int = graph.num_edges()
    for i in range(graph.num_vertices() - 1):
        graph.edge(i, i + 1, add_missing=True)
    graph.ep.count.fa[num_edges:] = link_val

    genomic: list[bool] = [False] * num_edges + [True] * (graph.num_edges() - num_edges)
    graph.ep["genomic"] = graph.new_edge_property("bool", genomic)


class HiconaGraph:
    """Graph representation of a portion of 3D chromatin conformation data."""

    def __init__(
        self,
        *,
        bins: "DataFrame" | "DfStream",
        pixels: "DataFrame" | "DfStream",
        default_link: int = 1,
    ):

        bins, pixels = to_iterable(bins, pixels)
        bins = (
            pl.concat(convert(bins, "polars"))
            .with_row_index("node_id")
            .with_columns(pl.col("node_id").cast(pl.Int32))
        )

        graph = gt.Graph(bins.height, directed=False)

        # TODO: Maybe review this for the sake of memory efficiency
        # AFAIK, properties must be created in one go, no chunking allowed.
        # Theoretically, one could create an empty property and update it steb by step,
        # though I do not know whether doing that through ep.count.fa is intended.
        # Moreover, it would probably be slower using edge accessors given the loop.
        # For this reason all pixels properties are stored in a df and added at the end.
        eprops: list[pl.DataFrame] = []
        for chunk in convert(pixels, "polars"):
            eprops.append(chunk.select(pl.all().exclude("bin1_id", "bin2_id")))
            graph.add_edge_list(_bin_to_node_id(chunk, bins).to_numpy())

        # Add bin and pixel properties, as well as genomic links
        _add_df_as_vp(graph, bins.drop("node_id"))
        _add_df_as_ep(graph, pl.concat(eprops))
        _add_genomic_link(graph, default_link)

        self._graph: gt.Graph = graph

    @property
    def graph(self) -> gt.Graph:
        """graph_tool.Graph instance associated with the HiconaGraph."""
        return self._graph

    def get_bins(
        self,
        region: str | None = None,
        *,
        drop_node_id: Bool = True,
        bare: Bool = False,
        store_size: int = 10_000_000,
    ) -> "BinTable":
        """Return the bins as a dataframe."""

        node_chunks: PlChunks = _iter_bin_chunks(self._graph, store_size)

        if drop_node_id:
            node_chunks = (c.drop("node_id") for c in node_chunks)

        if bare:
            bare_cols: tuple[str, ...] = ("bin_id", "chrom", "start", "end")
            node_chunks = (c.select(bare_cols) for c in node_chunks)

        table = BinTable(node_chunks, store_size)
        return table if region is None else table.subset(region)

    def get_pixels(
        self,
        region: str | None = None,
        *,
        keep_genomic: str | None = None,
        store_size: int = 10_000_000,
    ) -> "PixelTable":
        """Return the pixels as a dataframe."""

        # TODO: Remove type ignore once the type checker is fixed
        id_table: pl.DataFrame = pl.concat(_iter_bin_chunks(self._graph, store_size))
        edge_chunks: PlChunks = _iter_pix_chunks(self._graph, store_size)
        edge_chunks = (_node_to_bin_id(c, id_table) for c in edge_chunks)  # type: ignore
        if not keep_genomic:
            edge_chunks = (
                c.filter(pl.col("genomic") == 0).drop("genomic") for c in edge_chunks
            )

        table = PixelTable(
            edge_chunks,
            bins=self.get_bins(store_size=store_size),
            store_size=store_size,
        )
        return table if region is None else table.subset(region)

    @classmethod
    def from_pixel_table(
        cls, table: "PixelTable", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a HiconaTable instance."""

        # TODO: Remove type ignore once the type checker is fixed
        pixels: "PlChunks" = table.get_chunks(region, dtype="polars")  # type: ignore
        bins: pl.DataFrame = table.bins.get_dataframe(region, dtype="polars")  # type: ignore

        return cls(bins=bins, pixels=pixels)

    @classmethod
    def from_cooler(
        cls, handle: "HiconaCooler", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a HiconaCooler instance."""

        pixels: "PixelTable" = handle.get_pixels(region)
        return cls.from_pixel_table(pixels, region=region)

    def compute_clustering(self):
        """Compute hierarchical clustering of the bins based on the pixels."""
