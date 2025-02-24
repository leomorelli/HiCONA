"""Generally useful operations for graph_tool Graph objects."""

import graph_tool.all as gt
import numpy as np
import polars as pl

from .df_dtypes import NpChunks, PlChunks
from .dtype_conversion import NP_TO_GT, GT_TO_NP


def _iter_item_chunks(
    iterable: NpChunks,
    colnames: list[str],
    coltypes: list[str],
    chunk_size: int,
) -> PlChunks:
    """Convert a generator of numpy arrays into a generator of polars DataFrames."""

    buffer: list[np.ndarray | None] = [None] * chunk_size
    index: int = 0
    for item in iterable:
        buffer[index] = item
        index += 1

        if index == chunk_size:
            generator = zip(colnames, coltypes, np.array(buffer).T)
            yield pl.DataFrame(
                {n: v.astype(t) for n, t, v in generator}, nan_to_null=True
            )

            buffer = [None] * chunk_size
            index = 0

    if index:
        buffer = [b for b in buffer if b is not None]
        generator = zip(colnames, coltypes, np.array(buffer).T)
        yield pl.DataFrame({n: v.astype(t) for n, t, v in generator}, nan_to_null=True)


def iter_bin_chunks(graph: gt.Graph, chunk_size: int) -> PlChunks:
    """Iterate over the bins of a graph in chunks."""

    # Get names and types of the vertex properties
    colnames: list[str] = list(graph.vp.keys())
    coltypes: list[str] = [GT_TO_NP[graph.vp[c].value_type()] for c in colnames]

    # The first column in the iterator is always the node_id
    colnames = ["node_id"] + colnames
    coltypes = ["int32"] + coltypes

    generator: NpChunks = graph.iter_vertices(vprops=graph.vp.values())
    return _iter_item_chunks(generator, colnames, coltypes, chunk_size)


def iter_pix_chunks(graph: gt.Graph, chunk_size: int) -> PlChunks:
    """Iterate over the rows of a graph in chunks."""

    # Get names and types of the edge properties
    colnames: list[str] = list(graph.ep.keys())
    coltypes: list[str] = [GT_TO_NP[graph.ep[c].value_type()] for c in colnames]

    # The first two columns in the iterator are always the node1_id and node2_id
    colnames = ["node1_id", "node2_id"] + colnames
    coltypes = ["int32", "int32"] + coltypes

    generator: NpChunks = graph.iter_edges(eprops=graph.ep.values())
    return _iter_item_chunks(generator, colnames, coltypes, chunk_size)


def add_df_as_vp(graph: gt.Graph, df: pl.DataFrame):
    """Add vertex properties to a graph from a DataFrame."""
    # NOTE: Assumes that the nodes are sorted, which should be the case
    # NOTE: Assumes that df is sorted the same way as the graph

    for col in df.columns:
        np_col: np.ndarray = df[col].to_numpy()
        np_dtype: str = NP_TO_GT.get(np_col.dtype.name, "object")
        graph.vp[col] = graph.new_vertex_property(np_dtype, np_col)


def add_df_as_ep(graph: gt.Graph, df: pl.DataFrame):
    """Add edge properties to a graph from a DataFrame."""
    # NOTE: Assumes that the edge are sorted, which should be the case at the beginning
    # NOTE: Assumes that df is sorted the same way as the graph
    # TODO: This does not seem too reliable, maybe check edge by edge even if slow

    for col in df.columns:
        np_col: np.ndarray = df[col].to_numpy()
        np_dtype: str = NP_TO_GT.get(np_col.dtype.name, "object")
        graph.ep[col] = graph.new_edge_property(np_dtype, np_col)
