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

from functools import partial
from typing import Any, cast, Generator, Literal, overload, TYPE_CHECKING

import numpy as np
import polars as pl
import graph_tool.all as gt  # type: ignore

from .._utils.chunked_ops import convert, to_iterable
from .._utils.dtype_conversion import NP_TO_GT, GT_TO_NP


if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfStream,
        PdChunks,
        PlChunks,
        DfDtype,
    )
    from .cooler import HiconaCooler
    from .table import PixelTable


__all__ = ["HiconaGraph"]


#########################################################################################
#################### UTILITY FUNCTIONS FOR GENERAL GRAPH MANIPULATION ###################
#########################################################################################


def _iter_item_chunks(
    iterable: Generator[np.ndarray],
    colnames: list[str],
    coltypes: list[str],
    chunk_size: int,
) -> Generator[pl.DataFrame]:
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
    # NOTE: Assumes that the nodes are sorted, which should be the case
    # NOTE: Assumes that df is sorted the same way as the graph

    for col in df.columns:
        np_col: np.ndarray = df[col].to_numpy()
        np_dtype: str = NP_TO_GT.get(np_col.dtype.name, "object")
        graph.vp[col] = graph.new_vertex_property(np_dtype, np_col)


def _add_df_as_ep(graph: gt.Graph, df: pl.DataFrame):
    """Add edge properties to a graph from a DataFrame."""
    # NOTE: Assumes that the edge are sorted, which should be the case at the beginning
    # NOTE: Assumes that df is sorted the same way as the graph
    # TODO: This does not seem too reliable, maybe check edge by edge even if slow

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
    # NOTE: Assumes that the nodes are sorted, which should be the case

    graph.ep["genomic"] = graph.new_edge_property("bool", False)

    for i in range(graph.num_vertices() - 1):
        edge = graph.edge(i, i + 1)

        # Skip if the node already exists
        if edge:
            continue

        # Skip if the bins belong to different chroms
        if graph.vp["chrom"][i] != graph.vp["chrom"][i + 1]:
            continue

        edge = graph.add_edge(i, i + 1)
        graph.ep.count[edge] = link_val
        graph.ep.genomic[edge] = True


#########################################################################################
############################ UTILITY FUNCTIONS FOR CLUSTERING ###########################
#########################################################################################


def _get_callback_func() -> tuple[partial, list[gt.Graph], list]:
    """Create partial function to use as callback in hierarchical_clustering."""

    def callback_func(
        state: gt.MixedMeasuredBlockState,
        graph: list[gt.Graph],
        parts: list[gt.PropertyArray],
    ):
        new_graph = state.collect_marginal(graph[0] if len(graph) > 0 else None)
        try:
            graph[0] = new_graph
        except IndexError:
            graph.append(new_graph)

        bstate = cast(gt.NestedBlockState, state.get_block_state())
        parts.append(bstate.levels[0].b.a.copy())

        # g = state.get_graph()
        # MAX_VAL = 55
        # max_num_edges = g.num_vertices() * (g.num_vertices() - 1) // 2
        # M = g.num_edges() * MAX_VAL  # + (max_num_edges - g.num_edges())
        # T = g.ep.count.fa.sum()
        # TODO: add check for no diagonal

    new_graph: list[gt.Graph] = []  # In list since the argument must be mutable
    partitions: list[gt.PropertyArray] = []
    callback: partial = partial(
        callback_func,
        graph=new_graph,
        parts=partitions,
    )

    return callback, new_graph, partitions


def _update_with_prob_graph(old: gt.Graph, new: gt.Graph):
    """Update a graph with a new one containing edge probabilities.

    Adds any new edge found in the probability graph to the old graph.
    Adds edge probability as a new edge property "edge_prob".
    Adds a categorical edge property to distinguish between:
    - 0: edges which were in the old graph and remain in the new one
    - 1: edges which were not in the old graph and are in the new one
    - 2: edges which were in the old graph and are not in the new one
    """

    probs_map: gt.EdgePropertyMap = old.new_edge_property("float", val=0)
    group_map: gt.EdgePropertyMap = old.new_edge_property("int", val=2)

    # TODO: is it an issue to add edges in the loop?
    for new_edge in new.edges():
        old_edge = old.edge(new_edge.source(), new_edge.target())  # TODO: Type hint?

        if old_edge is not None:
            group_map[old_edge] = 0
        else:
            old_edge = old.add_edge(new_edge.source(), new_edge.target())
            group_map[old_edge] = 1

        probs_map[old_edge] = new.ep.eprob[new_edge]

    # prob_map: gt.EdgePropertyMap = old.new_edge_property("float", val=1)
    # for e in old.edges():
    #     if new.edge(e.source(), e.target()) is not None:
    #         prob_map[e] = new.ep.eprob[new.edge(e.source(), e.target())]
    # old.ep["edge_prob"] = prob_map

    old.ep["edge_prob"] = probs_map
    old.ep["edge_group"] = group_map

    if any(group_map.get_array() == 2):
        print(
            "Warning: Some edges were removed from the graph.",
            "Check the edge_group property for more information.",
        )


def _add_vertex_clusters(graph: gt.Graph, state: gt.MixedMeasuredBlockState) -> None:
    """Add vertex annotations corresponding to the hierarchical clustering levels."""
    # NOTE: Assumes that the nodes are sorted.

    # Fetch the underlying nested block state
    blocks: gt.NestedBlockState = cast(gt.NestedBlockState, state.get_block_state())

    # Project partitions of the block state to vertex level
    num_vertices: int = state.get_graph().num_vertices()
    groups: np.ndarray = np.zeros((num_vertices, len(blocks.get_bs())), dtype=int)
    for level in range(len(blocks.get_bs())):
        groups[:, level] = blocks.project_partition(level, 0).get_array()

    # Rename partitions to consecutive integers
    levels: pl.DataFrame = pl.DataFrame(groups)
    for col in levels.columns:
        levels = levels.with_columns(pl.col(col).rank("dense"))

    # Save all levels with at least two clusters as vertex properties
    for col in levels.columns:
        if levels[col].n_unique() == 1:
            continue  # Not break to avoid assuming that df columns are sorted
        colname: str = f"level_({col.split('_')[1]})"
        graph.vp[colname] = graph.new_vp("int", levels[col])


def _add_edge_clusters(graph: gt.Graph) -> None:
    """Project node clusters on the edges where both nodes are in the same cluster."""

    # Compute the number of levels and initialize that many edge property maps
    num_levels: int = len([c for c in graph.vp.keys() if c.startswith("level_")])
    for i in range(num_levels):
        graph.ep[f"level_({i})"] = graph.new_edge_property("int", 0)

    # For each edge, check if the bins are in the same cluster at each level
    for edge in graph.edges():
        source, target = edge.source(), edge.target()

        for i in range(num_levels):
            clust_source: int = graph.vp[f"level_({i})"][source]
            clust_target: int = graph.vp[f"level_({i})"][target]

            if clust_source == clust_target:
                graph.ep[f"level_({i})"][edge] = clust_source


class HiconaGraph:
    """Graph representation of a portion of 3D chromatin conformation data.

    A wrapper class for the :class:`graph_tool.Graph` class, implementing methods
    which are specific for Hi-C-like data, which avoids having to directly interface
    with the graph object for specialized operations.

    Parameters
    ----------
    bins : polars.DataFrame, pandas.DataFrame or a interable of either.
        A dataframe containing bin information, provided in full or in chunks.
        The dataframe must contain at least the columns `node_id`, `chrom`, `start`
        and `end`.
    pixels: polars.DataFrame, pandas.DataFrame or a interable of either.
        A dataframe containing pixel information, provided in full or in chunks.
        The dataframe must contain at least the columns `bin1_id`, `bin2_id` and
        `count`.
    default_link : int, optional
        Default value for the genomic link property, e.i. default values for
        genomically contiguous bins if they do not have an edge already. Default is 1.


    Warning
    -------
    This class requires loading all the bins and pixels into memory, which can be
    memory-intensive, especially at high resolutions. Consider subsetting the data
    to a chromosome or a region of interest before creating the class instance.
    """

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
            chunk = chunk.filter(pl.col("bin1_id") != pl.col("bin2_id"))  # Rm loops
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

    @overload
    def get_bins(
        self,
        *,
        drop_node_id: bool = ...,
        bare: bool = ...,
        chunk_size: int = ...,
    ) -> "PlChunks": ...

    @overload
    def get_bins(
        self,
        *,
        drop_node_id: bool = ...,
        bare: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["polars"],
    ) -> "PlChunks": ...

    @overload
    def get_bins(
        self,
        *,
        drop_node_id: bool = ...,
        bare: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["pandas"],
    ) -> "PdChunks": ...

    def get_bins(
        self,
        *,
        drop_node_id: bool = True,
        bare: bool = False,
        chunk_size: int = 10_000_000,
        dtype: DfDtype = "polars",
    ) -> "DfChunks":
        """Return the nodes as an iterable of bins.

        Return the nodes from the graph as an iterable of bed-like dataframes.
        All vertex properties are saved as columns, both those with which the
        graph was initially generated, as well as those added at runtime.

        Parameters
        ----------
        drop_node_id : bool, optional
            Whether to drop the `node_id` column. Default is True.
        bare : bool, optional
            Whether to return only the columns `bin_id`, `chrom`, `start` and `end`.
            Default is False.
        chunk_size : int, optional
            Max number of bins per chunk. Default is 10_000_000.
        dtype : {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is "polars".

        Returns
        -------
        A generator of pandas or polars dataframes.
            Bin chunks.

        """

        node_chunks: PlChunks = _iter_bin_chunks(self._graph, chunk_size)

        if drop_node_id:
            node_chunks = (c.drop("node_id") for c in node_chunks)

        if bare:
            bare_cols: tuple[str, ...] = ("bin_id", "chrom", "start", "end")
            node_chunks = (c.select(bare_cols) for c in node_chunks)

        return convert(node_chunks, dtype)

    @overload
    def get_pixels(
        self,
        *,
        keep_genomic: bool = ...,
        chunk_size: int = ...,
    ) -> "PlChunks": ...

    @overload
    def get_pixels(
        self,
        *,
        keep_genomic: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["polars"],
    ) -> "PlChunks": ...

    @overload
    def get_pixels(
        self,
        *,
        keep_genomic: bool = ...,
        chunk_size: int = ...,
        dtype: Literal["pandas"],
    ) -> "PdChunks": ...

    def get_pixels(
        self,
        *,
        keep_genomic: bool = False,
        chunk_size: int = 10_000_000,
        dtype: DfDtype = "polars",
    ) -> "DfChunks":
        """Return the edges as an iterable of pixels.

        Return the edges from the graph as an iterable of bed-like dataframes.
        All edge properties are saved as columns, both those with which the
        graph was initially generated, as well as those added at runtime.

        Parameters
        ----------
        keep_genomic : bool, optional
            Whether to keep the genomic-link pixels added during graph creation.
            Default is False.
        chunk_size : int, optional
            Max number of pixels per chunk. Default is 10_000_000.
        dtype : {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is "polars".

        Returns
        -------
        A generator of pandas or polars dataframes.
            Pixel chunks.

        """

        id_table: pl.DataFrame = pl.concat(_iter_bin_chunks(self._graph, chunk_size))
        edge_chunks: PlChunks = _iter_pix_chunks(self._graph, chunk_size)
        edge_chunks = (_node_to_bin_id(c, id_table) for c in edge_chunks)
        if not keep_genomic:
            edge_chunks = (
                c.filter(pl.col("genomic") == 0).drop("genomic") for c in edge_chunks
            )

        return convert(edge_chunks, dtype)

    @classmethod
    def from_pixel_table(
        cls, table: "PixelTable", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a PixelTable instance.

        Parameters
        ----------
        table : PixelTable
            PixelTable instance to create the graph from.
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        HiconaGraph instance

        """

        pixels: "PlChunks" = table.get_chunks(region, dtype="polars")
        bins: pl.DataFrame = table.bins.get_dataframe(region, dtype="polars")
        return cls(bins=bins, pixels=pixels)

    @classmethod
    def from_cooler(
        cls, handle: "HiconaCooler", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a HiconaGraph from a HiconaCooler instance.

        Parameters
        ----------
        handle : HiconaCooler
            Cooler file handler.
        region : str, optional
            Genomic region of interest in the format "chr:start-end" or "chr".

        Returns
        -------
        HiconaGraph instance

        """

        pixels: "PixelTable" = handle.get_pixel_table(region)
        return cls.from_pixel_table(pixels, region=region)

    def hierarchical_clustering(
        self,
        force_niter: int = 50_000,
        mcmc_niter: int = 50,
        equil_kwargs: dict[str, Any] | None = None,
        rand_seed: int = 42,
    ) -> None:
        """Compute hierarchical clustering of the bins based on the pixels.

        Perform network reconstruction and hierarchical clustering of the bins.
        The method uses a :class:`graph_tool.MixedMeasuredBlockState` object to
        represent the state of the Markov chain, therefore assuming independent
        edge noise when modeling.
        # TODO: Explain better and add citation to paper.

        # TODO: Document parameters once the interface becomes stable.
        """

        # Create initial block state
        max_value: int = self._graph.ep.count.get_array().max()
        n: gt.EdgePropertyMap = self._graph.new_ep("int", max_value)
        x: gt.EdgePropertyMap = self._graph.new_ep("int", self._graph.ep.count.copy())
        state = gt.MixedMeasuredBlockState(self._graph, n=n, x=x)

        # TODO: leave n_default=1, x_default=0?
        # TODO: Leave fn_params and fp_params as default?

        # max_num_edges: int = (
        #     self.graph.num_vertices() * (self.graph.num_vertices() - 1) // 2
        # )
        # print(max_value)
        # N = (
        #     self.graph.num_edges() * max_value
        #     + (max_num_edges - self.graph.num_edges()) * 1
        # )  # TODO: maybe put in terms of maps
        # X = self.graph.ep.count.fa.sum()
        # print(N)
        # print(X)

        # Set random state for reproducibility
        np.random.seed(rand_seed)
        gt.seed_rng(rand_seed)

        # Calibrate the state
        equil_kwargs = equil_kwargs or {"wait": 1000, "mcmc_args": {"niter": 10}}
        gt.mcmc_equilibrate(state, **equil_kwargs)

        # Actual hierarchical clustering
        callback, new_graph, partitions = _get_callback_func()
        gt.mcmc_equilibrate(
            state,
            force_niter=force_niter,
            mcmc_args={"niter": mcmc_niter},
            callback=callback,
        )

        _update_with_prob_graph(self._graph, new_graph[0])
        _add_vertex_clusters(self._graph, state)
        _add_edge_clusters(self._graph)

        # TODO: Max marginal? Node marginal in thiago
        # TODO: Allow to decide hierarchy level to project on instead of always 0
        # 1 entropy per level, bstate.entropy
        # Regenerate block state (same partition, same entropy)
