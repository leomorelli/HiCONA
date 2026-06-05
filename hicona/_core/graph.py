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

from typing import TYPE_CHECKING, Literal, overload

import graph_tool.all as gt  # type: ignore
import polars as pl

from .._utils.chunked_ops import convert, rechunk, to_iterable
from .._utils.graph_ops import (
    add_df_as_ep,
    add_df_as_vp,
    iter_bin_chunks,
    iter_pix_chunks,
)
from ._clustering import compute_clustering

if TYPE_CHECKING:
    from .._utils.df_dtypes import (
        DataFrame,
        DfChunks,
        DfDtype,
        DfStream,
        PdChunks,
        PlChunks,
    )
    from .cooler import HiconaCooler
    from .table import PixelTable


__all__ = ["HiconaGraph"]


#########################################################################################
#################### UTILITY FUNCTIONS FOR GENERAL GRAPH MANIPULATION ###################
#########################################################################################


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


class HiconaGraph:
    """Graph representation of 3D chromatin conformation data.

    Handler class for a graph where the vertices correspond to the bins, the
    edges to the pixels. Any column, in either bin or pixel table, is added
    as a vertex or edge property, respectively.

    If two genomically contiguous bins do not have any edge (= pixel) among
    them, an edge with a default value, called ``genomic_link``, is added
    among them. This is necessary for network clustering purposes.

    This class wraps an instance of the :py:class:`graph_tool.Graph` class
    and implements methods which are specific for Hi-C-like data. This avoids
    having to interface directly with the low level API of graph-tool for
    specialized operations. Any network operation which is not directly
    implemented by HiCONA can be performed by accessing directly the
    :py:class:`graph_tool.Graph` instance. This is possibile because
    :py:class:`HiconaGraph` does not store any data itself.

    Parameters
    ----------
    bins : polars.DataFrame, pandas.DataFrame or an interable of either.
        A dataframe containing bin information, provided in full or in chunks.
        The dataframe must contain at least the columns ``bin_id``, ``chrom``,
        ``start`` and ``end``.
    pixels: polars.DataFrame, pandas.DataFrame or an interable of either.
        A dataframe containing pixel information, provided in full or in chunks.
        The dataframe must contain at least the columns ``bin1_id``, ``bin2_id``
        and ``count``.
    default_link : int, optional
        Default value for the genomic link property, e.i. default values for
        genomically contiguous bins if they do not have an edge already.
        Default is ``1``.

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
        add_df_as_vp(graph, bins.drop("node_id"))
        add_df_as_ep(graph, pl.concat(eprops))
        _add_genomic_link(graph, default_link)

        self._graph: gt.Graph = graph

    @property
    def graph(self) -> gt.Graph:
        """Associated :py:class:`graph_tool.Graph` instance.

        For algorithms implemented by HiCONA, call the methods of the
        :py:class:`HiconaGraph` class. For other algorithms, access this object
        directly and use its API directly.

        Returns
        -------
        :py:class:`graph_tool.Graph`
            The associated :py:class:`graph_tool.Graph` instance.

        """
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
        All vertex properties are saved as columns, both those which the
        graph was initially generated with, as well as those added later.

        Parameters
        ----------
        drop_node_id : bool, optional
            Whether to drop the ``node_id`` column. Default is ``True``.
        bare : bool, optional
            Whether to return only the columns ``bin_id``, ``chrom``, ``start``
            and ``end``. Default is ``False``.
        chunk_size : int, optional
            Max number of bins per chunk. Default is ``10_000_000``.
        dtype : one of {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is ``polars``.

        Returns
        -------
        Generator of :py:class:`polars.DataFrame` or :py:class:`pandas.DataFrame`
            Bin data chunks.

        """

        node_chunks: PlChunks = iter_bin_chunks(self._graph, chunk_size)

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

        All edge properties are saved as columns, both those which the
        Return the edges from the graph as an iterable of pixel-like dataframes.
        graph was initially generated with, as well as those added later.

        Parameters
        ----------
        keep_genomic : bool, optional
            Whether to keep the genomic-link pixels added during graph creation.
            Ignored if the graph was reconstructed. Default is ``False``.
        chunk_size : int, optional
            Max number of pixels per chunk. Default is ``10_000_000``.
        dtype : one of {"polars", "pandas"}, optional
            Whether to return the chunks as polars or pandas dataframes.
            Default is ``polars``.

        Returns
        -------
        Generator of :py:class:`polars.DataFrame` or :py:class:`pandas.DataFrame`
            Pixel data chunks.

        """

        id_table: pl.DataFrame = pl.concat(iter_bin_chunks(self._graph, chunk_size))
        edge_chunks: PlChunks = iter_pix_chunks(self._graph, chunk_size)
        edge_chunks = (_node_to_bin_id(c, id_table) for c in edge_chunks)
        reconstructed: bool = "pix_group" in self.graph.ep

        match keep_genomic, reconstructed:
            case True, False:
                pass  # No need to do anything
            case False, False:
                edge_chunks = (
                    c.filter(pl.col("genomic") == 0).drop("genomic")
                    for c in edge_chunks
                )
            case _, True:
                # Currently silently ignored, maybe change behavior
                # from warning import warn
                # warn("Network was reconstructed, ignoring keep_genomic parameter.")
                edge_chunks = (
                    c.drop("genomic").with_columns(pl.col("count").fill_null(0))
                    for c in edge_chunks
                )

        # Since network reconstruction could add edges, pixels could be put of order
        # Explicit sorting before returning them is required to ensure consistent order
        # This is currently a bottleneck since requires loading all pixels at once
        # TODO: potentially fix this by either:
        #  - Implementing out of memory sorting (complicated)
        #  - Fetching pixels in order iterating by node ids (potentially slow)
        edge_chunks = rechunk(
            (pl.concat(edge_chunks).sort(by=["bin1_id", "bin2_id"]),),
            chunk_size,
        )

        return convert(edge_chunks, dtype)

    @classmethod
    def from_pixel_table(
        cls, table: "PixelTable", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a :py:class:`HiconaGraph` starting from a :py:class:`PixelTable` instance.

        Create a graph starting from a :py:class:`PixelTable`. If a genomic region
        is provided both pixels and bins are subsetted to that region.

        Parameters
        ----------
        table : :py:class:`PixelTable`
            Table instance to create the graph from.
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If provided, subset bins and pixels to that region. Default is ``None``.

        Returns
        -------
        :py:class:`HiconaGraph`
            The graph representation of the input table.

        Warning
        -------
        Since providing a genomic region during creation subsets the binning,
        a part of the bins is lost when converting the graph back to a table
        directly. When creating a :py:class:`PixelTable` starting from a
        :py:class:`HiconaGraph`, always provide a full genomic binning on to
        which you merged the bins from the graph. This process will be
        simplified in a future version.

        """

        pixels: "PlChunks" = table.get_chunks(region, dtype="polars")
        bins: pl.DataFrame = table.bins.get_dataframe(region, dtype="polars")
        return cls(bins=bins, pixels=pixels)

    @classmethod
    def from_cooler(
        cls, handle: "HiconaCooler", *, region: str | None = None
    ) -> "HiconaGraph":
        """Create a :py:class:`HiconaGraph` starting from a :py:class:`HiconaCooler` instance.

        Create a graph starting from a :py:class:`HiconaCooler`. If a genomic region
        is provided both pixels and bins are subsetted to that region.

        Parameters
        ----------
        handle : :py:class:`HiconaCooler`
            Cooler file handler.
        region : str, optional
            Genomic region of interest in the format ``chr:start-end`` or ``chr``.
            If provided, subset bins and pixels to that region. Default is ``None``.

        Returns
        -------
        :py:class:`HiconaGraph`
            The graph representation of the input cooler.

        """

        pixels: "PixelTable" = handle.get_pixel_table(region)
        return cls.from_pixel_table(pixels, region=region)

    def compute_clustering(
        self,
        *,
        marginals: Literal["no", "bins", "pixels"] = "no",
        seed: int = 42,
        logging_level: str = "INFO",  # TODO: create logging level type
    ) -> gt.NestedBlockState:
        """Compute hierarchical clustering on the network.

        Compute the hierarchical clustering on the entire network, and save
        the results as edge or vertex properties in the graph.

        The marginals parameter defines which algorithm is used to compute
        the clustering, as well as which probabilities (=marginals) are
        computed:

            - "no": do not compute bins or pixel marginals. Uses description
              length minimization followed by simulated annealing.
            - "bins": compute confidence of assignment of each bin to its
              cluster. Same as "no" but also adds an equilibration step.
            - "pixels": compute both confidence of assignment of each bin to
              its cluster and posterior probability of each edge to be a "real"
              edge (e.i. not the result of random noise and biases). Same as
              "bins" but the equilibration is computed using a mixed measured
              block state.

        # TODO: finish description

        Parameters
        ----------
        marginals : "no", "bins", "pixels"
            Which probabilites to compute. Default is "no".
        seed : int
            Rng seed for reproducibility. Default is 42
        logging_level : valid logging level string
            Console log verbosity level. Default is "INFO".

        Returns
        -------
        gt.NestedBlockState
            The last nested block state computed during clustering.
        """

        # Defer all steps of the procedure to functions in a dedicated file
        # for better organization and managing logging
        return compute_clustering(self._graph, marginals, seed, logging_level)
