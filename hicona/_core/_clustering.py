"""All network clustering steps and substeps, grouped for clarity and logging."""

import logging
from functools import partial
from typing import Callable, Literal, cast

import graph_tool.all as gt
import numpy as np
import polars as pl

from .._utils.graph_ops import add_df_as_vp

BIN_PROB_COL: str = "bin_prob"
PIX_PROB_COL: str = "pix_prob"
PIX_GROUP_COL: str = "pix_group"
LEVEL_COL: str = "level_({})"


# TODO: Add time stamp to logger
LOGGER: logging.Logger = logging.getLogger("Clustering")


##############################################################################
######################## Marginals computing functions #######################
##############################################################################


def _get_callback_func() -> tuple[partial, list[gt.Graph], list]:
    """Create partial function to use as callback in hierarchical_clustering."""

    def callback_func(
        state: gt.MixedMeasuredBlockState,
        graph: list[gt.Graph],
        parts: list[gt.PropertyArray],
    ):
        new_graph = state.collect_marginal(graph[0] if len(graph) > 0 else None)
        if not graph:
            graph.append(new_graph)
        else:
            graph[0] = new_graph

        bstate = cast(gt.NestedBlockState, state.get_block_state())
        parts.append(bstate.levels[0].b.a.copy())

    new_graph: list[gt.Graph] = []  # In list since the argument must be mutable
    partitions: list[gt.PropertyArray] = []
    callback: partial = partial(
        callback_func,
        graph=new_graph,
        parts=partitions,
    )

    return callback, new_graph, partitions


def _add_pix_marginals(old: gt.Graph, new: gt.Graph):
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

    for new_edge in new.edges():
        old_edge = old.edge(new_edge.source(), new_edge.target())  # TODO: Type hint?

        if old_edge is not None:
            group_map[old_edge] = 0
        else:
            old_edge = old.add_edge(new_edge.source(), new_edge.target())
            group_map[old_edge] = 1

        probs_map[old_edge] = new.ep.eprob[new_edge]

    old.ep[PIX_PROB_COL] = probs_map
    old.ep[PIX_GROUP_COL] = group_map

    if any(group_map.get_array() == 2):
        LOGGER.warning(
            "Some edges were removed during reconstruction. "
            "Returned state will not match the initial graph."
        )


def _add_bin_marginals(graph: gt.Graph, partition_state: gt.PartitionModeState):
    """Project maximum probability partition onto the nodes of the graph."""
    # TODO: there is probably a vectorized, more efficient way
    bin_probs = [np.max(x) / np.sum(x) for x in partition_state.get_marginal(graph)]
    graph.vp[BIN_PROB_COL] = graph.new_vp("float", vals=bin_probs)


def marginals_no(graph: gt.Graph, state: gt.NestedBlockState) -> gt.NestedBlockState:
    """Simply return the model. Needed for compatibility."""
    LOGGER.debug("Simply returning the state")
    return state


def marginals_bin(graph: gt.Graph, state: gt.NestedBlockState) -> gt.NestedBlockState:
    """Compute bin marginals by equilibrating the initial clustering model."""

    LOGGER.info("Computing bin marginals (equilibration)")

    LOGGER.debug("Equilibrating chain")
    partitions = []
    gt.mcmc_equilibrate(
        state,
        force_niter=1000,
        mcmc_args={"niter": 10},
        callback=lambda s: partitions.append(s.get_bs()),
    )

    LOGGER.debug("Adding bin marginals")
    partition_state = gt.PartitionModeState(partitions, nested=True, converge=True)
    _add_bin_marginals(graph, partition_state)

    return state.copy(bs=partition_state.get_max_nested())


def marginals_pix(graph: gt.Graph, state: gt.NestedBlockState) -> gt.NestedBlockState:
    """Computed pixel marginals using a mixed measured stochastic block model."""

    LOGGER.info("Computing bin and pixels marginals (network reconstruction)")

    LOGGER.debug("Transforming counts to integers")
    MULTIPLIER: int = 1_000
    n_values: np.ndarray = graph.ep.count.a
    if not str(n_values.dtype).startswith("int"):
        n_values = (np.log(n_values + 1) * MULTIPLIER).astype(int)
    LOGGER.debug(f"Before: min={graph.ep.count.a.min()}, max={graph.ep.count.a.max()}")
    LOGGER.debug(f"After: min={n_values.min()}, max={n_values.max()}")

    LOGGER.debug("Setting up mixed measured block state")
    n_default = n_values.max()
    x_default = 0
    n = graph.new_edge_property("int", val=n_default)
    x = graph.new_edge_property("int", vals=n_values)

    mixed_state = gt.MixedMeasuredBlockState(
        graph,
        n=n,
        n_default=n_default,
        x=x,
        x_default=x_default,
        state_args={"bs": state.get_bs()},
    )

    callback, new_graph, partitions = _get_callback_func()

    LOGGER.debug("Reconstructing network")
    gt.mcmc_equilibrate(
        mixed_state,
        force_niter=50000,
        mcmc_args={"niter": 10},
        callback=callback,
    )

    LOGGER.debug("Adding bin marginals")
    partition_state = gt.PartitionModeState(partitions, converge=True)
    _add_bin_marginals(graph, partition_state)

    LOGGER.debug("Adding pixel marginals")
    _add_pix_marginals(graph, new_graph[0])

    return cast(gt.NestedBlockState, mixed_state.get_block_state())


MARGINALS_FUNCS: dict[str, Callable] = {
    "no": marginals_no,
    "bins": marginals_bin,
    "pixels": marginals_pix,
}

##############################################################################
######################## General clustering functions ########################
##############################################################################


def dl_anneal_clustering(graph: gt.Graph, value_col: str) -> gt.NestedBlockState:
    """Clustering using description length minimization + simulated annealing."""

    # May not need to create a copy of the edge property map
    # For now leaving to avoid breaking changes, potentially clean up later

    score: gt.EdgePropertyMap = graph.new_ep("double", vals=graph.ep[value_col].a)
    if score.a.min() < 0 or score.a.max() > 1:
        raise ValueError(
            f"""Score values are in the range ['{score.a.min()}', '{score.a.max()}']"""
            """ but should be in the range [0,1]."""
        )

    LOGGER.info("Computing rough clustering (description length minimization)")
    state: gt.NestedBlockState = gt.minimize_nested_blockmodel_dl(
        graph,
        state_args={"recs": [score], "rec_types": ["real-normal"]},
        multilevel_mcmc_args={"niter": 10},
    )

    LOGGER.info("Refining clustering (simulated annealing)")
    gt.mcmc_anneal(
        state,
        beta_range=(1, 10),
        niter=1000,
        mcmc_equilibrate_args={"force_niter": 10},
    )

    return state


def add_bin_clustering(graph: gt.Graph, state: gt.NestedBlockState):
    """Convert the clustering from a nested block state into dataframe format."""

    # Create an array with rows=nodes, cols=clustering levels
    num_vertices: int = len(state.get_bs()[0])
    num_clusters: int = -1
    num_levels: int = len(state.get_bs())
    groups_array: np.ndarray = np.zeros((num_vertices, num_levels), dtype=int)

    LOGGER.debug(f"Starting with {num_levels} clustering levels")

    # Remove bad trailing levels (repetitive or one big cluster)
    for level in range(num_levels):
        # Project partitions of the block state to vertex level
        level_groups: np.ndarray = state.project_partition(level, 0).get_array()
        level_clusts: int = len(np.unique(level_groups))

        same_cluster_num: bool = level_clusts == num_clusters
        only_one_cluster: bool = level_clusts == 1 and level != 0
        if same_cluster_num or only_one_cluster:
            groups_array = groups_array[:, :level]
            break

        num_clusters = level_clusts
        groups_array[:, level] = level_groups

    LOGGER.debug(f"{groups_array.shape[1]} levels left after removing trailing")

    # Remove bad heading levels (all bins on their own)
    while True:
        first_all_split: bool = len(np.unique(groups_array[:, 0])) == num_vertices
        multiple_levels: bool = len(groups_array) > 1
        if not (multiple_levels and first_all_split):
            break
        groups_array = groups_array[:, 1:]

    LOGGER.debug(f"{groups_array.shape[1]} levels left after removing heading")

    LOGGER.debug("Renaming levels")
    bin_clustering: pl.DataFrame = (
        pl.DataFrame(groups_array)
        .with_columns(pl.all().rank("dense"))  # Rename to consecutive integers
        .rename(lambda c: LEVEL_COL.format(c.split("_")[1]))
    )

    LOGGER.debug("Adding clustering")
    add_df_as_vp(graph, bin_clustering)


def project_clustering(graph: gt.Graph):
    """Project node clusters on the edges where both nodes are in the same cluster."""

    LOGGER.debug("Projecting bin clustering onto pixels")

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


##############################################################################
########################## Main clustering function ##########################
##############################################################################


def compute_clustering(
    graph: gt.Graph,
    on: str,
    marginals: Literal["no", "bins", "pixels"],
    seed: int,
    logging_level: str,  # TODO: Change to logging level dtype
) -> gt.NestedBlockState:

    LOGGER.setLevel(logging_level)

    if "level_(0)" in graph.vp:
        raise ValueError("Cannot perform clustering twice on the same table.")

    LOGGER.debug("Fetching marginals function")
    marginals_function = MARGINALS_FUNCS.get(marginals)
    if not marginals_function:
        raise ValueError(f"`{marginals}` is not a valid option for marginals.")

    LOGGER.debug("Setting rng seed")
    np.random.seed(seed)
    gt.seed_rng(seed)

    LOGGER.info(f"Started graph clustering (marginals={marginals}, seed={seed})")
    state: gt.NestedBlockState = dl_anneal_clustering(graph, on)
    state = marginals_function(graph, state)

    LOGGER.info("Projecting clustering on the graph")
    add_bin_clustering(graph, state)
    project_clustering(graph)

    return state
