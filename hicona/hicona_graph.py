"""graph_tool.Graph specialized subclass for Hi-C data network analysis.

Extend the graph_tool.Graph class, without overwriting any of its methods,
implementing algorithms for network analysis of pixel tables.
"""

from collections.abc import Iterable
import time

import numpy as np
import pandas as pd
import graph_tool.all as gt

from .hicona_cooler import HiconaCooler
from .hicona_table import HiconaTable
from .utils.misc import annotation_combinations
from .utils.table_ops import pd2gt_dtype

__all__ = ["HiconaGraph"]


class HiconaGraph(gt.Graph):
    """Graph-tool `Graph` specialized for Hi-C data network analysis.

    This class extends the `graph_tool.Graph` class, adding methods to
    facilitate the analysis of Hi-C data networks. All the methods from the
    parent class are still available and are not overwritten.

    Parameters
    ----------
    table : HiconaTable
        Pixel table to create the graph from.
    query : str, optional
        Pandas-like query string to filter the pixel table. (Default is None)
    """

    def __init__(self, table: HiconaTable, query: str | None = None):

        # Initialize the object
        super().__init__(directed=False)

        # Add the new properties
        self._ids_table = pd.DataFrame(columns=["bin_id", "node_id"], dtype=int)
        self._table = table
        self._has_genomic_links = False

        # Fetch the data to create the graph
        # NOTE: might be expensive if the table is very big but chunks are not
        #       an option, as the maps must be created in one go
        data_df = table.dataframe(query=query)

        # Create the initial bin to node id conversion table
        self._update_ids_table(data_df["bin1_id"])
        self._update_ids_table(data_df["bin2_id"])

        # Add edges to the graph and edge properties
        edge_list, eprops = self._to_edge_list(data_df)
        self.add_edge_list(edge_list, eprops=eprops)
        # TODO: Maybe remove bin1_id and bin2_id from the edge properties?

        # Add vertex properties
        self._refresh_vertex_properties()

    def _update_ids_table(self, new_bins: Iterable[int]):
        """Update the ids_table with new bin ids."""

        # List of new bins to add
        old_bins = set(self._ids_table["bin_id"])
        new_bins = set(new_bins)
        add_bins = sorted(new_bins - old_bins)

        # New table chunk
        last_node = self._ids_table["node_id"].max()
        last_node = last_node if not np.isnan(last_node) else -1
        add_nodes = list(range(last_node + 1, last_node + len(add_bins) + 1))
        new_chunk = pd.DataFrame({"bin_id": add_bins, "node_id": add_nodes})

        # Update the table
        self._ids_table = pd.concat([self._ids_table, new_chunk], ignore_index=True)

    def _to_edge_list(self, dataf: pd.DataFrame) -> tuple[np.ndarray, list[tuple]]:
        """Convert the pixel table to an edge list and edge properties."""

        # Merge the data with the ids_table and drop the bin_id columns
        dataf = dataf.merge(self._ids_table, left_on="bin1_id", right_on="bin_id")
        dataf = dataf.merge(self._ids_table, left_on="bin2_id", right_on="bin_id")
        dataf.drop(columns=["bin_id_x", "bin_id_y"], inplace=True)

        # Create the edge list and edge properties
        prop_cols = [c for c in dataf.columns if c not in ["node_id_x", "node_id_y"]]
        edge_prop = [(p, pd2gt_dtype(dataf[p].dtype.name)) for p in prop_cols]
        edge_list = dataf[["node_id_x", "node_id_y"] + prop_cols].values

        return (edge_list, edge_prop)

    def _fetch_bins(self) -> pd.DataFrame:
        """Fetch the bin table from the parent cooler."""

        cooler_uri = self._table.uris.cooler_uri()
        handle = HiconaCooler(cooler_uri)
        return handle.bins()[:]  # type: ignore

    def _refresh_vertex_properties(self):
        """Update the vertex properties with the latest bin data."""

        # Fetch the latest bin data
        bins = self._fetch_bins()
        ids_list = self._ids_table["bin_id"].to_list()
        annot_df = bins.filter(items=ids_list, axis=0)  # pylint: disable=no-member
        annot_df.reset_index(inplace=True, names="bin_id")

        # Update the properties
        vprops = [(p, pd2gt_dtype(annot_df[p].dtype.name)) for p in annot_df]
        for name, dtype in vprops:
            vprop = self.new_vertex_property(dtype, annot_df[name])
            self.vp[name] = vprop

    @property
    def ids_table(self):
        """Return the conversion table from bin_id to node_id."""
        return self._ids_table

    @property
    def table(self):
        """Return the pixel table used to create the graph."""
        return self._table

    def _node_statistics(self, stat, mask):
        """Compute node-level statistic (only required nodes if possible)."""

        # TODO: Look for an alternative to the big and ugly switch case
        if stat == "ave_degree":
            vlist = mask.nonzero()[0]
            stats_list = self.get_total_degrees(vlist)
        elif stat == "betweenness":
            weight_map = self.ep["exp_ratio"]
            stats_list, _ = gt.betweenness(self, weight=weight_map)
            stats_list = stats_list.get_array()
        elif stat == "clustering_coeff":
            weight_map = self.ep["exp_ratio"]
            stats_list = gt.local_clustering(self, weight=weight_map)
            stats_list = stats_list.get_array()
        else:
            raise ValueError(f"{stat} is not among the supported statistics.")

        return stats_list

    def _create_ann_ohe(self, attr_list):
        """Create array with ohe of each annotation as rows."""

        ohe = np.vstack([self.vp[a].get_array() for a in attr_list if a])
        if len(ohe) == 1:
            ohe = np.vstack([ohe, np.ones(self.num_vertices())])
        return ohe

    def _compute_perms(self, annot_ohe, node_vals, perms, rand_seed):
        """Return p-value for H1: stat(annA) - stat(annB) > 0"""
        # TODO: Try multiple permutations at one to speed up (memory cost?)

        def _perm_fc(values, ohe, rng=None):
            """Compute statistic fold change for a single permutation."""
            ohe = ohe[:, rng.permutation(ohe.shape[1])] if rng else ohe
            val_a, val_b = (ohe * values).sum(axis=1) / ohe.sum(axis=1)
            return np.log2(val_a / val_b)

        # Filter only for nodes with at least one of the annotations
        perm_mask = annot_ohe.sum(axis=0) != 0
        stat_vals = node_vals[perm_mask]
        annot_ohe = annot_ohe[:, perm_mask]
        del perm_mask

        # Return none if at least one of the annotation is fully zero
        if np.prod(annot_ohe.sum(axis=1)) == 0:
            return np.nan

        # Compute p-value by comparing original value and generated histogram
        rng = np.random.default_rng(rand_seed)
        original = _perm_fc(stat_vals, annot_ohe)
        permuted = [_perm_fc(stat_vals, annot_ohe, rng) for _ in range(perms)]
        result = ((original < permuted).sum() + 1) / (perms + 1)

        return result

    def permute_annotations(
        self,
        ann_list: str | Iterable[str],
        stat: str,
        num_perms: int = 1000,
        seed: int = 94206,
    ) -> pd.DataFrame:
        """Compute p-values for node label permutations.

        Given a subset of node annotations of the graph, first compute all
        combinations of one (annA vs universe, e.i. all nodes in the graph)
        or two annotations (annA vs annB), then for each of them compute the
        p-value for the test :math:`H1: log_2(stat(annA) / stat(annB)) > 0`
        using node label permutations (randomly permute the attributes among
        nodes without changing graph structure).

        Annotations must be in OHE form.
        The annotations can be overlapping, in which case the number of nodes
        with overlapping annotations remains constant (e.i. if N nodes have
        both annotations in the original set, N nodes will have both
        annotations in each individual permutation).

        Parameters
        ----------
        ann_list : str or Iterable[str]
            List of annotations (or single annotation) to permute.
        stat : str
            Node-level statistic to use in the test. Currently supported:

            - ``ave_degree``: average total node degree
            - ``betweenness``: node betweenness centrality
            - ``clustering_coeff``: local clustering coefficient

        num_perms : int, optional
            Number of permutation iterations to perform. (Default is 1000)
        seed : int, optional
            Rng seed for the permutations. (Default is 94206)

        Returns
        -------
        pd.DataFrame:
            dictionary containing permutation parameters and results
        """

        # TODO: Add pvalue correction
        # TODO: Maybe add more values in return (magnitude/fold change?)
        # TODO: Maybe add number of nodes per annotation and overlap
        # TODO: Maybe create an entire object to return and plot the results?

        annos = [ann_list, None] if isinstance(ann_list, str) else ann_list
        node_vals = self._node_statistics(stat, np.ones(self.num_vertices()))

        # Compute pvalues for all 1 and 2 annotation pairs
        res_dicts = []
        for pair in annotation_combinations(annos):
            pair_ohe = self._create_ann_ohe(pair)
            pval = self._compute_perms(pair_ohe, node_vals, num_perms, seed)
            res_dicts.append(
                {
                    "a": pair[0],
                    "b": pair[1] if len(pair) == 2 else "universe",
                    "pval": pval,
                    "stat": stat,
                    "num_perms": num_perms,
                }
            )

        return pd.DataFrame(res_dicts)

    def add_chromsomal_edges(self):
        """Add edges between consecutive genomic regions.

        Add edges between nodes representing consecutive genomic regions, i.e.
        add an edge between nodes with consecutive bin ids.
        A new edge property map is added to the graph to store whether an edge
        is chromosomal or not.

        NOTE: Currently the start end end bin for consecutive chromosomes are
        joined. Until this is fixed, be sure to provide an intrachromosomal
        pixel table.

        NOTE: Older edge maps are filled with the default value for those
        maps. This behaviour might lead to unexpected results and will be
        changed in the future.
        """

        min_bin: int = self.ids_table["bin_id"].min()
        max_bin: int = self.ids_table["bin_id"].max()

        self._update_ids_table(range(min_bin, max_bin))

        # TODO: check that only interchromosomal edges are added
        new_edges = [(i, i + 1) for i in range(min_bin, max_bin)]
        new_edges = pd.DataFrame(new_edges, columns=["bin1_id", "bin2_id"])

        # Add the new edges to the graph
        edge_list, _ = self._to_edge_list(new_edges)
        self.add_edge_list(edge_list)

        # Update property maps
        # TODO: maybe also update the edge properties?
        self._refresh_vertex_properties()

        # Add map to state whether an edge is chromosomal or not
        is_chromosomal_map = [False] * self.num_edges()
        is_chromosomal_map[-len(new_edges) :] = [True] * len(new_edges)

        link_prop = self.new_edge_property("bool", vals=is_chromosomal_map)
        self.ep["is_genomic_link"] = link_prop

        self._has_genomic_links = True

    def compute_clustering(
        self,
        min_steps: int = 10,
        genomic_links: bool = True,
        equilibrate: bool = True,
        annotate: bool = False,
    ) -> pd.DataFrame:
        """Computed nested blockmodel clustering of the graph.

        Return hierarchical clustering of the graph using a nested blockmodel.
        The output is a `pd.DataFrame` with the cluster labels for each node
        at each level of the hierarchy, where level `0` is the level with the
        highest number of clusters, while level `n` is the level with the
        lowest number of clusters.

        For more detail on the clustering algorithm see the `graph_tool`
        documentation.

        Parameters
        ----------
        min_steps : int, optional
            Number of iterations of the entropy minimization step.
            (Default is 10)
        genomic_links : bool, optional
            Whether to add edges between genomically consecutive bins.
            (Default is True)
        equilibrate : bool, optional
            Whether to equilibrate the model after the minimization step.
            (Default is True)
        annotate : bool, optional
            Whether to add bin annotations to the output DataFrame.
            (Default is False)

        Returns
        -------
        pd.DataFrame:
            DataFrame with the cluster labels for each node at each level.
            Optionally, the DataFrame can also contain bin annotations.
        """

        print("Starting clustering...")
        start_time = time.time()

        if genomic_links and not self._has_genomic_links:
            print("Adding chromosomal edges...")
            self.add_chromsomal_edges()

        print("Starting model creation...")
        state_args = {"deg_corr": True}  # Usually lower entropy
        state_type = gt.BlockState  # Default state type

        if genomic_links:
            state_args.update({"ec": self.ep.is_genomic_link, "layers": True})
            state_type = gt.LayeredBlockState

        state_dict = {"base_type": state_type, "state_args": state_args}

        model: gt.NestedBlockState | None = None
        for _ in range(min_steps):
            new_model = gt.minimize_nested_blockmodel_dl(self, state_args=state_dict)
            if not model or new_model.entropy() < model.entropy():
                model = new_model

        # This should in theory never happen
        if not model:
            raise ValueError("Model could not be created.")

        if equilibrate:
            print("Equilibrating the model (might take a while)...")
            gt.mcmc_equilibrate(model, wait=1000, mcmc_args={"niter": 10})

        end_time = time.time()
        print(f"Clustering took {end_time - start_time} seconds.")

        # From each level, project the partition to the vertex level
        # NOTE: Labels are sorted by node, not by cluster
        num_levels = len(model.get_bs())
        levels = np.zeros((self.num_vertices(), num_levels), dtype=int)
        for level in range(num_levels):
            levels[:, level] = model.project_partition(level, 0).get_array()

        # Make each column a category from 0 to n
        levels = pd.DataFrame(levels).astype("category")
        for col in levels.columns:
            num_cat = len(levels[col].cat.categories)
            new_cat = [str(x) for x in range(num_cat)]
            levels[col] = levels[col].cat.rename_categories(new_cat)

        # Only keep levels with more than one cluster
        cols_to_drop = [c for c in levels if len(set(levels[c])) == 1]
        levels.drop(columns=cols_to_drop, inplace=True)

        # Convert node ids to extended bin form
        if annotate:
            merge_kws = {"how": "left", "left_index": True, "right_on": "node_id"}
            levels = pd.merge(levels, self._ids_table, **merge_kws)

            merge_kws = {"how": "left", "left_on": "bin_id", "right_index": True}
            bins = self._fetch_bins()[["chrom", "start", "end"]]
            levels = pd.merge(levels, bins, **merge_kws)

            levels.drop(columns=["bin_id", "node_id"], inplace=True)

        return levels
