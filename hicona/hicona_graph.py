"""graph_tool.Graph specialized subclass for Hi-C data network analysis.

Extend the graph_tool.Graph class, without overwriting any of its methods,
implementing algoritms for network analysis of chromosome-level tables.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import graph_tool.all as gt

from .utils.misc import annotation_combinations
from .utils.table_ops import pd_to_gt_dtype

__all__ = ["HiconaGraph"]


class HiconaGraph(gt.Graph):
    """Specialized Graph subclass for Hi-C data network analysis.

    :py:class:`graph_tool.Graph` subclass created from a chromosome-level
    table, where the nodes are the bins while the edges are the pixels.
    Unless specified otherwise, edges are annotated with all pixel columns.
    Unless specified otherwise, nodes are annotated only with the ``bin_id``;
    in general it is suggested to specify only the required annotations, in
    order to prevent graphs too big for memory. Some commonly used network
    analysis algorithms are implemented as methods.

    Parameters
    ----------
    dataf : :py:class:`DataFrame`
        Chromosome-level pixel table to create the network from.
    to_keep : Iterable[str], optional
        Iterable of pixel table columns to add as edge properties.
        If None, all pixel table columns are kept. (Default is None)
    ann_df : :py:class:`DataFrame`, optional
        DataFrame whose columns are added as node properties.
        Usually a slice of the ``bins`` table. (Default is None)
    """

    def __init__(
        self,
        dataf: pd.DataFrame,
        to_keep: Iterable[str] = None,
        ann_df: pd.DataFrame = None,
    ):
        # List of df columns that will/will NOT be used as edge properties
        to_keep = to_keep if to_keep is not None else dataf.columns.tolist()
        to_drop = [c for c in dataf.columns.tolist() if c not in to_keep]

        # Create id conversion table
        vids = np.sort(
            np.union1d(
                dataf["bin1_id"].unique(),
                dataf["bin2_id"].unique(),
            )
        )
        bin_ind = pd.DataFrame(
            {
                "bin_ids": vids,
                "node_id": np.arange(0, len(vids)),
            }
        )

        # Add columns to the df corresponding to the new reindexed bin ids
        merge_opts = {"how": "left", "right_on": "bin_ids"}
        dataf = dataf.merge(bin_ind, left_on="bin1_id", **merge_opts)
        dataf = dataf.merge(bin_ind, left_on="bin2_id", **merge_opts)
        dataf.drop(to_drop + ["bin_ids_x", "bin_ids_y"], axis=1, inplace=True)

        # Assumed that "node_id_x" and "node_id_y" are the last two columns
        df_cols = dataf.columns.tolist()
        df_cols = df_cols[-2:] + df_cols[:-2]
        dataf = dataf[df_cols]

        # Initialize the object with edge properties
        eprops = [(p, pd_to_gt_dtype(dataf[p].dtype.name)) for p in to_keep]
        super().__init__(dataf.values, directed=False, eprops=eprops)

        # TODO: Currently if ann_df==None it breaks
        # Reduce annotation dataframe to only the nodes in the network
        ann_df = ann_df.filter(items=vids, axis=0)
        ann_df.reset_index(inplace=True)
        ann_df.rename(columns={"index": "bin_id"}, inplace=True)
        assert len(vids) == len(ann_df)

        # Add node annotations via bin dataframe
        vprops = [(p, pd_to_gt_dtype(ann_df[p].dtype.name)) for p in ann_df]
        for name, dtype in vprops:
            vprop = self.new_vertex_property(dtype, ann_df[name])
            self.vp[name] = vprop

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
    ) -> dict:
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
        dict :
            dictionary containing permutation parameters and results
        """

        # TODO: Add pvalue correction
        # TODO: Maybe add more values in return (magnitude/fold change?)
        # TODO: Maybe add number of nodes per annotation and overlap
        # TODO: Maybe create an entire object to return and plot the results?

        ann_list = [ann_list, None] if isinstance(ann_list, str) else ann_list
        node_vals = self._node_statistics(stat, np.ones(self.num_vertices()))

        # Compute pvalues for all 1 and 2 annotation pairs
        res_dicts = []
        for pair in annotation_combinations(ann_list):
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

        return res_dicts
