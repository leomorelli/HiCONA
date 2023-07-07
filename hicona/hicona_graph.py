"""Placeholder
Placeholder
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import graph_tool.all as gt

from .utils import pd_to_gt_dtype, annotation_combinations, console_log

__all__ = ["HiconaGraph"]


# Select rows belonging to either annotation (keep row annotations tied)
# Randomly shuffle the rows
# Also return the number of group assignments mismatching the initial labels
# This is just to get a statistic about proper randomization
# Create graph view?
# Also return aboslute values and percentages


class HiconaGraph(gt.Graph):
    """Placeholder
    Placeholder
    """

    def __init__(
        self,
        dataf: pd.DataFrame,
        to_keep: bool | Iterable[str] = None,
        ann_df: pd.DataFrame = None,
    ):
        # List of df columns that will NOT be used as edge properties
        to_keep = to_keep if to_keep is not None else dataf.columns.tolist()
        to_drop = [c for c in dataf.columns.tolist() if c not in to_keep]

        # Create id conversion table
        vids = np.sort(np.union1d(dataf["bin1_id"].unique(), dataf["bin2_id"].unique()))
        bin_ind = pd.DataFrame({"bin_ids": vids, "node_id": np.arange(0, len(vids))})

        # Add columns to the df corresponding to the new reindexed bin ids
        dataf = dataf.merge(bin_ind, how="left", left_on="bin1_id", right_on="bin_ids")
        dataf = dataf.merge(bin_ind, how="left", left_on="bin2_id", right_on="bin_ids")
        dataf.drop(to_drop + ["bin_ids_x", "bin_ids_y"], axis=1, inplace=True)

        # Assumed that "node_id_x" and "node_id_y" are always the last two columns
        df_cols = dataf.columns.tolist()
        df_cols = df_cols[-2:] + df_cols[:-2]
        dataf = dataf[df_cols]

        # Initialize the object with edge properties
        eprops = [(p, pd_to_gt_dtype(dataf[p].dtype.name)) for p in to_keep]
        super().__init__(dataf.values, eprops=eprops)

        # Reduce annotation dataframe to only the nodes in the network
        filt_ann = ann_df.filter(items=vids, axis=0)
        filt_ann.reset_index(inplace=True)
        filt_ann.rename(columns={"index": "bin_id"}, inplace=True)
        assert len(vids) == len(filt_ann)

        # Add node annotations via bin dataframe
        # TODO: Typing is manually fixed for now, have to define exact conversion
        vprops = [(p, pd_to_gt_dtype(filt_ann[p].dtype.name)) for p in filt_ann]
        for name, dtype in vprops:
            vprop = self.new_vertex_property(dtype, filt_ann[name])
            self.vp[name] = vprop

    def _node_statistics(self, stat, mask):
        """Placeholder
        Placeholder
        """
        # TODO: Look for an alternative to the big and ugly switch case

        if stat == "ave_degree":
            vlist = mask.nonzero()[0]
            stats_list = self.get_out_degrees(vlist)
        elif stat == "betweenness":
            weight_map = self.ep["exp_ratio"]
            stats_list, _ = gt.betweenness(self, weight=weight_map)
        elif stat == "clustering_coeff":
            weight_map = self.ep["exp_ratio"]
            stats_list = gt.local_clustering(self, weight=weight_map).get_array()
        else:
            raise ValueError(f"{stat} is not among the supported statistics.")

        return stats_list

    def _create_ann_ohe(self, attr_list):
        """Placeholder"""
        ohe = np.vstack([self.vp[a].get_array() for a in attr_list if a is not None])
        ohe = np.vstack([ohe, np.ones(self.num_vertices())]) if len(ohe) == 1 else ohe
        return ohe

    def _compute_perms(
        self,
        annot_ohe: np.array,
        node_vals: np.array,
        num_perms: int,
        rand_seed: int,
    ):
        """Return p-value for H1: stat(annA) - stat(annB) > 0"""
        # TODO: Try multiple permutations at one to speed up (memory cost?)

        def _perm_diff(values, ohe, rng=None):
            """Compute average stat difference for a single permutation."""
            ohe = ohe if rng is None else ohe[:, rng.permutation(ohe.shape[1])]
            res = -np.diff((ohe * values).sum(axis=1) / ohe.sum(axis=1))[0]
            return res

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
        original = _perm_diff(stat_vals, annot_ohe)
        permuted = [_perm_diff(stat_vals, annot_ohe, rng) for _ in range(num_perms)]
        result = ((original < permuted).sum() + 1) / (num_perms + 1)

        return result

    def permute_annotations(
        self,
        ann_list: str | Iterable[str],
        statistic: str,
        num_perms: int = 1000,
        seed: int = 94206,
    ):
        """Compute p-values for node label permutations.

        Given a subset of node annotations of the graph, first compute all
        combinations of one (annA vs universe, e.i. all nodes in the graph) or
        two annotations (annA vs annB), then for each of them compute the
        p-value for the test H1: stat(annA) - stat(annB) > 0 using node label
        permutations (randomly permute the attributes among nodes without
        changing graph structure).

        Parameters
        ----------
        ann_list : str | Iterable[str]
            List of annotations (or single annotation) to permute.
        statistic : str
            Node-level statistic to use in the test.
        num_perms : int = 1000
            Number of label permutation iterations to perform.
        seed : int = 94206
            Rng seed for the label permutations.
        """
        # TODO: Add pvalue correction
        ann_list = [ann_list, None] if isinstance(ann_list, str) else ann_list

        node_vals = self._node_statistics(statistic, np.ones(self.num_vertices()))

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
                    "num_perms": num_perms,
                    "statistic": statistic,
                }
            )

        return res_dicts
