"""Placeholder
Placeholder
"""

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
import graph_tool.all as gt
from matplotlib import pyplot as plt

from utils import pd_to_gt_dtype, console_log

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

    def __init__(self, dataf: pd.DataFrame, to_keep: bool | Iterable[str] = None):
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

        # Initialize the object
        eprops = [(p, pd_to_gt_dtype(dataf[p].dtype)) for p in to_keep]
        super().__init__(dataf.values, eprops=eprops)
        vprop = self.new_vertex_property("int", vids)
        self.vp["bin_id"] = vprop
        # TODO: Make vprops iterable to annotate with chrom region?

    def _node_statistics(self, stat, mask):
        """Placeholder
        Placeholder
        """
        # TODO: Look for an alternative to the big and ugly switch case

        if stat == "ave_degree":
            vlist = mask.nonzero()[0]
            stats_list = self.get_out_degrees(vlist)
        else:
            raise ValueError(f"{stat} is not among the supported statistics.")

        return stats_list

    def _perm_diff(self, stats_list, attr_ohe, rng):
        """Compute average difference of the statistic for a single permutation."""
        # TODO: Remove self or redefine in inner scope?
        perm_ohe = attr_ohe[:, rng.permutation(attr_ohe.shape[1])]
        diff = -np.diff((perm_ohe * stats_list).sum(axis=1) / perm_ohe.sum(axis=1))[0]
        return diff

    def _attr_permutation(self, attr1, attr2, statistic, num_trials, seed):
        """Placeholder
        Placeholder
        """

        # Retrieve indices of the two annotations (each column is a node)
        arr1 = self.vp[attr1].get_array()
        arr2 = self.vp[attr2].get_array() if attr2 else np.ones(self.num_vertices())
        ohe = np.vstack((arr1, arr2))
        del arr1, arr2

        # Compute the statistic only for the nodes of interest
        perm_mask = ohe.sum(axis=0) != 0  # Node has at least one annotation
        stat_vals = self._node_statistics(statistic, perm_mask)
        ohe = ohe[:, perm_mask]

        # Compute average difference of the statistic for the original annotation...
        # ... then for all permutation and compute p-value
        rng = np.random.default_rng(seed)
        original = -np.diff((ohe * stat_vals).sum(axis=1) / ohe.sum(axis=1))[0]
        permuted = [self._perm_diff(stat_vals, ohe, rng) for _ in range(num_trials)]
        p_val = (original > permuted).sum() / num_trials

        """
        # Plot statistic difference distribution
        plt_lim = max(abs(min(permuted)), abs(max(permuted)))
        plt.hist(permuted, range=(-plt_lim, plt_lim), bins=20)
        plt.axvline(original, color="#eb7a34")
        plt.show()
        """

        return p_val

    @console_log
    def permute_attributes(
        self,
        attributes: str | Iterable[str],
        statistic: str,
        num_trials: int = 1000,
        pval_correction: str = "Bonferroni",
        seed: int = 42069,
    ):
        """Placeholder
        Placeholder
        """
        # TODO: check that the attributes are in OHE

        # If the attribute is a string, convert it to list with one element
        attributes = [attributes] if isinstance(attributes, str) else attributes
        attributes = [*attributes, None] if len(attributes) == 1 else attributes

        print("#" * 40)
        print(f"Working on pair {attributes}")

        pvals = []
        for comb in combinations(attributes, 2):
            pvals.append(self._attr_permutation(*comb, statistic, num_trials, seed))

        return pvals


if __name__ == "__main__":
    import random
    import time

    random.seed(42069)

    FILE_PATH = "../test_files/HUVEC_chr2.csv"
    df = pd.read_csv(FILE_PATH)
    df = df[df["spar_alpha"] < 0.15]
    print(df.shape)

    network = HiconaGraph(df, ["count", "exp_ratio"])

    deg1 = [1 if v.out_degree() == 1 else 0 for v in network.vertices()]
    mock1 = network.new_vertex_property("bool", deg1)
    network.vp["deg1"] = mock1

    deg2plus = [1 if v.out_degree() > 1 else 0 for v in network.vertices()]
    mock2 = network.new_vertex_property("bool", deg2plus)
    network.vp["deg2+"] = mock2

    num_nodes = network.num_vertices()
    rand_vals = random.sample(range(num_nodes), round(num_nodes * 0.2))
    indexer = [1 if i in rand_vals else 0 for i in range(num_nodes)]
    mock3 = network.new_vertex_property("bool", indexer)
    network.vp["rand"] = mock3

    network.list_properties()

    print(network.permute_attributes("rand", "ave_degree", 1_000))
    print(network.permute_attributes(["deg1"], "ave_degree", 1_000_000))
    print(network.permute_attributes(["deg1", "deg2+"], "ave_degree", 1_000_000))
