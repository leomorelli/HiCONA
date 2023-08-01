"""Placeholder

Placeholder
"""

from collections import deque
from math import dist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import round_half_up


class ChromTable:
    """Class used to handle preprocessed chromosome level table.

    Class implementing methods for further manipulation of already
    preprocessed chromosome level tables. It allows to prepare the tables
    for network conversion (for instance by filtering pixels based on
    sparsification results). Class mostly meant for instantiation through
    a :py:class:`ChromTablesIterator` instance.

    Parameters
    ----------
    pix_table : :py:class:`DataFrame`
        Pixels-like table (e.i. "bin1_id", "bin2_id", "count" columns must
        be present).
    params_info : dict
        Dictionary containing information about table preprocessing.
    """

    def __init__(self, pix_table: pd.DataFrame, params_info: dict):
        self._data = pix_table
        self._params_info = params_info
        self._optim_alpha = None
        self._alphas_grid = None

    @property
    def data(self) -> pd.DataFrame:
        """:py:class:`DataFrame` of pixels."""
        return self._data

    @property
    def params_info(self) -> dict:
        """Dictionary of parameters used during preprocessing."""
        return self._params_info

    @property
    def optim_alpha(self) -> float | None:
        """Optimal alpha to filter the pixels (if computed, else None)."""
        return self._optim_alpha

    def _get_alpha_pts(self, thresholds):
        """Return dataframe with filtering statistics for a threshold grid."""
        # NOTE: Assumed that alpha thresholds are in decreasing order.

        def num_nodes(table):
            """Count unique nodes in pixels-like table."""
            return len(set(table["bin1_id"]) | set(table["bin2_id"]))

        # Initialize input parameters and output container
        alpha_thr = deque(thresholds)
        cur_table = self._data.copy()
        res_list = []

        # Define normalization values
        tot_nodes = num_nodes(cur_table)
        tot_edges = cur_table.size

        # Iterate over each alpha value
        while alpha_thr:
            alpha = alpha_thr.popleft()

            # Filter and compute filtering statistics
            cur_table = cur_table[cur_table["spar_alpha"] <= alpha]
            cur_nodes = num_nodes(cur_table) / tot_nodes
            cur_edges = cur_table.size / tot_edges

            # Store results as matrix row
            res_list.append(
                {
                    "alpha": alpha,
                    "nodes_f": cur_nodes,
                    "edges_f": cur_edges,
                    "eu_dist": dist((1, 0), (cur_nodes, cur_edges)),
                }
            )

        return pd.DataFrame(res_list)

    def compute_opt_alpha(
        self,
        decimals: int = 3,
        num_pts: int = 11,
        verbose: bool = True,
    ) -> float:
        """Return the optimal alpha value for filtering the pixel table.

        Compute the optimal alpha value for by iterating these steps:

        - define ``num_pts`` alpha values centered around the current optimal
          and spaced in order to cover 1 unit of previous decimal position.
        - for each alpha value, filter the table and compute the remaining
          fraction of edges and nodes.
        - for each alpha compute the Euclidean distance from point ``(1, 0)``
          in the space ``x: fraction of nodes``, ``y: fraction of edges``.
        - set as new optimal alpha the one minimizing the Euclidean distance.
        - move to the next decimal position and repeat the procedure.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal positions to compute for the alpha value. Must
            be at least 1. (default is 3)
        num_pts : int, optional
            Number of alpha values to test at each decimal position. Must be
            at least 3. (default is 11)
        verbose : bool, optional
            Print progress to console. (default is True)

        Returns
        -------
        float :
            Optimal alpha value
        """

        # TODO: If num_pts is removed, add check of pre-existing alphas
        # TODO: Maybe add recompute parameter but not the most elegant
        # TODO: Could add check that minimum is not grid extrema

        # Initialize optimal alpha and result container
        opt_alpha = 0.5  # Middle of initial search space 0-1
        alpha_vals = []

        for pos in range(decimals):
            if verbose:
                print(f"Optimal alpha: computing decimal {pos+1}")

            # Define the grid of alpha values to test
            step = 0.5 * 10 ** (-pos)
            grid = np.linspace(opt_alpha + step, opt_alpha - step, num_pts)
            grid = [a for a in grid if 0 <= a <= 1]

            # Compute the new statistics, then update optimal alpha
            new_pts = self._get_alpha_pts(grid)
            opt_alpha = new_pts["alpha"].iloc[new_pts["eu_dist"].idxmin()]
            alpha_vals.append(new_pts)

        # Store results
        alpha_vals = pd.concat(alpha_vals).reset_index()
        opt_alpha = round_half_up(opt_alpha, decimals)
        self._alphas_grid = alpha_vals
        self._optim_alpha = opt_alpha

        if verbose:
            print(f"Optimal alpha: {opt_alpha}")

        return opt_alpha

    def plot_alpha_selec(self, img_path: str = None, show: bool = False):
        """Plot the grid used to compute the optimal alpha value.

        Plot and/or show the grid of alpha values tested when computing the
        optimal alpha value for pixel filtering. The plot has the fraction
        of retained nodes on the ``x`` axis and the fraction of retained
        edges on the ``y`` axis.

        Parameters
        ---------
        img_path : str, optional
            If provided, path to save the plot to. (default is None)
        show : bool, optional
            If True, display the plot in a :py:mod:`matplotlib` window.
            (default is False)
        """

        # Define plotting parameters
        # TODO: Find definition that does not break with rescaling
        x_offset = -0.23
        y_offset = +0.03
        optim_style = {
            "marker": "o",
            "markersize": 10,
            "markerfacecolor": "tab:orange",
            "markeredgewidth": 0.0,
        }
        other_style = {
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "tab:blue",
            "markeredgewidth": 0.0,
        }

        # Create main plot
        axes = sns.lineplot(self._alphas_grid, x="nodes_f", y="edges_f")
        axes.set(
            title="Optimal Alpha Test Grid",
            xlabel="Node Fraction",
            ylabel="Edge Fraction",
            aspect="equal",
        )

        # Manually plot markers
        for _, pts in self._alphas_grid.iterrows():
            axes.plot(pts["nodes_f"], pts["edges_f"], **other_style)

        # Redraw optimal alpha marker to have artist on top
        optim = self._alphas_grid.iloc[self._alphas_grid["eu_dist"].idxmin()]
        axes.plot(optim["nodes_f"], optim["edges_f"], **optim_style)
        plt.text(
            optim["nodes_f"] + x_offset,
            optim["edges_f"] + y_offset,
            rf"$\alpha$ = {optim['alpha']:.3f}",
        )

        if img_path:
            plt.savefig(img_path)

        if show:
            plt.show()

    def filter_alpha(self, alpha: str | float = "optimal") -> pd.DataFrame:
        """Return the pixel table filtered according to some alpha value.

        Return a :py:class:`DataFrame` where only the pixels having a
        sparsification alpha value below the provided threshold are kept.

        Parameters
        ----------
        alpha : str or float, optional
            Keeping only the pixels with alpha smaller than this threshold.
            If "optimal", use the previously computed optimal alpha value.
            (default is "optimal")

        Returns
        -------
        :py:class:`DataFrame` :
            Dataframe of filtered pixels.
        """

        if alpha == "optimal":
            if not self._optim_alpha:
                msg = "Cannot filter by optimal alpha before computing it."
                raise UnboundLocalError(msg)
            alpha = self._optim_alpha

        filt_df = self._data.copy()
        filt_df = filt_df[filt_df["spar_alpha"] < alpha]

        return filt_df
