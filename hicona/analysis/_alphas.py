"""Placeholder"""

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from hicona._dtypes import AlphaModType, OptionalAxes
from hicona._numeric import rounding
from hicona.analysis import _plotting

if TYPE_CHECKING:
    from hicona._table import HiconaTable


__all__ = ["AlphaGrid"]


class AlphaGrid:
    """Class for the computation of the optimal alpha value for filtering.

    During class initialization, the optimal alpha value is computed using
    the following steps:

    - define a grid of alpha values centered around the current optimal
      and spaced in order to cover 1 unit of previous decimal position.
    - for each alpha value, filter the table and compute the remaining
      fraction of edges and nodes.
    - for each alpha compute the Euclidean distance from point ``(1, 0)``
      in the space ``x: fraction of nodes``, ``y: fraction of edges``.
    - set as new optimal alpha the one minimizing the Euclidean distance.
    - move to the next decimal position and repeat the procedure.

    Parameters
    ----------
    table : HiconaTable
        The table for which the optimal alpha value is computed.
    alpha_mod : "alpha_min" or "alpha_max"
        The alpha mode to use for filtering.
    decimals : int
        The number of decimal positions to consider when computing the grid.
    verbose : bool
        Whether to log the progress of the computation.
    """

    def __init__(
        self,
        table: "HiconaTable",
        alpha_mod: AlphaModType,
        decimals: int,
        verbose: bool,
    ):
        self._table = table
        self._alpha_mod = alpha_mod
        self._decimals = decimals
        self._alpha_grid = self._compute_grid(decimals, verbose)

    def _get_stats(self, thr: float) -> tuple[int, int]:
        """Return number of nodes and edges in a table filtered by alpha."""

        nodes: set = set()
        edges: int = 0

        for chunk in self._table.chunks():
            chunk = chunk.query(f"{self._alpha_mod} <= {thr}")

            nodes |= set(chunk["bin1_id"]) | set(chunk["bin2_id"])
            edges += len(chunk)

        return len(nodes), edges

    def _get_alpha_pts(self, thresholds: list[float]) -> pd.DataFrame:
        """Return dataframe with filtering statistics for a threshold grid."""
        # NOTE: Assumed that alpha thresholds are in decreasing order.

        res_list = []

        # Define normalization values
        tot_nodes, tot_edges = self._get_stats(1)

        # Iterate over each alpha value
        for alpha in thresholds:
            num_nodes, num_edges = self._get_stats(alpha)
            res_list.append(
                {
                    "alpha": alpha,
                    "nodes_n": num_nodes,
                    "edges_n": num_edges,
                    "nodes_f": (frac_nodes := num_nodes / tot_nodes),
                    "edges_f": (frac_edges := num_edges / tot_edges),
                    "eu_dist": math.dist((1, 0), (frac_nodes, frac_edges)),
                }
            )

        return pd.DataFrame(res_list)

    def _compute_grid(self, decimals: int, verbose: bool) -> pd.DataFrame:
        """Return the grid containing the statistics for each alpha value."""

        # Initialize optimal alpha and result container
        opt_alpha = 0.5  # Middle of initial search space 0-1
        num_points = 21  # Must be 10x + 1 to guarantee 1 unit span
        alpha_vals = []

        for pos in range(decimals):
            if verbose:
                print(f"Optimal alpha: computing decimal {pos+1}")

            # Define the grid of alpha values to test
            step = 0.5 * 10 ** (-pos)
            grid = np.linspace(opt_alpha + step, opt_alpha - step, num_points)
            grid = [a for a in grid if 0 <= a <= 1]

            # Compute the new statistics, then update optimal alpha
            new_pts = self._get_alpha_pts(grid)
            opt_alpha = self._get_minimal_dist(new_pts)
            alpha_vals.append(new_pts)

        return pd.concat(alpha_vals).reset_index()

    @staticmethod
    def _get_minimal_dist(table: pd.DataFrame) -> float:
        """Return the alpha value minimizing the Euclidean distance."""

        position = table["eu_dist"].idxmin()
        if not isinstance(position, int):
            raise ValueError("Non numeric index. This should not happen.")
        return table["alpha"].iloc[position]

    def optimal_alpha(self) -> float:
        """Return the optimal alpha value for filtering the pixel table."""

        optimal_value = self._get_minimal_dist(self._alpha_grid)
        return rounding.round_half_up(optimal_value, self._decimals)

    def plot(
        self,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the grid object.

        Plot and/or show the grid of alpha values tested when computing the
        optimal alpha value for pixel filtering. The plot has the fraction
        of retained nodes on the ``x`` axis and the fraction of retained
        edges on the ``y`` axis.

        Parameters
        ----------
        img_path : str or None, optional
            If provided, path to save the plot to. (default is None)
        show : bool, optional
            If True, display the plot in a :py:mod:`matplotlib` window.
            (default is False)

        Returns
        -------
        matplotlib.axes.Axes or None :
            If ``show`` if `False` and ``img_path`` is `None`, return plot
            axes. Otherwise, return None.
        """

        _plotting.plot_alpha_grid(self._alpha_grid, img_path, show)
