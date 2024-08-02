"""Placeholder"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import scipy as sp

from hicona._dtypes import AlphaModType, OptionalAxes
from hicona._ops import chunked, dataf
from hicona.analysis import _plotting

if TYPE_CHECKING:
    from hicona._table import HiconaTable


__all__ = ["AnnotDynamics"]


class AnnotDynamics:
    """Class for the compute and analyze pixel table annotation dynamics.

    Class used to compute, return and plot pixel table annotation dynamics,
    that is, the changes in frequency of pixel annotation pairs as a function
    of the alpha value used for filtering the table.

    The annotation dynamics algorithm works as follows:

    - compute the background fractions, that is, the fraction of pixels for
      each annotation pair in the entire unfiltered table.
    - for each interval, compute the interval fractions, which are computed
      analogously to the background ones but only considering pixels with an
      alpha value falling in the interval.
    - for each interval, compute the log odds ratios of the interval fractions
      with respect to the background fractions. Also compute p-values for the
      log odds ratios using Fisher's exact test and correct them for multiple
      testing using the Benjamini-Hochberg method.

    Objects of this class can also be initialized through the
    ``HiconaTable.get_annot_dynamics()`` method.

    Parameters
    ----------
    table : HiconaTable
        The table for which the annotation dynamics are computed.
    annot_name : str
        The name of the bin annotation column to consider.
    as_quantiles : bool, optional
        Whether the intervals are expressed in quantile points, rather than
        in absolute percentage points. Default is 'True'.
    alpha_mod : 'alpha_min' or 'alpha_max', optional
        The alpha mode to use for filtering. Default is 'alpha_min'.

    See Also
    --------
    hicona.HiconaTable.get_annot_dynamics:
        Method to create an instance of the class starting from a table object.

    Examples
    --------
    Create an instance of the class starting from a table object:

    >>> import hicona
    >>> handle = hicona.HiconaCooler("path/to/cool_file.cool")
    >>> table = handle.fetch_table("hicona")
    >>> dynamics = table.get_annot_dynamics("chromatin_state")

    Create an instance of the class using the constructor:

    >>> dynamics = hicona.AnnotDynamics(table, "chromatin_state")

    """

    def __init__(
        self,
        table: "HiconaTable",
        annot_name: str,
        as_quantiles: bool = True,
        alpha_mod: AlphaModType = "alpha_min",
    ):

        self._hic_table = table
        self._anno_name = annot_name
        self._alpha_mod = alpha_mod
        self._as_quants = as_quantiles

        # Automatically generate a grid with 0.01 wide intervals
        # Which can be aggregated to the desired intervals when queried
        self._alpha_distr = self._hic_table.get_distribution(alpha_mod)
        self._break_pts = self._compute_breakpoints(0.01, self._as_quants)
        self._abs_dynam = self._compute_dynamics()

    def _compute_breakpoints(self, size: float, as_quants: bool) -> list[float]:
        """Get the break points for the intervals."""

        # Define the discrete alpha break points
        num_pt: int = int(1 / size)
        decimals: int = len(str(size).split(".")[1])
        points: list[float] = [round(size * i, decimals) for i in range(1, num_pt + 1)]

        # Convert break points to quantiles if needed
        if as_quants:
            chunks = self._hic_table.chunks()
            quants = chunked.chunked_quants(chunks, self._alpha_mod, points)
            points = quants[self._alpha_mod].tolist()

        return points

    def _compute_dynamics(self) -> pd.DataFrame:
        """Compute absolute frequencies of annotation pairs per interval."""

        # Set up variables
        anno_cols = [f"{self._anno_name}1", f"{self._anno_name}2"]

        # NOTE: dedup is needed since quantiles can yield duplicate points
        points = list(np.unique(self._break_pts))
        points.reverse()

        # Retrieve annotation dynamics in chunks
        dynam_parts = []
        chunk_cols = anno_cols + [self._alpha_mod]
        for chunk in self._hic_table.chunks(annotated=True, columns=chunk_cols):

            # Have annotations alpahebetically sorted to avoid duplicates
            dataf.swap_columns(chunk, *anno_cols)

            # Compute cumulative absolute frequencies per interval per chunk
            for pt in points:
                chunk.query(f"{self._alpha_mod} <= {pt}", inplace=True)
                chunk_parts = chunk.groupby(anno_cols, as_index=False).count()
                chunk_parts["upper"] = pt
                dynam_parts.append(chunk_parts)

        # Create full dynamics table
        dynam = pd.concat(dynam_parts, ignore_index=True)
        dynam = dynam.groupby(anno_cols + ["upper"], as_index=False).sum()

        # Convert to table with annotations as rows, upper bounds as columns
        dynam = dynam.pivot_table(
            index=anno_cols,
            columns="upper",
            values=self._alpha_mod,
            fill_value=0,
        )

        # Convert to matrix to fix missing columns and such
        matrix = np.zeros([len(dynam), len(self._break_pts)], dtype=int)
        for ind, val in enumerate(self._break_pts):
            matrix[:, ind] = dynam.get(val, 0)

        # Using ordering names since duplicate columns break many ops
        col_names = [i + 1 for i in range(len(self._break_pts))]
        return pd.DataFrame(matrix, index=dynam.index, columns=col_names)

    def get_dynamics(
        self,
        cumulative: bool = False,
        intervals: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return annotation dynamics and relative p-values as dataframes.

        Log odds ratios and p-values are computed as indicated in the class
        documentation. The results are returned as two dataframes, one for the
        log odds ratios and one for the p-values.

        .. warning::
            Currently, pixels with alpha value equal to zero are lost since the
            lower bound is excluded. This will likely be changed in the future
            even though the number of pixels with alpha equal to zero is usually
            very low.

        Parameters
        ----------
        cumulative : bool, optional
            Whether to compute the dynamics cumulatively. If set to 'True',
            each interval includes all pixels with alpha values up to the
            interval upper bound (included), otherwise it considers only
            pixels whose alphas are between the previous interval upper bound
            (excluded) and the current one (included). Default is 'False'.
        intervals : list of int or None, optional
            If provided, the upper bounds of the intervals to consider, else
            a 1 unit spaced grid is used. Interval boundaries should be in the
            range 0 < i <= 100. Default is 'None'.

        Returns
        -------
        log_odds : pandas.DataFrame
            The log odds ratios of the annotation dynamics.
        p_values : pandas.DataFrame
            The p-values of the annotation dynamics.

        Examples
        --------

        Create a dynamics object with the quantile option set to 'True':

        >>> import hicona
        >>> handle = hicona.HiconaCooler("path/to/cool_file.cool")
        >>> table = handle.fetch_table("hicona")
        >>> dynamics = table.get_annot_dynamics("HMM_annot", as_quantiles=True)

        Compute the cumulative dynamics for that dynamics object:

        >>> log_odds, p_values = dynamics.get_dynamics(cumulative=True)
        >>> log_odds
                                 0.1275    0.1597  ...    0.5090  0.6092
        HMM_annot1 HMM_annot2
        Enh        Enh         0.822894  0.701970  ...  0.794474     NaN
                   Het        -0.864572 -0.844857  ...  0.157273     NaN
                   Prom        0.560924  0.435358  ...  0.741693     NaN
        ...        ...              ...       ...  ...       ...     ...
        Tx         Tx          0.112815  0.167098  ...  0.524686     NaN
                   Void       -0.500922 -0.504499  ...  0.147691     NaN
        Void       Void        0.484172  0.508217  ... -0.680444     NaN
        <BLANKLINE>
        [21 rows x 100 columns]

        In this case, no intervals are provided, so the default grid of 1 unit
        (in this case 1 percentile since the quantile option is set to 'True')
        is used. Since the cumulative option is set to 'True', all pixels with
        alpha values up to the interval upper bound are included, meaning that
        the first column considers pixels with alpha values in the range
        ``(0, 0.1275]``, the second column in the range ``(0, 0.1597]``, and so on.

        Compute the non-cumulative dynamics for that dynamics object:

        >>> bounds = [10 * i for i in range(1, 11)]
        >>> log_odds, _ = dynamics.get_dynamics(cumulative=False, intervals=bounds)
        >>> log_odds
                                     10        20  ...         90       100
        HMM_annot1 HMM_annot2
        Enh        Enh         0.450639  0.119999  ...   0.033375 -0.376168
                   Het        -0.687511 -0.280919  ...   0.496997  0.516708
                   Prom        0.294477  0.069265  ...  -0.014289 -0.350122
        ...        ...              ...       ...  ...        ...       ...
        Tx         Tx          0.337442 -0.093102  ...  -0.956154 -0.805076
                   Void       -0.364742 -0.404569  ...  -0.172750 -0.174324
        Void       Void        0.270512  0.375850  ...   0.019111  0.400275
        <BLANKLINE>
        [21 rows x 10 columns]

        Since intervals are provided, the dynamics are computed every ten
        percentile points. Moreover, since the cumulative option is set to
        'False', the first column considers pixels with alpha values in the
        range ``(0, 10]``, the second column in the range ``(10, 20]``, and so on.

        """

        # TODO: Maybe change colnames when using quantile? Or at least making
        # the use consistent between giving and not giving intervals

        dynam: pd.DataFrame = self._abs_dynam.copy()
        print(dynam)

        # Restrict the table to the intervals of interest if provided
        if intervals is not None:

            if not all(0 < i <= 100 for i in intervals):
                raise ValueError("Bounds must be in 0 < i <= 100")
            if len(set(intervals)) != len(intervals):
                raise ValueError("Upper bounds must be unique")

            dynam = dynam[sorted(intervals)]

        # Convert to non cumulative if needed
        if not cumulative:
            first_col = dynam.columns[0]
            dynam = dynam.diff(axis=1)
            dynam[first_col] = self._abs_dynam[first_col]

        # Compute log odds ratios and p-values matrices
        ref_col = self._abs_dynam[self._abs_dynam.columns[-1]]

        res = [dataf.odds_ratios(dynam[c], ref_col) for c in dynam.columns]
        axes = {"columns": dynam.columns, "index": dynam.index}
        log_odds = pd.DataFrame(np.array([r[0] for r in res]).T, **axes)
        p_values = pd.DataFrame(np.array([r[1] for r in res]).T, **axes)

        # Apply FDR correction to p-values
        for _, row in p_values.iterrows():
            valid = (row >= 0) * (row <= 1)
            adj_p_values = sp.stats.false_discovery_control(row[valid])
            row[valid] = adj_p_values

        # Convert back to actual col names with duplicates
        log_odds.columns = intervals or self._break_pts  # type: ignore
        p_values.columns = intervals or self._break_pts  # type: ignore

        return log_odds, p_values

    def plot_full(
        self,
        cumulative: bool = True,
        intervals: list[int] | None = None,
        sort_rows: bool = True,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the full annotation dynamics for a pixel table.

        Plot the full annotation dynamics table as a heatmap with all the
        annotation pairs as rows and the thresholds as columns. On top of
        the heatmap, represent the alpha distribution and the thresholds
        of the heatmap cells (dashed lines).

        Parameters
        ----------
        cumulative : bool, optional
            Whether to compute the cumulative dynamics. Default is 'True'.
        intervals : list of int or None, optional
            If provided, the upper bounds of the intervals to consider, else
            a 1 unit spaced grid is used. Interval boundaries should be in the
            range 0 < i <= 100. Default is 'None'.
        sort_rows : bool, optional
            Whether to sort the rows of the heatmap by linkage. This groups
            similar annotations together, though it might hamper the
            comparison of multiple tables. Default is 'True'.
        img_path : str or None, optional
            If provided, save the plot to the specified path. Default is 'None'.
        show : bool, optional
            If 'True', display the plot in a matplotlib window. Default is 'False'.

        Returns
        -------
        matplotlib.axes.Axes or None
            If ``show`` if `False` and ``img_path`` is 'None', return plot
            axes. Otherwise, return 'None'.

        Examples
        --------

        .. image:: ../../_static/dynamics_cumulative.png
        """

        odds, _ = self.get_dynamics(cumulative, intervals)
        _plotting.plot_dynamics_full(
            odds,
            self._alpha_distr,
            sort_rows=sort_rows,
            inf_to_nan=True,
            img_path=img_path,
            show=show,
        )

    def plot_interval(
        self,
        lower_bound: int,
        upper_bound: int,
        img_path: str | None = None,
        show: bool = False,
    ) -> OptionalAxes:
        """Plot the annotation dynamics for a restricted p-value interval.

        Given a single alpha value interval (lower bound excluded, upper bound
        included), create a lower-triangular heatmap for the annotation
        enrichment in that interval.

        Parameters
        ----------
        lower_bound : int
            The lower bound of the alpha interval to plot. See `get_dynamics`
            for more information.
        upper_bound : int
            The upper bound of the alpha interval to plot. See `get_dynamics`
            for more information.
        img_path : str or None, optional
            If provided, path to save the plot to. (default is None)
        show : bool, optional
            If True, display the plot in a :py:mod:`matplotlib` window.
            (default is False)

        Returns
        -------
        matplotlib.axes.Axes or None
            If ``show`` if `False` and ``img_path`` is `None`, return plot
            axes. Otherwise, return None.
        """

        interv = [lower_bound, upper_bound]
        odds, pvals = self.get_dynamics(intervals=interv, cumulative=False)

        _plotting.plot_dynamics_interval(
            odds[upper_bound],
            pvals[upper_bound],
            img_path,
            show,
        )
