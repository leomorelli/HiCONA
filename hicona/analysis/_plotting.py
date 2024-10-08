"""Placehodler"""

import matplotlib.figure as fg
import matplotlib.axes as ax
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import coo_matrix


__all__ = ["plot_threshold_grid", "plot_jaccard_table"]


NAN_COLOR = "lightgrey"  # Color for NaN values in heatmaps
CM = 1 / 2.54  # cm to inches


def fine_grain_df(dataf):
    """Placeholder"""

    grain = max(
        int(str(c).split(".")[1]) if "." in str(c) else 0 for c in dataf.columns
    )
    fine_df = np.ndarray((len(dataf), 10 * grain), dtype=float)
    print(fine_df.shape)

    return fine_df


def _empty_subplot(axes):
    """Create whitespace in specified plot axes."""

    axes.axis("off")


def _plot_output(plot, img_path, show, padding=0):
    """Plotting function output behaviour. Return plot only if not shown."""

    plt.tight_layout(pad=padding)

    if img_path:
        plt.savefig(img_path)
        plt.close()
    if show:
        plt.show()
    else:
        return plot


def plot_threshold_grid(
    grid: pd.DataFrame,
    img_path: str | None = None,
    show: bool = False,
):
    """Plot the grid used to compute the optimal score threshold.

    Plot and/or show the grid of score thresholds tested when computing the
    optimal score threshold for pixel filtering. The plot has the fraction
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
    axes = sns.lineplot(grid, x="nodes_f", y="edges_f")
    axes.set(
        title="Optimal Score Threshold Test Grid",
        xlabel="Node Fraction",
        ylabel="Edge Fraction",
        aspect="equal",
    )

    # Manually plot markers
    for _, pts in grid.iterrows():
        axes.plot(pts["nodes_f"], pts["edges_f"], **other_style)

    # Redraw optimal optimal marker to have artist on top
    optim = grid.iloc[grid["eu_dist"].idxmin()]  # type: ignore
    axes.plot(optim["nodes_f"], optim["edges_f"], **optim_style)
    plt.text(
        optim["nodes_f"] + x_offset,
        optim["edges_f"] + y_offset,
        f"score = {optim['threshold']:.3f}",
    )

    return _plot_output(axes, img_path, show)


def plot_score_distr(score_distr, img_path: str | None = None, show: bool = False):
    """Plot score distribution as a line plot."""

    axes = sns.lineplot(score_distr)
    axes.set(
        title="Score Distribution",
        xlabel="Score",
        ylabel="Count",
    )

    return _plot_output(axes, img_path, show)


def _get_color_map():
    """Create a red-blue divergent color map for the heatmaps."""

    cols = ["mediumblue", "blue", "white", "red", "firebrick"]
    vals = [0, 0.15, 0.5, 0.85, 1]
    cmap = clr.LinearSegmentedColormap.from_list("rg", list(zip(vals, cols)))
    return cmap


def plot_dynamics_full(
    ann_dynamics: pd.DataFrame,
    score_distr: pd.DataFrame,
    sort_rows: bool = True,
    inf_to_nan: bool = True,
    img_path: str | None = None,
    show: bool = False,
):
    """Plot annotation dynamics matrix.

    Create a heatmap for the provided annotation dynamics matrix. Moreover,
    plot the score distribution as a line plot with overlaid lines
    for the thresholds used in the dynamics matrix.

    Parameters
    ----------
    ann_dynamics : pd.DataFrame
        The annotation dynamics matrix to plot.
    score_distr : pd.DataFrame
        The distribution of scores to plot.
    sort_rows : bool, optional
        Whether to sort the rows of the dynamics matrix by hierarchical
        clustering. Default is `True`.
    inf_to_nan : bool, optional
        Whether to replace infinite values with NaNs. This is to avoid messy
        behavior at the sided of the heatmap. Default is `True`.
    img_path : str, optional
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

    def get_row_order(dataf: pd.DataFrame):
        """Sort rows by linkage clustering for ease of visualization."""

        data = dataf.fillna(0)
        data = data.replace(np.inf, np.nanmax(data[data != np.inf]))
        data = data.replace(-np.inf, np.nanmin(data[data != -np.inf]))

        link = linkage(data, optimal_ordering=True)
        dendro = dendrogram(link, no_plot=True)
        return dendro["leaves"]

    tab = ann_dynamics.copy()

    # Set up the plot
    fig, axes = plt.subplots(
        2,
        2,
        height_ratios=[1, 3],
        width_ratios=[20, 1],
        figsize=(24 * CM, 18 * CM),
    )

    # TOP LEFT: Score distribution ------------------------------------------

    thresholds = [float(t) for t in tab.columns]
    score_distr = score_distr[score_distr["value"] <= max(thresholds)]

    sns.lineplot(data=score_distr, x="value", y="count", ax=axes[0][0])
    axes[0][0].set(
        xlabel="Score",
        ylabel="Number of Pixels",
        xlim=(-0.05, 1),
        ylim=(-score_distr["count"].max() * 0.1, max(score_distr["count"]) * 1.1),
    )
    axes[0][0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0][0].xaxis.set_label_coords(-0.05, -0.04)

    # Add vertical lines for the thresholds
    for t in thresholds:
        axes[0][0].axvline(t, color="red", linestyle="--", alpha=0.5)

    # TOP RIGHT: Empty subplot ----------------------------------------------
    _empty_subplot(axes[0][1])

    # BOTTOM: Annotation dynamics -------------------------------------------

    # Whether to perform hierarchical clustering on the rows
    if sort_rows:
        tab = tab.iloc[get_row_order(tab)]

    # To deal with infinite values breaking cbar
    if inf_to_nan:
        tab.replace([np.inf, -np.inf], np.nan, inplace=True)
    else:
        tab.replace(np.inf, np.nanmax(tab[tab != np.inf]), inplace=True)
        tab.replace(-np.inf, np.nanmin(tab[tab != -np.inf]), inplace=True)

    mask = np.isnan(tab)

    sns.heatmap(
        tab,
        ax=axes[1][0],
        cmap=_get_color_map(),
        center=0,
        cbar_ax=axes[1][1],
        mask=mask,
        vmax=2,  # TODO: change or implement
        vmin=-2,  # TODO: change or implement
    )

    axes[1][0].set(xlabel=None, ylabel=None, xticklabels=[])
    axes[1][0].xaxis.set_tick_params(labelbottom=False)
    axes[1][0].tick_params(bottom=False)
    axes[1][0].patch.set_color(NAN_COLOR)

    return _plot_output([fig, axes], img_path, show)


def plot_dynamics_interval(
    log_odds: pd.Series,
    p_values: pd.Series,
    img_path: str | None = None,
    show: bool = False,
):
    """Plot annotation dynamics for a restricted p-value interval.

    For a selected range of p-values, create a heatmap with intensity
    proportional to the log-odds ratio and annotations for statistical
    significance.

    Parameters
    ----------
    log_odds : pd.Series
        The log-odds ratios for the interval.
    p_values : pd.Series
        The p-values for the interval.
    img_path : str, optional
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

    odds_mat = log_odds.unstack().T
    anno_mat = p_values.unstack().T

    cmap = _get_color_map()

    axes = sns.heatmap(odds_mat, annot=anno_mat, cmap=cmap, center=0)
    axes.set(xlabel=None, ylabel=None)

    return _plot_output(axes, img_path, show)


def plot_comparison(
    distr_a: pl.DataFrame,
    distr_b: pl.DataFrame,
    points: pl.DataFrame,
    highlight: pl.DataFrame,
    names: tuple[str, str],
    img_path: str | None = None,
    show: bool = False,
):
    """Placeholder"""

    # CAP_VAL = max(10_000, max(norm_score_distr["count"]))

    main_ratio = 5  # Ratio of the main plot to the side plots
    sides_size = 8  # Size of the image side in inches
    distr_lims = (-0.01, 1.01)  # Limits for the distribution, assumed score
    label_size = 16  # Font size for the labels

    # Need to define it on the distribution somehow
    # cap_val = max(distr_a["count"].max(), distr_b["count"].max())
    cap_val = 100000

    # Set up the plot structure
    _, axes = plt.subplots(
        2,
        2,
        height_ratios=[main_ratio, 1],
        width_ratios=[1, main_ratio],
        figsize=(sides_size, sides_size),
    )

    # Normalized score distr subplot
    sns.lineplot(
        data=distr_b,
        x="count",
        y="value",
        errorbar=None,
        orient="y",
        ax=axes[0][0],
    )

    axes[0, 0].set(
        xlim=(cap_val * 1.05, -cap_val * 0.05),
        ylim=distr_lims,
        xlabel=None,
        xticklabels=[],
        xticks=[],
        # xscale="log",
    )
    # axes[0, 0].set_xscale("log")
    axes[0, 0].set_ylabel(names[1], fontsize=label_size)
    size_a = int(distr_a["count"].sum())
    axes[0, 0].text(cap_val, 0.95, f"N = {size_a}", fontsize=11)

    # Score vs score matrix
    sns.histplot(
        data=points,
        x="x",
        y="y",
        # aspect=1,
        # cbar=True,
        pmax=0.5,
        bins=200,
        ax=axes[0][1],
    )

    axes[0, 1].set(
        xlim=distr_lims,
        ylim=distr_lims,
        xlabel=None,
        ylabel=None,
        xticklabels=[],
        yticklabels=[],
        xticks=[],
        yticks=[],
    )
    axes[0, 1].text(0.80, 0.95, f"N = {len(points)}", fontsize=11)
    # axes[0, 1].spines[["right", "bottom", "left", "top"]].set_visible(False)
    # axes[0, 1].plot(highlight["x"], highlight["y"], "ro", markersize=0.1)

    # Empty lower left subplot
    axes[1, 0].axis("off")

    # Non normalized score distr subplot
    sns.lineplot(
        data=distr_a,
        x="value",
        y="count",
        errorbar=None,
        orient="x",
        ax=axes[1][1],
    )
    axes[1, 1].set(
        xlim=distr_lims,
        ylim=(cap_val * 1.05, -cap_val * 0.05),
        ylabel=None,
        yticklabels=[],
        yticks=[],
    )
    axes[1, 1].set_xlabel(names[0], fontsize=label_size)
    size_b = int(distr_b["count"].sum())
    axes[1, 1].text(0.80, cap_val, f"N = {size_b}", fontsize=11)

    return _plot_output(axes, img_path, show)


def plot_table_heatmap(
    table: pd.DataFrame,
    intensity_col: str,
    img_path: str | None = None,
    show: bool = False,
    **kwargs,
):
    """Placeholder."""

    _, axes = plt.subplots(1, 2, width_ratios=[1, 0.05])

    table = table.copy()
    table_min = min(min(table["bin1_id"]), min(table["bin2_id"]))
    table["bin1_id"] -= table_min
    table["bin2_id"] -= table_min
    table_max = max(max(table["bin1_id"]), max(table["bin2_id"]))
    print(table_min)

    print(table_max)

    data = coo_matrix(
        (table[intensity_col], (table["bin1_id"], table["bin2_id"])),
        (table_max + 1, table_max + 1),
    ).toarray()
    # table = table.pivot(index="bin1_id", columns="bin2_id", values=intensity_col)
    print(data)
    print(data.shape)

    print(axes)

    sns.heatmap(data, ax=axes[0], cbar_ax=axes[1], square=True)

    return _plot_output(axes, img_path, show)


def _edgelist_to_numpy(
    table: pl.DataFrame,
    value_col: str,
    out_side: int,
) -> np.ndarray:
    """Convert and edgelist to a dense numpy matrix."""

    # Create the empty dense matrix and fill it with the edge values
    edges: np.ndarray = table.select("bin1_id", "bin2_id", value_col).to_numpy()
    dense: np.ndarray = np.zeros((out_side, out_side), dtype=float)
    dense[edges[:, 0], edges[:, 1]] = edges[:, 2]

    return dense


def plot_table_comparison(
    table_a: pl.DataFrame,
    table_b: pl.DataFrame,
    values_col: str,
    *,
    log_scale: bool = False,
    binary: bool = False,
    gain_a: float = 1,
    gain_b: float = 1,
    **kwargs,
) -> tuple[fg.Figure, np.ndarray[ax.Axes]]:  # type: ignore
    """Given two edgelists and a values column, plot the comparison heatmap.

    Given two edge lists defined roughly over the same set of bins (reindexed
    starting from 0), plot the comparison heatmap of the two values columns.
    The final heatmap has table a as the upper triangular part, and table b as
    the lower triangular part. The diagonal is set to zero to avoid overlap.

    Gain is a multiplicative factor to apply to the values of the matrices
    (prior to eventual log scaling) in case of vast differences in the values.

    Kwargs are passed to the seaborn heatmap function.
    """

    if binary and log_scale:
        raise ValueError("Cannot use binary and log scale at the same time.")

    # Get the dense matrices from the edge lists and apply the gains
    side = max(
        int(table_a.select("bin2_id").to_series().max()),  # type: ignore
        int(table_b.select("bin2_id").to_series().max()),  # type: ignore
    )
    side += 1

    matrix_a = _edgelist_to_numpy(table_a, values_col, side) * gain_a
    matrix_b = _edgelist_to_numpy(table_b, values_col, side) * gain_b

    # Add the matrices and set the diagonal to zero to avoid overlap
    full_matrix = matrix_a + matrix_b.T
    np.fill_diagonal(full_matrix, 0)

    if log_scale:
        full_matrix = np.log1p(full_matrix)
    if binary:
        full_matrix = full_matrix > 0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        height_ratios=[16, 1],
        figsize=(15 * CM, 18 * CM),
    )

    sns.heatmap(
        full_matrix,
        square=True,
        cmap="viridis",
        cbar_kws={
            "label": f"ln(1 + {values_col})" if log_scale else values_col,
            "orientation": "horizontal",
        },
        **kwargs,
        ax=axes[0],
        cbar_ax=axes[1],
    )
    axes[0].axis("off")

    # For some reason double tight layout is needed to fully compress the plot
    plt.tight_layout()
    plt.tight_layout()

    return fig, axes


def plot_jaccard_table(
    jaccard_table: pl.DataFrame,
    title: str | None = None,
) -> tuple[fg.Figure, np.ndarray[ax.Axes]]:  # type: ignore
    """Placeholder."""

    def custom_heatmap(
        matrix: np.ndarray,
        *,
        axes: ax.Axes,
        cbar_ax: ax.Axes,
        cbar_min: float,
        cbar_max: float,
        cmap: str,
        ylabel: str,
        **kwargs,
    ):
        """Placeholder."""

        sns.heatmap(
            matrix,
            ax=axes,
            cbar_ax=cbar_ax,
            annot=True,
            fmt=".3f",
            square=True,
            cbar_kws={"ticks": [cbar_min, (cbar_min + cbar_max) / 2, cbar_max]},
            cmap=cmap,
            vmin=cbar_min,
            vmax=cbar_max,
            **kwargs,
        )
        axes.set_yticklabels(axes.get_yticklabels(), rotation=0)
        axes.set_xticklabels(axes.get_xticklabels(), rotation=45)
        axes.vlines(1, 0, matrix.shape[0], colors="w", linestyles="-")
        axes.set_ylabel(ylabel, fontsize=14)

    table = jaccard_table.with_columns(
        pl.concat_str("Table A", "Table B", separator="-").alias("Pair")
    ).pivot("Flow", index="Pair", values="Jaccard")

    matrix: np.ndarray = table.drop("Pair").to_numpy()
    row_labels: list[str] = table["Pair"].to_list()
    col_labels: list[str] = table.drop("Pair").columns

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, matrix.shape[0])

    # TODO: Definitely to adjust
    dim_tolerance = 8 * CM
    cbar_width = 1 * CM
    plt_width = matrix.shape[1] * CM + dim_tolerance
    plt_height = matrix.shape[0] * CM * 3 + dim_tolerance
    width_ratios = [plt_width - cbar_width, cbar_width]

    fig, axes = plt.subplots(
        3, 2, figsize=(plt_width, plt_height), width_ratios=width_ratios
    )

    custom_heatmap(
        matrix,
        axes=axes[0, 0],
        cbar_ax=axes[0, 1],
        cbar_min=0,
        cbar_max=1,
        ylabel=r"$\mathbb{J}_x$",
        cmap="Reds",
        yticklabels=row_labels,
        xticklabels=False,
    )

    custom_heatmap(
        np.log2(matrix / matrix[:, :1]),
        axes=axes[1, 0],
        cbar_ax=axes[1, 1],
        cbar_min=-3,
        cbar_max=3,
        ylabel=r"$\mathrm{log}_2(\mathbb{J}_x)$",
        cmap="coolwarm",
        yticklabels=row_labels,
        xticklabels=False,
    )

    custom_heatmap(
        matrix - matrix[:, :1],
        axes=axes[2, 0],
        cbar_ax=axes[2, 1],
        cbar_min=-1,
        cbar_max=1,
        ylabel=r"$\mathbb{J}_x - \mathbb{J}_0$",
        cmap="bwr",
        yticklabels=row_labels,
        xticklabels=col_labels,
    )

    if title:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig, axes
