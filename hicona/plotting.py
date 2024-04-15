"""Placehodler"""

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


__all__ = ["plot_alpha_grid"]

# TODO: there probably is a better way to import the style, maybe in init
plt.style.use("hicona/resources/paper_style.mplstyle")


def _empty_subplot(axes):
    """Create whitespace in specified plot axes."""

    axes.axis("off")


def _plot_output(plot, img_path, show):
    """Plotting function output behaviour. Return plot only if not shown."""

    plt.tight_layout()

    if img_path:
        plt.savefig(img_path)
        plt.close()
    if show:
        plt.show()
    else:
        return plot


def plot_alpha_grid(
    alpha_grid: pd.DataFrame,
    img_path: str | None = None,
    show: bool = False,
):
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
    axes = sns.lineplot(alpha_grid, x="nodes_f", y="edges_f")
    axes.set(
        title="Optimal Alpha Test Grid",
        xlabel="Node Fraction",
        ylabel="Edge Fraction",
        aspect="equal",
    )

    # Manually plot markers
    for _, pts in alpha_grid.iterrows():
        axes.plot(pts["nodes_f"], pts["edges_f"], **other_style)

    # Redraw optimal alpha marker to have artist on top
    optim = alpha_grid.iloc[alpha_grid["eu_dist"].idxmin()]  # type: ignore
    axes.plot(optim["nodes_f"], optim["edges_f"], **optim_style)
    plt.text(
        optim["nodes_f"] + x_offset,
        optim["edges_f"] + y_offset,
        rf"$\alpha$ = {optim['alpha']:.3f}",
    )

    return _plot_output(axes, img_path, show)


def plot_alpha_distr(alpha_distr, img_path: str | None = None, show: bool = False):
    """Plot alpha value distribution as a line plot."""

    axes = sns.lineplot(alpha_distr)
    axes.set(
        title="Alpha Score Distribution",
        xlabel="Alpha Score",
        ylabel="Count",
    )

    return _plot_output(axes, img_path, show)


def _get_color_map():
    """Create a red-blue divergent color map for the heatmaps."""

    cols = ["mediumblue", "blue", "white", "red", "firebrick"]
    vals = [0, 0.15, 0.5, 0.85, 1]
    cmap = LinearSegmentedColormap.from_list("rg", list(zip(vals, cols)))
    return cmap


def plot_dynamics_full(
    ann_dynamics: pd.DataFrame,
    alpha_distr: pd.DataFrame,
    sort_rows: bool = True,
    inf_to_nan: bool = True,
    img_path: str | None = None,
    show: bool = False,
):
    """Plot annotation dynamics matrix.

    Create a heatmap for the provided annotation dynamics matrix. Moreover,
    plot the distribution of alpha values as a line plot with overlaid lines
    for the thresholds used in the dynamics matrix.

    Parameters
    ----------
    ann_dynamics : pd.DataFrame
        The annotation dynamics matrix to plot.
    alpha_distr : pd.DataFrame
        The distribution of alpha values to plot.
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

        link = linkage(dataf.fillna(0), optimal_ordering=True)
        dendro = dendrogram(link, no_plot=True)
        return dendro["leaves"]

    tab = ann_dynamics.copy()

    # Set up the plot
    fig, axes = plt.subplots(
        2,
        2,
        height_ratios=[1, 3],
        width_ratios=[20, 1],
        figsize=(12, 9),
    )

    # TOP LEFT: Alpha distribution ------------------------------------------

    # Remove unwanted white space on the right of the plot
    tab.dropna(axis=1, how="all", inplace=True)
    thresholds = [float(t) for t in tab.columns]
    alpha_distr = alpha_distr[alpha_distr["value"] <= max(thresholds)]

    sns.lineplot(data=alpha_distr, x="value", y="count", ax=axes[0][0])
    axes[0][0].set(
        xlabel="Alpha",
        ylabel="Number of Pixels",
        xlim=(0, max(thresholds)),
        ylim=(-alpha_distr["count"].max() * 0.1, max(alpha_distr["count"]) * 1.1),
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

    tab.fillna(0, inplace=True)

    sns.heatmap(
        tab,
        ax=axes[1][0],
        cmap=_get_color_map(),
        center=0,
        cbar_ax=axes[1][1],
    )

    axes[1][0].set(xlabel=None, ylabel=None, xticklabels=[])
    axes[1][0].xaxis.set_tick_params(labelbottom=False)
    axes[1][0].tick_params(bottom=False)

    fig.tight_layout(pad=0)  # TODO: Check if this is necessary

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

    # title = rf"{interval[0]} $< \alpha \leq$ {interval[1]}"
    cmap = _get_color_map()

    axes = sns.heatmap(odds_mat, annot=anno_mat, cmap=cmap, center=0)
    axes.set(xlabel=None, ylabel=None)

    return _plot_output(axes, img_path, show)
