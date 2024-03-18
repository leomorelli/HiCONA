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

    if img_path:
        plt.savefig(img_path)

    if show:
        plt.show()
    else:
        return plot


def _multiaxes_heatmap(data, data_ax, cbar_ax):
    pass


def plot_alpha_grid(alpha_grid, img_path: str | None = None, show: bool = False):
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
    optim = alpha_grid.iloc[alpha_grid["eu_dist"].idxmin()]
    axes.plot(optim["nodes_f"], optim["edges_f"], **optim_style)
    plt.text(
        optim["nodes_f"] + x_offset,
        optim["edges_f"] + y_offset,
        rf"$\alpha$ = {optim['alpha']:.3f}",
    )

    return _plot_output(axes, img_path, show)


def plot_alpha_distr(alpha_distr, img_path: str = None, show: bool = False):
    """Placeholder"""

    axes = sns.lineplot(alpha_distr)
    axes.set(
        title="Alpha Score Distribution",
        xlabel="Alpha Score",
        ylabel="Count",
    )

    return _plot_output(axes, img_path, show)


def plot_annot_dynamics(
    ann_dynamics: pd.DataFrame,
    alpha_distr: pd.DataFrame,
    ann_name: str,
    sort_rows: bool = True,
    img_path: str = None,
    show: bool = False,
):
    """Placeholder."""

    def get_color_map():
        """Placeholder"""
        cols = ["mediumblue", "blue", "white", "red", "firebrick"]
        vals = [0, 0.15, 0.5, 0.85, 1]
        cmap = LinearSegmentedColormap.from_list("rg", list(zip(vals, cols)))
        return cmap

    def get_row_order(dataf):
        """Placeholder"""
        link = linkage(dataf, optimal_ordering=True)
        dendro = dendrogram(link, no_plot=True)
        return dendro["leaves"]

    ann_cols = [f"{ann_name}1", f"{ann_name}2"]
    table = ann_dynamics.copy()

    # Quantiles threhsolding values
    thresholds = table["alpha"].unique()

    # Compute total background table
    bkg = table.groupby(ann_cols)["num_pixels"].sum().reset_index()
    bkg["num_pixels"] = bkg["num_pixels"] / bkg["num_pixels"].sum()
    bkg["annot"] = bkg.pop(ann_cols[0]) + "-" + bkg.pop(ann_cols[1])
    bkg.set_index(["annot"], inplace=True)

    # Change main table index
    table["annot"] = table.pop(ann_cols[0]) + "-" + table.pop(ann_cols[1])
    table = table.set_index(["annot", "alpha"]).squeeze().unstack()

    # Trasform main table in log2 fold change
    for col in table.columns:
        table[col] = table[col] / table[col].sum()
        table[col] = table[col].divide(bkg["num_pixels"], fill_value=0)
    table = np.log2(table)

    # Whether to perform hierarchical clustering on the rows
    if sort_rows:
        table = table.iloc[get_row_order(table)]

    fig, axes = plt.subplots(
        2, 2, height_ratios=[1, 3], width_ratios=[20, 1], figsize=(12, 9)
    )

    sns.lineplot(alpha_distr, ax=axes[0][0])
    axes[0][0].set(
        xlabel="Alpha",
        ylabel="Number of Pixels",
        xlim=(0, thresholds[-1] * 1.01),
        ylim=(-max(alpha_distr) * 0.1, max(alpha_distr) * 1.1),
    )
    axes[0][0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0][0].xaxis.set_label_coords(-0.05, -0.04)

    for t in thresholds:
        axes[0][0].axvline(t, color="red")

    _empty_subplot(axes[0][1])

    sns.heatmap(
        table,
        ax=axes[1][0],
        cmap=get_color_map(),
        center=0,
        cbar_ax=axes[1][1],
    )
    axes[1][0].set(xlabel=None, ylabel=None, yticklabels=table.index)
    axes[1][0].tick_params(bottom=False)

    fig.tight_layout(pad=0)

    return _plot_output([fig, axes], img_path, show)


def plot_annot_prop(dataf):
    """Placeholder."""

    CHROMOSOMES = [f"chr{n}" for n in range(1, 23)] + ["chrX", "chrY"]

    dataf = dataf[dataf["chrom"].isin(CHROMOSOMES)]

    fig, axes = plt.subplots(1, 1)

    sns.barplot(dataf, x="chrom", y="a_frac", hue="a_annot", order=CHROMOSOMES, ax=axes)
    plt.show()
