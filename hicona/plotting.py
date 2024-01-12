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


def plot_alpha_grid(alpha_grid, img_path: str = None, show: bool = False):
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

    if img_path:
        plt.savefig(img_path)

    if show:
        plt.show()


def plot_alpha_distr(alpha_distr, img_path: str = None, show: bool = False):
    """Placeholder"""

    axes = sns.lineplot(alpha_distr)
    axes.set(
        title="Alpha Score Distribution",
        xlabel="Alpha Score",
        ylabel="Count",
    )

    if img_path:
        plt.savefig(img_path)

    if show:
        plt.show()


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

    print(table.groupby("alpha")["num_pixels"].sum())
    print(bkg)

    # Change main table index
    table["annot"] = table.pop(ann_cols[0]) + "-" + table.pop(ann_cols[1])
    table = table.set_index(["annot", "alpha"]).squeeze().unstack()

    # Trasform main table in log2 fold change
    for col in table.columns:
        table[col] = table[col] / table[col].sum()
        table[col] = table[col].divide(bkg["num_pixels"], fill_value=0)
    table = np.log2(table)
    print(table)

    # Whether to perform hierarchical clustering on the rows
    if sort_rows:
        table = table.iloc[get_row_order(table)]

    fig, axes = plt.subplots(2, 1, height_ratios=[1, 3], figsize=(12, 9))

    print(thresholds[-1] * 1.1)
    sns.lineplot(alpha_distr, ax=axes[0])
    axes[0].set(
        xlabel="Alpha",
        ylabel="Number of Pixels",
        # xticks=thresholds,
        xlim=(0, thresholds[-1] * 1.01),
        ylim=(-max(alpha_distr) * 0.1, max(alpha_distr) * 1.1),
    )
    # axes[0].set(xticklabels=[])
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].xaxis.set_label_coords(-0.05, -0.04)

    for t in thresholds:
        axes[0].axvline(t, color="red")

    sns.heatmap(
        table,
        robust=True,
        cmap=get_color_map(),
        # cbar_kws={"pad": 0.05},
        cbar_kws={"location": "bottom", "shrink": 0.5, "pad": 0.05},
        ax=axes[1],
        yticklabels=table.index,
        center=0,
        # vmax=1.5,
        # vmin=-1.5,
    )
    axes[1].set(xlabel=None, ylabel=None)  # , xticklabels=[])
    axes[1].tick_params(bottom=False)

    fig.tight_layout(pad=0)

    if img_path:
        plt.savefig(img_path)

    if show:
        plt.show()


# THRESHOLD, NON CUMULATIVE VERSION
# def plot_annot_dynamics(
#     ann_dynamics: pd.DataFrame,
#     alpha_distr: pd.DataFrame,
#     ann_name: str,
#     sort_rows: bool = True,
#     img_path: str = None,
#     show: bool = False,
# ):
#     """Placeholder."""

#     def get_color_map():
#         """Placeholder"""
#         cols = ["mediumblue", "blue", "white", "red", "firebrick"]
#         vals = [0, 0.15, 0.5, 0.85, 1]
#         cmap = LinearSegmentedColormap.from_list("rg", list(zip(vals, cols)))
#         return cmap

#     def get_row_order(dataf):
#         """Placeholder"""
#         link = linkage(dataf, optimal_ordering=True)
#         dendro = dendrogram(link, no_plot=True)
#         return dendro["leaves"]

#     ann_cols = [f"{ann_name}1", f"{ann_name}2"]
#     table = ann_dynamics.copy()

#     bkg = table.groupby(ann_cols)["num_pixels"].sum().reset_index()
#     bkg["num_pixels"] = bkg["num_pixels"] / bkg["num_pixels"].sum()
#     bkg["annot"] = bkg.pop(ann_cols[0]) + "-" + bkg.pop(ann_cols[1])
#     bkg.set_index(["annot"], inplace=True)

#     table["annot"] = table.pop(ann_cols[0]) + "-" + table.pop(ann_cols[1])
#     table = table.set_index(["annot", "alpha"]).squeeze().unstack()
#     # table = table.drop(columns=[c for c in table.columns if c > max_alpha])

#     for col in table.columns:
#         table[col] = table[col] / table[col].sum()
#         table[col] = table[col].divide(bkg["num_pixels"], fill_value=0)
#     table = np.log2(table)

#     if sort_rows:
#         table = table.iloc[get_row_order(table)]

#     fig, axes = plt.subplots(2, 1, height_ratios=[1, 3], figsize=(15, 10))

#     sns.lineplot(alpha_distr, ax=axes[0])
#     axes[0].set(
#         xlabel="Alpha",
#         ylabel="Number of Pixels",
#         # xticks=[0.05 * i for i in range(0, round(max_alpha / 0.05) + 1)],
#         # xlim=(0, max_alpha),
#         ylim=(-max(alpha_distr) * 0.1, max(alpha_distr) * 1.1),
#     )
#     axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
#     axes[0].xaxis.set_label_coords(-0.05, -0.04)

#     sns.heatmap(
#         table,
#         robust=True,
#         cmap=get_color_map(),
#         cbar_kws={"location": "bottom", "shrink": 0.5, "pad": 0.05},
#         ax=axes[1],
#         yticklabels=table.index,
#         center=0,
#         # vmax=1.5,
#         # vmin=-1.5,
#     )
#     axes[1].set(xlabel=None, ylabel=None, xticklabels=[])
#     axes[1].tick_params(bottom=False)

#     fig.tight_layout(pad=0)

#     if img_path:
#         plt.savefig(img_path)

#     if show:
#         plt.show()


# CUMULATIVE VERSION
# def plot_annot_dynamics(
#     ann_dynamics: pd.DataFrame,
#     alpha_distr: pd.DataFrame,
#     ann_name: str,
#     img_path: str = None,
#     show: bool = False,
# ):
#     """Placeholder."""

#     table = ann_dynamics.copy()

#     table["annot"] = table.pop("HMM_annot1") + "-" + table.pop("HMM_annot2")
#     table = table.set_index(["annot", "alpha"]).squeeze().unstack()

#     _, axes = plt.subplots(2, 1)

#     sns.heatmap(table, robust=True, cbar=False, ax=axes[0])
#     sns.lineplot(alpha_distr, ax=axes[1])

#     axes[1].set_xlim(0, 0.65)
#     plt.show()


# Plotting individual matrices for dynamics
# ann_columns = [f"{ann_name}1", f"{ann_name}2"]

# alpha_vals = ann_dynamics["alpha"].unique()
# alpha_vals[::-1].sort()
# num_cols = 2  # 1 if len(alpha_vals) == 1 else 2
# num_rows = int(round_half_up(len(alpha_vals) / 2))

# _, axes = plt.subplots(num_rows, num_cols)

# for ind, val in enumerate(alpha_vals):
#     print(axes)
#     print(ind)
#     print(axes[ind])
#     table = ann_dynamics.query(f"alpha == {val}").drop(columns=["alpha"])
#     table = table.set_index(ann_columns).squeeze().unstack().T
#     print(table)
#     sns.heatmap(table, annot=True, ax=axes[ind])
