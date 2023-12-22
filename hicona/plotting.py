import matplotlib.pyplot as plt
import seaborn as sns


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
