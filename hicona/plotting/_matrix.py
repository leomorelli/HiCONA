"""Plotting functions for contact matrices."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import matplotlib.axes as ax
import matplotlib.gridspec as gsp
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from .._core.table import PixelTable

__all__ = ["CmapSpecs", "get_cmap_specs", "plot_pixel_matrix"]
_DefaultSpecs = Literal["hic", "clusters"]


@dataclass
class CmapSpecs:
    """Class to simplify argument passing to matrix plotting functions.

    Parameters
    ----------
    palette : str
        The colormap to use (must be recognized by matplotlib or seaborn).
    log_scale : bool, optional
        Whether to apply a logarithmic scale to the values, by default False.
    cbar_show : bool, optional
        Whether to show the colorbar in the final plot, by default True.
    cbar_title : str, optional
        The title of the colorbar, by default None.
    cbar_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to the colorbar, by default None.
    """

    palette: str
    log_scale: bool = False
    cbar_show: bool = True
    cbar_title: str | None = None
    cbar_kwargs: dict[str, Any] | None = None


def get_cmap_specs(specs: _DefaultSpecs) -> CmapSpecs:
    """Return one of the predefined cmap specs.

    Parameters
    ----------
    specs : {"hic", "clusters"}
        The name of the predefined cmap specs.

    Returns
    -------
    CmapSpecs
        The cmap specs object.
    """

    match specs:
        case "hic":
            return CmapSpecs("viridis", log_scale=True, cbar_title="ln\ncounts")
        case "clusters":
            return CmapSpecs("tab20", log_scale=False, cbar_show=False)
        case _:
            raise ValueError(f"Unknown cmap specs: {specs}")


def _get_axes(num_cbars: int) -> tuple[ax.Axes, ax.Axes | None, ax.Axes | None]:
    """Return the main and colorbar axes."""

    match num_cbars:
        case 0:
            fig = plt.figure(figsize=(8, 8))
            return fig.add_subplot(111), None, None
        case 1:
            fig = plt.figure(figsize=(8.8, 8))
            gs = gsp.GridSpec(100, 2, width_ratios=[30, 1])
            ax_main = fig.add_subplot(gs[:, 0])
            ax_cbar = fig.add_subplot(gs[30:70, 1])
            return ax_main, ax_cbar, None
        case 2:
            fig = plt.figure(figsize=(8.8, 8))
            gs = gsp.GridSpec(100, 2, width_ratios=[30, 1])
            ax_main = fig.add_subplot(gs[:, 0])
            ax_cbar_top = fig.add_subplot(gs[10:40, 1])
            ax_cbar_bot = fig.add_subplot(gs[60:90, 1])
            return ax_main, ax_cbar_top, ax_cbar_bot
        case _:
            raise ValueError(f"Invalid number of colorbars: {num_cbars}")


def plot_pixel_matrix(
    table_a: PixelTable,
    *,
    table_b: PixelTable | None = None,
    region: str | None = None,
    value_cols: str | tuple[str, str] = ("count", "count"),
    cmap_specs: CmapSpecs | tuple[CmapSpecs, CmapSpecs] | None = None,
) -> tuple[ax.Axes, ...]:
    """Plot one or more contact matrices in a single figure.

    Parameters
    ----------
    table_a : PixelTable
        Table to plot in the upper triangle.
    table_b : PixelTable, optional
        Table to plot in the lower triangle. If None, it defaults to `table_a`.
    region : str, optional
        Optional genomic region subsetting.
    value_cols : str or tuple[str, str], optional
        The columns to use as values in the matrices. If a string, it will be
        used for both matrices. If a tuple, the first element will be used for
        the upper triangle and the second for the lower triangle. By default,
        ("count", "count").
    cmap_specs : CmapSpecs or tuple[CmapSpecs, CmapSpecs], optional
        The colormap specifications for the matrices. If a single object, it
        will be used for both matrices. If a tuple, the first element will be
        used for the upper triangle and the second for the lower triangle. By
        default, use the default "hic" cmap specs.

    Returns
    -------
    tuple[ax.Axes, ...]
        The main and colorbar axes (if any).
    """

    # TODO: check for definition on the same bins

    ###########################################################################
    # Make inputs consistent
    ###########################################################################

    if table_b is None:
        table_b = table_a

    if isinstance(value_cols, str):
        value_cols = (value_cols, value_cols)

    if cmap_specs is None:
        cmap_specs = get_cmap_specs("hic")
    if isinstance(cmap_specs, CmapSpecs):
        cmap_specs = (cmap_specs, cmap_specs)

    ###########################################################################
    # Get the matrices
    ###########################################################################

    matrix_a: np.ndarray = table_a.get_matrix(
        region,
        value_col=value_cols[0],
        mask_diagonal=True,
        mode="upper",
    )
    matrix_b: np.ndarray = table_b.get_matrix(
        region,
        value_col=value_cols[1],
        mask_diagonal=True,
        mode="lower",
    )

    if cmap_specs[0].log_scale:
        matrix_a = np.log1p(matrix_a)
    if cmap_specs[1].log_scale:
        matrix_b = np.log1p(matrix_b)

    ###########################################################################
    # Plot the matrices
    ###########################################################################

    same_cmap: bool = cmap_specs[0].palette == cmap_specs[1].palette
    num_axes: int = sum(s.cbar_show for s in cmap_specs)
    num_axes = max(0, num_axes - 1) if same_cmap else num_axes

    main_ax, cbar_top, cbar_bot = _get_axes(num_axes)

    if same_cmap:
        matrix = matrix_a + matrix_b
        sns.heatmap(
            matrix,
            ax=main_ax,
            cmap=cmap_specs[0].palette,
            cbar=cmap_specs[0].cbar_show,
            cbar_ax=cbar_top,
            square=True,
        )
    else:
        sns.heatmap(
            matrix_a,
            ax=main_ax,
            cmap=cmap_specs[0].palette,
            cbar=cmap_specs[0].cbar_show,
            cbar_ax=cbar_top,
            mask=np.tril(np.ones_like(matrix_a, dtype=bool)),
            square=True,
        )
        sns.heatmap(
            matrix_b,
            ax=main_ax,
            cmap=cmap_specs[1].palette,
            cbar=cmap_specs[1].cbar_show if not same_cmap else False,
            cbar_ax=cbar_bot,
            mask=np.triu(np.ones_like(matrix_b, dtype=bool)),
            square=True,
        )

    if cbar_top and cmap_specs[0].cbar_title:
        cbar_top.set_title(cmap_specs[0].cbar_title)
    if cbar_bot and cmap_specs[1].cbar_title:
        cbar_bot.set_title(cmap_specs[1].cbar_title)

    main_ax.set_xticks([])  # Remove x-axis ticks
    main_ax.set_yticks([])  # Remove y-axis ticks
    plt.subplots_adjust(hspace=0, wspace=0)

    return tuple(a for a in [main_ax, cbar_top, cbar_bot] if a is not None)
