"""Standardized plotting functions created using the custom tracks."""

import re
from functools import partial
from typing import Callable, Iterable, Literal

import coolbox.api as ca
import matplotlib as mpl
from matplotlib.figure import Figure
import polars as pl

from .._core.table import PixelTable
from ._styling import get_rc_style
from .coolbox_api import (
    XAxis,
    BinClusters,
    ChromName,
    PixelClusters,
    PixelCounts,
    PixelProbs,
)


__all__ = ("plot_comparison", "plot_clustering", "plot_table")


_DEFAULT_PLOT_STYLE_WIDTH: dict[str, int] = {
    "publication": 9,
    "slides": 25,
}

_HEADER_RATIO = 16  # width/x is the height of header tracks
_BIN_CLUST_RATIO = 20  # width/x is the height of bin cluster tracks
# Maybe make height function of font size? Else rescaling breaks too much

ClusterStyle = Literal["bins", "pixels"]
MatrixStyle = Literal["matrix", "triangular", "window"]
PlotStyle = Literal["publication", "slides"]


def _extended_chrom_format(region, table) -> str:
    """Ensure that the genomic query is in the form `chrom:start-end`."""

    if len(region.split(":")) == 1:  # "chrN" format
        _, upper_bin = table.bins.extent(region)

        bin_table: pl.DataFrame = table.bins.get_dataframe()
        row: dict[str, str | int] = bin_table.row(
            by_predicate=(pl.col("bin_id") == (upper_bin - 1)),
            named=True,
        )
        region = f"{row['chrom']}:0-{row['end']}"

    return region


def _infer_pix_track(col_name: str) -> Callable:
    """Infer the desired pixel track depending on the name of the column."""

    # NOTE: this is kind of ugly, but both match case as well as get from a dict do
    # not allow the use of regexes, which are needed in this circumstance.
    if col_name == "count":
        return partial(PixelCounts, value_col=col_name)
    elif re.search("prob", col_name):
        return partial(PixelProbs, value_col=col_name)
    elif re.search("level", col_name):
        level = re.search(r"\d+", col_name)  # TODO: Not the best, first match
        if not level:
            raise ValueError("Clustering column does not contain level number.")
        return partial(PixelClusters, level=level.group())
    else:
        raise ValueError(f"Cannot infer track type for column `{col_name}`.")


def _get_frame_width(width: float | None, style: str) -> float:
    """Return the actual float value of the width."""

    if not width:
        width = _DEFAULT_PLOT_STYLE_WIDTH.get(style)
    if not width:
        raise ValueError("Must provide a width value for custom formats.")
    return width


def _plot_figure(
    region: str,
    *,
    main_tracks: list[ca.Track],
    user_tracks: None | ca.Track | Iterable[ca.Track],
    width: float | None,
    style: str,
    path: str | None
):
    """Generalized function to set aspects which are shared by all types of plots."""

    # Header tracks
    width = _get_frame_width(width, style)
    height_unit = width / _HEADER_RATIO
    frame: ca.Frame = ChromName(height=height_unit) + XAxis(height=height_unit)  # type: ignore
    frame.properties["width"] = width

    # Additional user provided tracks
    user_tracks = [] if user_tracks is None else user_tracks
    user_tracks = [user_tracks] if isinstance(user_tracks, ca.Track) else user_tracks
    # NOTE: uncomment this code if you decide to go back to the idea of arbitrary units
    # for track in user_tracks:
    #     if not hasattr(track, "get_track_height"):
    #         if not track.properties.get("height"):
    #             track.properties["height"] = 1
    #         track.properties["height"] *= UNIT_RATIO
    main_tracks.extend(user_tracks)

    # Add all tracks
    for track in main_tracks:
        frame += track  # type: ignore
    assert isinstance(frame, ca.Frame)

    # Plot using a style sheet
    with mpl.rc_context(fname=get_rc_style(style)):
        figure: Figure = frame.plot(region)

    # Save figure
    if path:
        figure.savefig(path)

    return figure


def plot_clustering(
    table: "PixelTable",
    region: str,
    *,
    modality: ClusterStyle = "pixels",
    min_level: int = 0,
    max_level: int | None = None,
    depth_ratio: float = 0.5,
    tracks: ca.Track | Iterable[ca.Track] | None = None,
    style: PlotStyle | str = "slides",
    width: float | None = None,
    path: str | None = None,
) -> Figure:
    """Plot hierarchical clustering results for a pixel table.

    Given a pixel table on which hierarchical clustering was performed, create a
    plot displaying the results below the count matrix.

    Parameters
    ----------
    table : PixelTable
        Table on which hierachical clustering was performed.
    region : str
        Genomic region of interest.
    modality : "lane" or "matrix", optional
        Modality in which to display the results of the clustering.

        - "lane": display the bin cluster levels as rows below the count matrix. Default.
        - "matrix": display the clustering levels using pixel matrices, where each pixel
          is colored according to its cluster if both bins belong to the same cluster.

    min_level : int, optional
        First level of the clustering hierarchy to plot. Default is 0.
    max_level : int, optional
        Lase level of the clustering hierarchy to plot. Default is 1.
    depth_ratio : float, optional
        Fraction of the height of the pixels to display. Default is 0.5.
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the default ones. Track height is treated as centimeters.
    style : "slides", "publication" or path string, optional
        Name of a default plotting style or path to a valid mplstyle file.
        Default is `slides`.
    width : float, optional
        Width of the plot in centimeters. Required when using a custom plotting style,
        otherwise default width for the default style.
    path : str, optional
        If provided, save the figure at that position. Default is None.

    Returns
    -------
    Figure
        The composite figure.

    """

    region = _extended_chrom_format(region, table)

    _BIG_NUMBER: int = 10_000  # Unreasonably big number for cluster level

    # How to plot clustering tracks
    match modality:
        case "bins":
            track_height = _get_frame_width(width, style) / _BIN_CLUST_RATIO
            factory = partial(
                BinClusters,
                table.bins,
                depth_ratio=depth_ratio,
                height=track_height,
            )
        case "pixels":
            factory = partial(PixelClusters, table, depth_ratio=depth_ratio)
        case _:
            raise ValueError(f"`{modality}` is not a valid modality.")

    # Main tracks definition
    main_tracks: list[ca.Track] = [PixelCounts(table, depth_ratio=depth_ratio)]
    max_level = max_level if max_level is not None else _BIG_NUMBER
    for lev in range(min_level, max_level + 1):
        try:
            main_tracks.append(factory(lev))
        except KeyError:
            break

    return _plot_figure(
        region,
        main_tracks=main_tracks,
        user_tracks=tracks,
        width=width,
        style=style,
        path=path,
    )


def plot_table(
    table: "PixelTable",
    region: str,
    *,
    value_col: str = "count",
    modality: MatrixStyle = "triangular",
    depth_ratio: float = 0.5,
    tracks: ca.Track | Iterable[ca.Track] | None = None,
    style: PlotStyle | str = "slides",
    width: float | None = None,
    path: str | None = None,
) -> Figure:
    """Plot some column from a pixel table.

    Parameters
    ----------
    table : PixelTable
        Pixel table to plot
    region : str
        Genomic region of interest.
    value_col : str, optional
        Column from the pixel table to use as values for the plot. Column name is also
        used to infer the type of track to use for plotting. Default is "count".
    modality : "matrix", "triangular" or "window", optional
        Display modality of the matrix track. See coolbox documentation for more
        information. Default is "window".
    depth_ratio : float, optional
        Fraction of the height of the pixels to display. Ignored if `modality = matrix`.
        Default is 0.5.
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the default ones. Track height is treated as centimeters.
    style : "slides", "publication" or path string, optional
        Name of a default plotting style or path to a valid mplstyle file.
        Default is `slides`.
    width : float, optional
        Width of the plot in centimeters. Required when using a custom plotting style,
        otherwise default width for the default style.
    path : str, optional
        If provided, save the figure at that position.

    Returns
    -------
    Figure
        The composite figure.

    """

    region = _extended_chrom_format(region, table)

    main_track: ca.Track = _infer_pix_track(value_col)(
        table, style=modality, depth_ratio=depth_ratio
    )

    return _plot_figure(
        region,
        main_tracks=[main_track],
        user_tracks=tracks,
        width=width,
        style=style,
        path=path,
    )


def plot_comparison(
    tables: "PixelTable" | Iterable["PixelTable"],
    region: str,
    *,
    value_cols: str | Iterable[str] = "count",
    depth_ratio: float = 0.5,
    tracks: ca.Track | Iterable[ca.Track] | None = None,
    style: PlotStyle | str = "slides",
    width: float | None = None,
    path: str | None = None,
) -> Figure:
    """Plot the comparison of pixel tables and value columns.

    Either plot multiple value columns from the same pixel table, or the same value
    column for multiple pixel tables. If only two tracks are plotted, the second one
    will have inverted orientation to facilitate comparison.

    Parameters
    ----------
    tables : PixelTable or iterable of them.
        The pixel tables(s) from which to extract the value columns.
    region : str
        Genomic region of interest.
    value_cols : str or iterable of them, optional
        Column(s) from the pixel table(s) to use as values for the tracks in the plot.
    depth_ratio : float, optional
        Fraction of the height of the pixels to display. Default is 0.5.
        Default is "count".
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the default ones. Track height is treated as centimeters.
    style : "slides", "publication" or path string, optional
        Name of a default plotting style or path to a valid mplstyle file.
        Default is `slides`.
    width : float, optional
        Width of the plot in centimeters. Required when using a custom plotting style,
        otherwise default width for the default style.
    path : str, optional
        If provided, save the figure at that position.

    Returns
    -------
    Figure
        The composite figure.

    """

    # Input checks
    tables = [tables] if isinstance(tables, PixelTable) else [*tables]
    value_cols = [value_cols] if isinstance(value_cols, str) else [*value_cols]

    if len(value_cols) > 1 and len(tables) > 1:
        raise ValueError("Multiple value cols are accepted only with 1 pixel table.")

    if any([len(value_cols) == 0, len(tables) == 0]):
        raise ValueError("Expected at least one table and one value column.")

    region = _extended_chrom_format(region, tables[0])

    # Main tracks definition
    main_tracks: list[ca.Track]
    match len(tables), len(value_cols):
        case 1, 2:
            main_tracks = [
                _infer_pix_track(value_cols[0])(tables[0], depth_ratio=depth_ratio),
                _infer_pix_track(value_cols[1])(
                    tables[0], orientation="inverted", depth_ratio=depth_ratio
                ),
            ]
        case 2, 1:
            main_tracks = [
                _infer_pix_track(value_cols[0])(tables[0], depth_ratio=depth_ratio),
                _infer_pix_track(value_cols[0])(
                    tables[1], orientation="inverted", depth_ratio=depth_ratio
                ),
            ]
        case 1, _:
            main_tracks = [
                _infer_pix_track(col)(tables[0], depth_ratio=depth_ratio)
                for col in value_cols
            ]
        case _, 1:
            main_tracks = [
                _infer_pix_track(value_cols[0])(tab, depth_ratio=depth_ratio)
                for tab in tables
            ]
        case _, _:
            raise ValueError("This branch should not be reachable, some check failed.")

    return _plot_figure(
        region,
        main_tracks=main_tracks,
        user_tracks=tracks,
        width=width,
        style=style,
        path=path,
    )
