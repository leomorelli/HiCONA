"""Standardized plotting functions created using the custom tracks."""

import re
from functools import partial

from typing import Callable, Iterable

import coolbox.api as ca
from matplotlib.figure import Figure

from hicona.plotting.tracks._pix_tracks import PixelProbs

from .._core.table import PixelTable
from ._dtypes import ClusterStyle, MatrixStyle
from .tracks import BinClusters, PixelClusters, PixelCounts


__all__ = ("plot_comparison", "plot_clustering", "plot_table")


# Mapping from plotting cluster modality to objects to use. In the tuple, in order:
# - Class of the track to use for each level
# - Lambda function to obtain the data input for the class starting from a PixelTable
_CLUSTER_STYLES: dict[str, tuple[Callable, Callable]] = {
    "lane": (BinClusters, lambda x: x.bins),
    "matrix": (PixelClusters, lambda x: x),
}
_DPIS: int = 300


def plot_clustering(
    table: "PixelTable",
    region: str,
    *,
    style: ClusterStyle = "lane",
    min_level: int = 0,
    max_level: int | None = None,
    tracks: ca.Track | Iterable[ca.Track] | None = None,
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
    style : "lane" or "matrix", optional
        Style in which to display the results of the clustering.

        - "lane": display the bin cluster levels as rows below the count matrix. Default.
        - "matrix": display the clustering levels using pixel matrices, where each pixel
          is colored according to its cluster if both bins belong to the same cluster.

    min_level : int, optional
        First level of the clustering hierarchy to plot. Default is 0.
    max_level : int, optional
        Lase level of the clustering hierarchy to plot. Default is 1.
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the clustering ones.
    path : str, optional
        If provided, save the figure at that position.

    Returns
    -------
    Figure
        The composite figure.
    """

    _BIG_NUMBER: int = 10_000  # Unreasonably big number for cluster level

    try:
        mod_class, mod_data = _CLUSTER_STYLES[style]
    except KeyError:
        raise KeyError(f"`{style}` is not a valid modality.")

    # Base plot
    frame: ca.Frame = ca.ChromName(fontsize=20) + ca.XAxis()  # type: ignore
    frame += PixelCounts(table)  # type: ignore

    # Clustering part
    max_level = max_level if max_level is not None else _BIG_NUMBER
    for lev in range(min_level, max_level + 1):
        try:
            frame += mod_class(mod_data(table), lev)  # type: ignore
            frame += ca.Spacer(0.1)  # type: ignore
        except KeyError:
            break

    # Add any additional user provided track
    if tracks:
        tracks = [tracks] if isinstance(tracks, ca.Track) else tracks
        for track in tracks:
            frame += track  # type: ignore

    assert isinstance(frame, ca.Frame)
    figure: Figure = frame.plot(region)

    if path:
        figure.savefig(path, dpi=_DPIS)

    return figure


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


def plot_table(
    table: "PixelTable",
    region: str,
    *,
    value_col: str = "count",
    style: MatrixStyle = "triangular",
    tracks: ca.Track | Iterable[ca.Track] | None = None,
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
    style : "matrix", "triangular" or "window", optional
        Display modality of the matrix track. See coolbox documentation for more
        information. Default is "window".
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the clustering ones.
    path : str, optional
        If provided, save the figure at that position.

    Returns
    -------
    Figure
        The composite figure.

    """

    main_track: ca.Track = _infer_pix_track(value_col)(table, style=style)

    frame: ca.Frame = ca.ChromName(fontsize=20) + ca.XAxis()  # type: ignore
    frame += main_track  # type: ignore

    # Add any additional user provided track
    if tracks:
        tracks = [tracks] if isinstance(tracks, ca.Track) else tracks
        for track in tracks:
            frame += track  # type: ignore

    assert isinstance(frame, ca.Frame)
    figure: Figure = frame.plot(region)

    if path:
        figure.savefig(path, dpi=_DPIS)

    return figure


def plot_comparison(
    tables: "PixelTable" | Iterable["PixelTable"],
    region: str,
    *,
    value_cols: str | Iterable[str] = "count",
    tracks: ca.Track | Iterable[ca.Track] | None = None,
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
        Default is "count".
    tracks : coolbox Tracks or an iterable of them, optional
        Any additional Track object (or instance of a class inheriting from it) to add
        below the clustering ones.
    path : str, optional
        If provided, save the figure at that position.

    Returns
    -------
    Figure
        The composite figure.

    """

    tables = [tables] if isinstance(tables, PixelTable) else [*tables]
    value_cols = [value_cols] if isinstance(value_cols, str) else [*value_cols]

    if len(value_cols) > 1 and len(tables) > 1:
        raise ValueError("Multiple value cols are accepted only with 1 pixel table.")

    if any([len(value_cols) == 0, len(tables) == 0]):
        raise ValueError("Expected at least one table and one value column.")

    # Create all tracks and their orientations
    frame: ca.Frame = ca.ChromName(fontsize=20) + ca.XAxis()  # type: ignore
    match len(tables), len(value_cols):
        case 1, 2:
            frame += _infer_pix_track(value_cols[0])(tables[0])  # type: ignore
            frame += _infer_pix_track(value_cols[1])(tables[0], orientation="inverted")  # type: ignore
        case 2, 1:
            frame += _infer_pix_track(value_cols[0])(tables[0])  # type: ignore
            frame += _infer_pix_track(value_cols[0])(tables[1], orientation="inverted")  # type: ignore
        case 1, _:
            for col in value_cols:
                frame += _infer_pix_track(col)(tables[0])  # type: ignore
        case _, 1:
            for tab in tables:
                frame += _infer_pix_track(value_cols[0])(tab)  # type: ignore

    # Add any additional user provided track
    if tracks:
        tracks = [tracks] if isinstance(tracks, ca.Track) else tracks
        for track in tracks:
            frame += track  # type: ignore

    assert isinstance(frame, ca.Frame)
    figure: Figure = frame.plot(region)

    if path:
        figure.savefig(path, dpi=_DPIS)

    return figure
