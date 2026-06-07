"""Utilities for resolving matplotlib style sheet paths."""

import importlib.resources as pkg_resources
import os

import hicona.plotting._mpl_styles

STYLES_FOLDER: str = "_mpl_styles"


def get_default_style_names() -> tuple[str, ...]:
    """Return the names of all valid matplotlib rc styles."""
    with pkg_resources.path(hicona.plotting, STYLES_FOLDER) as styles_path:
        return tuple(f.split(".")[0] for f in os.listdir(styles_path))


def get_default_style_path(name: str) -> str:
    """Return the path to a style file."""

    if name not in get_default_style_names():
        raise FileNotFoundError(f"`{name}` is not a valid style name.")

    with pkg_resources.path(hicona.plotting, STYLES_FOLDER) as styles_path:
        return os.path.join(styles_path, f"{name}.mplstyle")


def get_rc_style(style: str) -> str:
    """Return the path to a default style or the input itself if it is a valid path."""

    # Custom style
    if os.path.isfile(style):
        return style

    # Default style
    try:
        style_path: str = get_default_style_path(style)
    except FileNotFoundError:
        err_msg: str = "Input must be either a valid mplstyle or a default style name."
        raise ValueError(err_msg)

    return style_path
