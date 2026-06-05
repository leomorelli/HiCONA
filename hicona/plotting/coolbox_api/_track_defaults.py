"""Code to handle track default parameters selection."""

from typing import Any

from .._palettes import Palettes

__all__ = ("get_updated_defaults",)

_TRACK_DEFAULTS: dict[str, dict] = {
    "BinClusters": {
        "height": 1.0,
        "cmap": Palettes.CLUST.value,
    },
    "BinTrack": {},
    "PixelClusters": {
        "transform": False,
        "cmap": Palettes.CLUST.value,
        "color_bar": "no",
        "min_value": 1,
    },
    "PixelCounts": {
        "title": "Pixel\ncounts",
    },
    "PixelProbs": {
        "transform": False,
        "max_value": 1,
        "min_value": 0,
        "cmap": Palettes.PROB.value,
        "title": "Pixel\nprobs",
    },
    "PixelTrack": {
        "depth_ratio": 0.5,
    },
}


def get_updated_defaults(track_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the default kwargs for the track updated with the provided ones.

    Parameters
    ----------
    track_name : str
        Name of the class for which to fetch the defaults.
    kwargs : dict[str, Any]
        Provided kwargs.

    Returns
    -------
    dict
        Default parameters for the class updated with the provided ones.

    """

    defaults = _TRACK_DEFAULTS.get(track_name)
    if defaults is None:
        defaults = {}
        print(f"Class {track_name} not recognized, using default track params.")

    defaults = defaults.copy()
    defaults.update(kwargs)

    return defaults
