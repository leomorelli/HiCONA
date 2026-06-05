from importlib.util import find_spec as _find_spec

if _find_spec("graph_tool") is None:
    raise ImportError(
        "graph-tool is required but is not installed.\n"
        "Install it via conda:\n\n"
        "    conda install -c conda-forge graph-tool\n\n"
        "See https://graph-tool.skewed.de for details."
    )
del _find_spec

from . import misc, plotting  # noqa: E402
from ._core import (  # noqa: E402
    BinTable,
    HiconaCooler,
    HiconaGraph,
    PixelTable,
    strategies,
)

__all__ = [
    "BinTable",
    "HiconaCooler",
    "HiconaGraph",
    "PixelTable",
    "misc",
    "plotting",
    "strategies",
]
