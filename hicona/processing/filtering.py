"""Placeholder."""

import importlib.resources as imp_res

from . import table_filter_funs as ffuns
from ..utils.hdf5_ops import resize_table, write_chunk
from ..utils.io_ops import read_resource


# TODO: Move into a module?
def apply_filter(table, filter_fun, **kwargs):
    """Apply a filtering function to a table."""

    filt_table_size = 0
    for chunk in filter_fun(table, **kwargs):
        write_chunk(table.store, table.pixels_uri, chunk, filt_table_size)
        filt_table_size += len(chunk)
    resize_table(table.store, table.pixels_uri, filt_table_size)


class FiltersManager:
    """Pixels table filters scheduler.

    Stores a list of filters to apply to a pixel table.
    Filters are applied in the order they are provided.
    A table filtering function is a function which takes as input a table
    object and some other keywords, then yields filtered chunks of the table.
    """

    def __init__(self):
        self._filters = []
        self._default = [f for f in dir(ffuns) if callable(getattr(ffuns, f))]

    @property
    def filters(self):
        """Currently scheduled filters with respective parameters."""
        return self._filters

    def available_filters(self):
        """Summary of the filter functions implemented by HiCONA."""

        sep_line = "-" * 79 + "\n"
        out = sep_line + "Available Filters\n" + sep_line
        for func in self._default:
            out += f"{func}:\n{getattr(ffuns, func).__doc__}\n"
        out += sep_line
        print(out.strip())

    def add_filter(self, filter_fun, fun_kwargs):
        """Add a filter to the workflow."""

        if isinstance(filter_fun, str):
            try:
                filter_fun = getattr(ffun, filter_fun)
            except AttributeError:
                raise ValueError(f"{filter_fun} is not a filter function.")

        self._filters.append([filter_fun, fun_kwargs])

    def remove_filter(self, filter_fun):
        """Remove all scheduled instances of the provided filter."""

        if callable(filter_fun):
            filter_fun = filter_fun.__name__

        kept = [f for f in self._filters if f[0].__name__ != filter_fun]
        self._filters = kept

    def reset_filters(self):
        """Reset all scheduled filters."""

        self._filters = []


def get_filter_manager(norm: str = None, pre: bool = True) -> FiltersManager:
    """Return a filter manager already populated with the default filters.

    Parameters
    ----------
    norm: str, optional
        Name of the normalization for which to fetch the defaults filters.
        Default is None.
    pre: bool, optional
        Whether to fetch the default filters for pre normalization filtering
        (True) or post normalization filtering (False). Default is True.

    Returns
    -------
    FiltersManager already populated with the default filters for the provided
    situation (if any).
    """

    manager = FiltersManager()

    defaults = read_resource("default_filters.json")  # TODO: settings?
    pre_post = "pre" if pre is True else "post"
    dict_name = "_".join(norm, pre_post)
    if norm_dict := defaults.get(dict_name):
        for fun_name, fun_kwargs in norm_dict.items():
            manager.add_filter(fun_name, fun_kwargs)

    return manager
