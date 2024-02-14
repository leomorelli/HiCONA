"""Placeholder"""

import abc
from collections.abc import Callable
import inspect


import hicona.processing.filt_funs as ffuns
import hicona.processing.norm_funs as nfuns
from ..utils.io_ops import read_resource


SCHEDULERS_PATH = "schedulers.json"  # TODO: Settings?


class OpsScheduler(abc.ABC):
    """Abstract base class for pixel table operation schedulers."""

    def __init__(self):
        self._scheduled = []
        self._funcs_mod = self._load_module()
        self._available = self._load_available()

    @property
    def scheduled(self) -> list[tuple[Callable, dict]]:
        """Scheduled functions as list of (function, kwargs) pairs."""
        return self._scheduled

    @property
    def available(self) -> dict[str, Callable]:
        """Available functions as list of (name, function) pairs."""
        return self._available

    def _load_available(self):
        """Fetch all functions available by default."""
        return dict(inspect.getmembers(self._funcs_mod, inspect.isfunction))

    @abc.abstractmethod
    def _load_module(self):
        """Load the module to fetch the default functions from."""

    def add_function(self, fun_obj: str | Callable, fun_kwargs: dict):
        """Add a function (with its arguments) to the schedule."""

        if isinstance(fun_obj, str):
            fun_obj = self._available.get(fun_obj)

        if not inspect.isfunction(fun_obj):
            raise ValueError(f"{fun_obj} is not a function object.")

        self._scheduled.append((fun_obj, fun_kwargs))

    def remove_function(self, fun_obj: str | Callable):
        """Remove all instances of a function from the schedule."""

        fun_obj = fun_obj if isinstance(fun_obj, str) else fun_obj.__name__
        self._scheduled = [f for f in self._scheduled if f[0] != fun_obj]

    def reset_schedule(self):
        """Remove all functions from the schedule."""

        self._scheduled = []


class FiltScheduler(OpsScheduler):
    """Scheduler of filtering functions to apply to a pixel table.

    To schedule a function, provide the function and a dictionary of kwargs
    to pass to the function. Functions available by default can be obtained
    through the `available` attribute. Any function can be passed as long as:
    - it takes a pixel table as input
    - it returns an iterator of processed chunks as pandas DataFrame.
    """

    def _load_module(self):
        return ffuns


class NormScheduler(OpsScheduler):
    """Scheduler of normalization functions to apply to a pixel table.

    To schedule a function, provide the function and a dictionary of kwargs
    to pass to the function. Functions available by default can be obtained
    through the `available` attribute. Any function can be passed as long as:
    - it takes a pixel table as input
    - it returns an iterator of processed chunks as pandas DataFrame.
    """

    def _load_module(self):
        return nfuns


class ProcessScheduler:
    """Class used to group operations schedulers for a sparsification run."""

    def __init__(self, json_data):
        self._pre_filters = FiltScheduler()
        self._norm_method = NormScheduler()
        self._post_filters = FiltScheduler()

        pass

    @property
    def pre_filters(self) -> FiltScheduler:
        """Scheduler of the pre-normalization filters."""
        return self._pre_filters

    @property
    def norm_method(self) -> NormScheduler:
        """Scheduler of the normalization procedure."""
        return self._norm_method

    @property
    def post_filters(self) -> FiltScheduler:
        """Scheduler of the post-normalization filters."""
        return self._post_filters

    def to_json(self, out_path: str):
        """Save the object to a json file for later retrieval."""
        pass


def default_scheduler(norm_method: str | None = "hicona") -> ProcessScheduler:
    """Return template operation schedulers for the provided normalization.

    Provides a dictionary with the schedulers as values and the corresponding
    `HiconaCooler.create_table` parameter as as key (this way the return can
    be directly passed as kwargs).

    Parameters
    ----------
    norm_method: str or None, optional
        Normalization method for which to fetch the template schedulers.
        If set to None, return empty schedulers. Default is "hicona".

    Returns
    -------
    Dictionary with the schedulers as values.
    """

    defaults = read_resource(SCHEDULERS_PATH)
    pass


def load_scheduler(json_path: str) -> ProcessScheduler:
    """Placeholder"""

    pass
