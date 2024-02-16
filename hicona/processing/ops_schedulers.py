"""Scheduler object to customize normalization procedure.

Module containing objects (Schedulers) used to customize the functions
(Operations) to apply during pre-normalization filtering, normalization
and post-normalization filtering. These objects are implemented as 
generally as possible in order to allow for the usage of custom filtering
or normalization functions.

User is not meant to inteface with object constructors directly. Using the
creator functions one can instantiate a `ProcedureScheduler`, whose attributes
are Scheduler objects which can be modified.

# TODO: add support for loading custom function from json procedure shedulers.
"""

import abc
from collections.abc import Callable
import functools
import inspect


import hicona.processing.filt_funs as ffuns
import hicona.processing.norm_funs as nfuns
from ..utils.io_ops import read_resource, write_resource


__all__ = ["default_scheduler", "load_scheduler"]


SCHEDULERS_PATH = "schedulers.json"  # TODO: Settings?


class Operation:
    """Individual operation to perform on a table.

    It is assumed that the `Callable` takes as first argument a pixel table,
    and that the pixel table is not in kwargs.

    Parameters
    ----------
    func: `Callable`
        Function callable.
    kwargs: dict
        Kwargs to pass to the function call.
    """

    def __init__(self, func: Callable, kwargs: dict):
        self._name = func.__name__
        self._func = func
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        """Shorthand for self.func.__name__"""
        return self._name

    @property
    def func(self) -> Callable:
        """Function callable."""
        return self._func

    @property
    def kwargs(self) -> dict:
        """Kwargs to pass to function call."""
        return self._kwargs

    def get_partial(self) -> functools.partial:
        """Return function partial signature missing only table as arg."""
        return functools.partial(self._func, **self._kwargs)


class OpsScheduler(abc.ABC):
    """Abstract base class for pixel table operation schedulers."""

    def __init__(self):
        self._scheduled = []
        self._available = self._load_available()

    @property
    def scheduled(self) -> dict[str, dict]:
        """Scheduled functions as a dictionary {fun_name: fun_kwargs}."""
        return {op.name: op.kwargs for op in self._scheduled}

    @property
    def available(self) -> dict[str, Callable]:
        """Available functions as a dictionary {fun_name: fun_object}."""
        return {op.name: op.func for op in self._available}

    def _load_available(self):
        """Fetch all functions available by default."""
        fun_module = self._load_module()
        funs = inspect.getmembers(fun_module, inspect.isfunction)
        return [Operation(name, obj) for name, obj in funs]

    @abc.abstractmethod
    def _load_module(self):
        """Load the module to fetch the default functions from."""

    def add_operation(self, fun_obj: str | Callable, fun_kwargs: dict):
        """Add a function (with its arguments) to the schedule."""

        if isinstance(fun_obj, str):
            fun_obj = self._available.get(fun_obj)

        if not inspect.isfunction(fun_obj):
            raise ValueError(f"{fun_obj} is not a function object.")

        self._scheduled.append(Operation(fun_obj, fun_kwargs))

    def remove_operation(self, fun_obj: str | Callable):
        """Remove all instances of a function from the schedule."""

        name = fun_obj if isinstance(fun_obj, str) else fun_obj.__name__
        self._scheduled = [op for op in self._scheduled if op.name != name]

    def reset_operations(self):
        """Remove all functions from the schedule."""

        self._scheduled = []

    def get_partials(self) -> tuple(functools.partial):
        """Return partial functions for all scheduled operations."""

        return (op.get_partial() for op in self._scheduled)


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
    """Class used to group operations schedulers for a sparsification run.

    The attributes of this object are `Scheduler` objects, to which one can
    add, modify or remove the operations to perform during sparsification.
    Any function can be passed as a filter or normalization step as long as
    it takes as first input a pixel table object and returns an iterator of
    processed pixel chunks.

    NOTE: This object is not meant for direct initialization but rather for
    initialization through the `default_scheduler` and the `load_scheduler`
    functions.

    Parameters
    ----------
    json_data: dict
        The data used to construct the schedulers, in json format.
    """

    def __init__(self, json_data: dict):
        self._pre_filters = FiltScheduler()
        self._norm_method = NormScheduler()
        self._post_filters = FiltScheduler()

        # Populate the schedules with the filters in the provided json data
        for attr_name, ops in json_data.items():
            sched = getattr(self, attr_name)
            for op_name, op_kwargs in ops:
                sched.add_operation(op_name, op_kwargs)

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

        data = {s: dict(o.scheduled) for s, o in self.__dict__.items()}
        write_resource(out_path, data)


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
    A `ProcessScheduler` object with operations already initialized.
    """

    default_json = read_resource(SCHEDULERS_PATH)
    return ProcessScheduler(default_json.get(norm_method))


def load_scheduler(json_path: str) -> ProcessScheduler:
    """Load a `ProcessScheduler` previously stored in a json file.

    Automatically instantiate a `ProcessScheduler` object using the data
    coming from a json, which is an instance of `ProcessScheduler` which was
    previously stored.

    NOTE: currently only schedules with all operations corresponding to
    default functions from HiCONA can be loaded from json. Loading custom
    functions-containing schedules will be implemented in the future.

    Parameters
    ----------
    json_path: str
        Path to the json file containing the scheduler data.

    Returns
    -------
    A `ProcessScheduler` object with operations already initialized.
    """

    custom_json = read_resource(json_path)
    return ProcessScheduler(custom_json)
