"""Scheduler object to customize normalization procedure.

Module containing objects (Schedulers) used to customize the functions
(Operations) to apply during pre-normalization filtering, normalization
and post-normalization filtering. These objects are implemented as 
generally as possible in order to allow for the usage of custom filtering
or normalization functions.

User is not meant to inteface with object constructors directly. Using the
creator functions one can instantiate a `ProcedureScheduler`, whose attributes
are Scheduler objects which can be modified.
"""

from collections.abc import Callable
from functools import partial
from importlib import import_module
from inspect import getmembers, isfunction
from typing import Any, Generator

from ..utils.io_ops import read_resource, write_resource


__all__ = ["ProcessingFlow"]


# TODO: move hardcoded paths to settings.
SCHEDULERS_PATH: str = "schedulers.json"
DEFAULT_MODULES: list[str] = [".processing.filt_funs", ".processing.norm_funs"]


class ProcessingFlow:
    """Placeholder."""

    def __init__(self):
        self._ops_flow: list = []
        self._source_default: dict[str, Callable] = {}
        self._source_custom: dict[str, Callable] = {}

        for mod in DEFAULT_MODULES:
            mod = import_module(mod, package="hicona")
            self._source_default.update(dict(getmembers(mod, isfunction)))

    def __eq__(self, other: "ProcessingFlow") -> bool:
        """Check equality among ProceProcessingFlow objects."""
        return self.as_json() == other.as_json()

    @classmethod
    def from_json(
        cls,
        json_data: dict[int, dict[str, Any]],
        sources: list[Callable] | None = None,
    ) -> "ProcessingFlow":
        """Create an instance with the provided functions already added.

        Create an instance of `ProcessingFlow` with the provided operations
        already added to the operations flow. The outer json keys should be
        the order of the operations, the intermediate keys the operations,
        and the inner most dictionary the kwargs to pass to the operation.

        Parameters
        ----------
        json_data: dict[int, dict[str, Any]]
            The operations to add to the flow, in json format.
        source: list[Callable] or None, optional
            Any non default function required by the workflow. These should
            take a `PixelTable` as first argument and return an iterator of
            processed chunks. Default is None.

        Returns
        -------
        A `ProcessingFlow` object with the provided operations already added.
        """

        flow = cls()

        if sources:
            for fun in sources:
                flow.source_add(fun)

        ind: int = 0
        while ind < len(json_data):

            key = str(ind)
            if key not in json_data:
                raise ValueError("The json data is not properly formatted.")

            op_dict = json_data[key]
            flow.ops_add(op_dict["name"], op_dict["kwargs"])

            ind += 1

        return flow

    @classmethod
    def from_file(
        cls,
        file_path: str,
        sources: list[Callable] | None = None,
    ) -> "ProcessingFlow":
        """Create an instance from a previously saved json file.

        Load a flow which was previously saved to json. If the flow contains
        any non default functions, these should be provided in the sources.

        Parameters
        ----------
        file_path: str
            Path to the json file containing the operations flow.
        source: list[Callable] or None, optional
            Any non default function required by the workflow. These should
            take a `PixelTable` as first argument and return an iterator of
            processed chunks. Default is None.

        Returns
        -------
        A `ProcessingFlow` object with the operations flow from the file.
        """

        json_data = read_resource(file_path)
        return cls.from_json(json_data, sources)

    @classmethod
    def from_default(cls, default_name: str) -> "ProcessingFlow":
        """Load a default ProcessingFlow.

        Load a default ProcessingFlow from the available ones. The default
        name should be one of the available ones, otherwise an error is raised.

        Parameters
        ----------
        default_name: str
            The name of the default flow to load.

        Returns
        -------
        A `ProcessingFlow` object with the default operations flow.
        """

        default_json = read_resource(SCHEDULERS_PATH)
        return cls.from_json(default_json[default_name])

    def _get_from_source(self, fun_name: str) -> Callable:
        """Get a function from the available sources (if present)."""

        fun_obj: Callable | None = self._source_custom.get(fun_name)
        fun_obj = fun_obj or self._source_default.get(fun_name)

        if not fun_obj:
            raise ValueError(f"{fun_name} is not available.")

        return fun_obj

    def source_add(self, fun_obj: Callable) -> None:
        """Add a custom function to the available ones."""

        self._source_custom[fun_obj.__name__] = fun_obj

    def ops_show(self) -> None:
        """Display the operations flow."""

        out_str = "Operations flow:\n"
        for op in self._ops_flow:
            out_str += f"{op.name} -> {op.kwargs}\n"

        print(out_str)

    def ops_reset(self) -> None:
        """Reset the operations flow."""

        self._ops_flow = []

    def ops_add(self, fun_name: str, fun_kwargs: dict) -> None:
        """Add a new operation to the operations flow."""

        _ = self._get_from_source(fun_name)  # Raise error is not available
        self._ops_flow.append([fun_name, fun_kwargs])

    def ops_remove(self, fun_name: str) -> None:
        """Remove the last occurrence of a function from the flow.

        Remove the last occurrence of a function from the operations flow.
        If the function is not present, exit silently.

        Parameters
        ----------
        fun_name: str
            The name of the function to remove.
        """

        for i in range(len(self._ops_flow) - 1, -1, -1):
            if self._ops_flow[i][0] == fun_name:
                del self._ops_flow[i]
                break

    def get_partials(self) -> Generator[partial, None, None]:
        """Return partial functions for all operations in the flow.

        Return a generator of partial functions for all operations in the flow,
        that is, the functions with all their arguments already set aside from
        the first one, which should be the pixel table. These functions can be
        used to apply the operations to a pixel table and return an iterator of
        processed chunks.

        Returns
        -------
        A generator of partial functions for all operations in the flow.
        """

        for fun_name, fun_kwargs in self._ops_flow:
            fun_obj = self._get_from_source(fun_name)
            yield partial(fun_obj, **fun_kwargs)

    def as_json(self) -> dict[int, dict[str, Any]]:
        """Return the flow in a json-like dictionary.

        Convert the flow to a json-like dictionary, where the keys are the
        order of the operations and the values are dictionaries with the
        operation name and its kwargs. This is done to preserve the order.

        Returns
        -------
        A dictionary with the operations flow in json-like format.
        """

        return {i: {"name": n, "kwargs": k} for i, (n, k) in enumerate(self._ops_flow)}

    def save_to_json(self, json_path: str) -> None:
        """Save the operations flow to a json file.

        Save the operations flow to a json file for later retrieval.
        If non-default functions are present, during loading these should
        be provided as a list of functions to the `from_file` method.
        """

        write_resource(json_path, self.as_json())
