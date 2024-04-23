"""Module to handle filtering and normalization functions for a table.

This module contains the `Flow` class, which is used to schedule
and retrieve the operations applied to a pixel table in order to filter and
normalize it.
"""

from functools import partial
from importlib import import_module
from inspect import getmembers, Parameter, signature
from typing import Any, Callable, Generator

from hicona._ops import io


__all__ = ["Flow"]


# TODO: move hardcoded paths to settings.
DEFAULT_FLOWS: str = "flows.json"
DEFAULT_MODULES: list[str] = [".preprocess.filt", ".preprocess.norm"]


# TODO: Maybe make the Flow immutable when fetching a table,
#       requiring to create a copy of the object to modify it again.


class Operation:
    """Individual filtering or normalization operation.

    Object to store, handle and retrieve information about a single filtering
    or normalization operation to apply on a table.

    Parameters
    ----------
    fun_obj: Callable
        The function object to be applied to the table.
    fun_kwargs: dict[str, Any]
        The kwargs to be passed to the function object.
    """

    def __init__(self, fun_obj: Callable, fun_kwargs: dict[str, Any]):
        self._fun_name = fun_obj.__name__
        self._fun_obj = fun_obj

        # Complete kwargs with default values, for non provided ones
        # This is done to avoid json mismatch due to implied defaults
        def_kwargs = {
            name: value.default
            for name, value in signature(fun_obj).parameters.items()
            if value.default is not Parameter.empty
        }
        def_kwargs.update(fun_kwargs)
        self._fun_kwargs = def_kwargs

        # NOTE: no check on mandatory arguments, since an error would be
        # raised at runtime anyway if they are not provided

    @property
    def fun_name(self) -> str:
        """Get function name in string form."""
        return self._fun_name

    @property
    def fun_kwargs(self) -> dict[str, Any]:
        """Get function kwargs."""
        return self._fun_kwargs

    def partial(self) -> partial:
        """Return a partial function for the operation."""
        return partial(self._fun_obj, **self._fun_kwargs)

    def json(self) -> dict[str, Any]:
        """Return the operation in a json-like dictionary."""
        return {"name": self._fun_name, "kwargs": self._fun_kwargs}


class Flow:
    """Class to handle filtering and normalization functions for a table.

    This class is used to schedule and retrieve the operations applied to a
    pixel table in order to filter and normalize it. The object can be
    initialized empty (default class constructor), with a default flow (using
    the `from_default` method), from a json-like dictionary (using the
    `from_json` method) or from a file containing a previously saved flow
    (using the `from_file` method).

    Any custom function can be added to the workflow as long as:
    - It takes a table object as its first argument
    - All other arguments are JSON data-types
    - It returns an iterator of processed chunks
    """

    def __init__(self):
        self._ops_flow: list[Operation] = []
        self._source_default: dict[str, Callable] = {}
        self._source_custom: dict[str, Callable] = {}

        for mod in DEFAULT_MODULES:
            mod = import_module(mod, package="hicona")
            funs = {k: v for k, v in getmembers(mod) if k in mod.__all__}
            self._source_default.update(funs)

    def __eq__(self, other: "Flow") -> bool:
        """Check equality among ProcessingFlow objects."""
        return self.as_json() == other.as_json()

    def __str__(self) -> str:
        """Return the operations flow as a string."""

        out_str = "Operations flow:\n"

        any_op = False
        for op in self._ops_flow:
            any_op = True
            out_str += f" - {op.fun_name} -> {op.fun_kwargs}\n"

        if not any_op:
            out_str += " - No operations added yet."

        return out_str.strip()

    @classmethod
    def from_json(
        cls,
        json_data: dict[int, dict[str, Any]],
        sources: list[Callable] | None = None,
    ) -> "Flow":
        """Create an instance with the provided functions already added.

        Create an instance of `Flow` with the provided operations
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
        A `Flow` object with the provided operations already added.
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
    ) -> "Flow":
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
        A `Flow` object with the operations flow from the file.
        """

        json_data = io.read_resource(file_path)
        return cls.from_json(json_data, sources)

    @classmethod
    def from_default(cls, default_name: str) -> "Flow":
        """Load a default Flow.

        Load a default Flow from the available ones. The default
        name should be one of the available ones, otherwise an error is raised.

        Parameters
        ----------
        default_name: str
            The name of the default flow to load.

        Returns
        -------
        A `Flow` object with the default operations flow.
        """

        default_json = io.read_resource(DEFAULT_FLOWS, is_static=True)
        return cls.from_json(default_json[default_name])

    def _get_from_source(self, fun_name: str) -> Callable:
        """Get a function from the available sources (if present)."""

        fun_obj: Callable | None = self._source_custom.get(fun_name)
        fun_obj = fun_obj or self._source_default.get(fun_name)

        if not fun_obj:
            raise ValueError(f"{fun_name} is not available. Add it to sources first.")

        return fun_obj

    def source_add(self, fun_obj: Callable) -> None:
        """Add a custom function to the available operations.

        Custom functions must be added as a source in order to be added to
        the operations flow. See class constructor documentation for the
        requirements of a custom function to be added to the flow.

        Parameters
        ----------
        fun_obj: Callable
            The custom function to add to the available sources.
        """

        self._source_custom[fun_obj.__name__] = fun_obj

    def ops_show(self) -> None:
        """Display the operations flow."""

        print(self)

    def ops_reset(self) -> None:
        """Reset the operations flow."""

        self._ops_flow = []

    def ops_add(self, fun_name: str, fun_kwargs: dict | None = None) -> None:
        """Add a new operation to the operations flow.

        Add a new operation to the operations flow. The function name should
        be one of the available ones, otherwise an error is raised. The kwargs
        are optional and should be provided as a dictionary.

        Parameters
        ----------
        fun_name: str
            The name of the function to add to the operations flow.
        fun_kwargs: dict or None, optional
            The kwargs to pass to the function. Default is None.
        """

        # Fetch function and raise error if it is not available
        fun_obj = self._get_from_source(fun_name)
        fun_kwargs = fun_kwargs or {}
        self._ops_flow.append(Operation(fun_obj, fun_kwargs))

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
            if self._ops_flow[i].fun_name == fun_name:
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

        for fun_obj in self._ops_flow:
            yield fun_obj.partial()

    def as_json(self) -> dict[str, dict[str, Any]]:
        """Return the flow in a json-like dictionary.

        Convert the flow to a json-like dictionary, where the keys are the
        order of the operations and the values are dictionaries with the
        operation name and its kwargs. This is done to preserve the order.

        Returns
        -------
        A dictionary with the operations flow in json-like format.
        """

        json_res = {
            str(order): {"name": op.fun_name, "kwargs": op.fun_kwargs}
            for order, op in enumerate(self._ops_flow)
        }

        return json_res

    def save_to_json(self, json_path: str) -> None:
        """Save the operations flow to a json file.

        Save the operations flow to a json file for later retrieval.
        If non-default functions are present, during loading these should
        be provided as a list of functions to the `from_file` method.
        """

        io.write_resource(json_path, self.as_json(), is_static=False)
