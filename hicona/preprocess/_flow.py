"""Module to handle filtering and normalization functions for a table.

This module contains the `Flow` class, which is used to schedule
and retrieve the operations applied to a pixel table in order to filter and
normalize it.

"""

from importlib import import_module
from inspect import getmembers, Parameter, signature
from typing import Any, Callable

from hicona._ops import io
from hicona._dtypes import JsonDict


__all__ = ["Flow"]


_DEFAULT_FLOWS: str = "flows.json"
_DEFAULT_MODULES: list[str] = [".preprocess._filt", ".preprocess._norm"]


# TODO: Maybe make the Flow immutable when fetching a table,
#       requiring to create a copy of the object to modify it again.


class Operation:
    """Individual filtering or normalization operation.

    Object to store, handle and retrieve information about a single filtering
    or normalization operation to apply on a table.

    Parameters
    ----------
    fun_obj : Callable
        The function object to be applied to the table.
    fun_kwargs : dict[str, Any]
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

    @property
    def fun_obj(self) -> Callable:
        """Get function object."""
        return self._fun_obj

    def json(self) -> dict[str, Any]:
        """Return the operation in a json-like dictionary."""
        return {"name": self._fun_name, "kwargs": self._fun_kwargs}


class Flow:
    """Class to organize filtering and normalization functions for a table.

    This class is used to specify a set of filtering and normalization
    functions (in an order sensitive manner), which can be used to create
    or fetch a processed pixel table.

    Though the object can be initialized directly using its constructor
    method (which takes no arguments), it should generally be created
    using one of its class methods (``from_default`` and ``from_json``).

    Any custom function can be added to the workflow as long as:

    - It takes an instance of ``HiconaTable`` as its first argument.
    - All other arguments are JSON data-types.
    - It returns an iterator of processed chunks as ``pandas.DataFrame``
      instances with the columns ``bin1_id``, ``bin2_id``, ``count`` and ``norm``.


    .. warning::
        No check is performed on the custom functions, since it is complex
        to test for a highly variable function signature. It is up to the
        user to ensure that the functions are correctly implemented; if
        they are not, the program might crash at runtime or, worst case,
        produce incorrect results. A check might me added in the future.

    .. warning::
        At the moment, it is manually needed to ensure that at least one
        normalization function is applied, that is, there there is at least
        one function which saves the values to the ``norm`` column. If no
        normalization is needed, add ``norm_none`` to the flow. This might
        be performed automatically in the future.

    See Also
    --------
    hicona.HiconaTable :
        The class to handle pixel tables.
    hicona.preprocess.Flow.from_default :
        Load a default flow from the ones available from HiCONA.
    hicona.preprocess.Flow.from_json :
        Load a flow from a json-like file or dictionary.
    hicona.preprocess.norm_none :
        A normalization function that only copies the values to the ``norm`` column.


    Examples
    --------

    Create an empty flow using its constructor:

    >>> from hicona.preprocess import Flow
    >>> flow = Flow()

    Load a default flow using the class methods:

    >>> flow = Flow.from_default("default_flow")
    >>> flow = Flow.from_json("path/to/file.json")

    """

    # TODO: find a way to remove the two warnings

    def __init__(self):
        self._ops_flow: list[Operation] = []
        self._default_funs: dict[str, Callable] = {}

        for mod in _DEFAULT_MODULES:
            mod = import_module(mod, package="hicona")
            funs = {k: v for k, v in getmembers(mod) if k in mod.__all__}
            self._default_funs.update(funs)

    def __eq__(self, other: "Flow") -> bool:
        """Check equality among ProcessingFlow objects."""
        return self.to_json() == other.to_json()

    def __len__(self) -> int:
        """Return the number of operations in the flow."""
        return len(self._ops_flow)

    def __str__(self) -> str:
        """Return the operations flow as a string."""

        out_str = "Operations flow:\n"

        for name, kwargs in self.ops_list():
            out_str += f" - {name} -> {kwargs}\n"

        if len(self._ops_flow) == 0:
            out_str += " - No operations added yet."

        return out_str.strip()

    def _fetch_fun(self, fun_name: str) -> Callable:
        """Fetch a function from its name."""

        fun_obj: Callable | None = globals().get(fun_name)
        fun_obj = fun_obj or self._default_funs.get(fun_name)

        if not fun_obj:
            raise ValueError(f"{fun_name} not found. Load it in the namespace.")

        return fun_obj

    @classmethod
    def from_json(cls, json: str | JsonDict) -> "Flow":
        """Create an instance from a json file or dictionary.

        Create an instance of ``Flow`` from a ``Flow`` previously saved to a
        json file or from a json-like dictionary representation of a ``Flow``.
        Any non default function required by the workflow must be accessible
        in the namespace.

        Parameters
        ----------
        json_data: str or json-like dictionary
            The operations to add to the flow, in json format.

        Returns
        -------
        Flow
            Workflow with the provided operations already added.

        Examples
        --------
        Create a flow from a json-like dictionary:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_json({0: {"name": "fun_name", "kwargs": {}})

        Create a flow from a json file:

        >>> flow = Flow.from_json("path/to/file.json")

        """

        # TODO: Maybe find an alternative to loading in the namespace

        flow = cls()

        # Convert to json-like dictionary if not already
        if isinstance(json, str):
            json_data: JsonDict = io.read_resource(json)
        else:
            json_data = json

        # Populate the flow with the operations
        ind: int = 0
        while ind < len(json_data):

            key = str(ind)
            if key not in json_data:
                raise ValueError("The json data is not properly formatted.")

            op_dict = json_data[key]
            flow.ops_add(flow._fetch_fun(op_dict["name"]), op_dict["kwargs"])

            ind += 1

        return flow

    @classmethod
    def from_default(cls, default_name: str) -> "Flow":
        """Load a default workflow provided by HiCONA.

        Load a default ``Flow`` from those available in HiCONA. The default
        name should be one of the available ones, otherwise an error is raised.

        Parameters
        ----------
        default_name: str
            The name of the default flow to load.

        Returns
        -------
        Flow
            Default workflow from HiCONA.

        Examples
        --------
        Load a default flow from HiCONA:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_default("hicona")
        >>> print(flow)
        Operations flow:
         - filt_self_looping -> {}
         - filt_inter_chroms -> {}
         - filt_genomic_dist -> {'min_dist': None, 'max_dist': 200000000}
         - norm_genomic_dist -> {'apply_col': 'count'}
         - filt_column_quant -> {'lower_quant': 0.05, 'upper_quant': None ...  # etc

        """

        default_json = io.read_resource(_DEFAULT_FLOWS, is_static=True)
        return cls.from_json(default_json[default_name])

    def ops_list(self) -> list[tuple[Callable, dict[str, Any]]]:
        """Return the operations flow as a list of operations.

        Return a list of tuples with the operations in the flow.
        Each tuple contains the function object and its kwargs.

        Returns
        -------
        list of tuples[Callable, dict[str, Any]]
            List of tuples with the operations in the flow.

        Examples
        --------
        List the operations in the flow:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_default("hicona")
        >>> [print(f.__name__, kwargs) for f, kwargs in flow.ops_list()]
        ('filt_self_looping', {})
        ('filt_inter_chroms', {})
        ('filt_genomic_dist', {'min_dist': None, 'max_dist': 200000000})
        ('norm_genomic_dist', {'apply_col': 'count'})
        ('filt_column_quant', {'lower_quant': 0.05, 'upper_quant': None ...  # etc
        """

        return [(op.fun_obj, op.fun_kwargs) for op in self._ops_flow]

    def ops_reset(self) -> None:
        """Reset the operations flow.

        Reset the operations flow, removing all operations added so far.

        Examples
        --------
        Reset the operations flow:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_default("hicona")
        >>> print(len(flow))
        5
        >>> flow.ops_reset()
        >>> print(len(flow))
        0

        """

        self._ops_flow = []

    def ops_add(self, fun_obj: Callable, fun_kwargs: dict | None = None) -> None:
        """Add a new operation to the operations flow.

        Add a new operation to the operations flow. The function can be either
        one of the default ones provided by HiCONA or a custom one, as long as
        it follows the requirements specified in the class docstring.

        Parameters
        ----------
        fun_obj: Callable
            The name of the function to add to the operations flow.
        fun_kwargs: dict or None, optional
            The kwargs to pass to the function. Default is 'None'.

        Examples
        --------
        Add a default operation to the flow:

        >>> from hicona.preprocess import Flow, filt_genomic_dist
        >>> flow = Flow.from_default("hicona")
        >>> print(len(flow))
        5
        >>> flow.ops_add(filt_genomic_dist, {"max_dist": 2000000})
        >>> print(len(flow))
        6

        Add a custom operation to the flow:

        >>> def custom_fun(table, arg1, arg2):
        ...     for chunk in table:
        ...         # Some custom code here
        ...         yield chunk
        ...
        >>> flow.ops_add(custom_fun, {"arg1": 1, "arg2": 2})
        >>> print(len(flow))
        7

        """

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

        Examples
        --------
        Remove an operation from the flow:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_default("hicona")
        >>> print(len(flow))
        5
        >>> flow.ops_remove("filt_genomic_dist")
        >>> print(len(flow))
        4

        """

        # TODO: change to take callables?
        for i in range(len(self._ops_flow) - 1, -1, -1):
            if self._ops_flow[i].fun_name == fun_name:
                del self._ops_flow[i]
                break

    def to_json(self, file_path=None | str) -> JsonDict:
        """Return the flow in a json-like dictionary, optionally save it.

        Convert the flow to a json-like dictionary, where the keys are the
        order of the operations and the values are dictionaries with the
        operation name and its kwargs. This is done to preserve the order.

        Returns
        -------
        A dictionary with the operations flow in json-like format.
        """

        json_res: JsonDict = {
            str(order): {"name": op.fun_name, "kwargs": op.fun_kwargs}
            for order, op in enumerate(self._ops_flow)
        }

        if isinstance(file_path, str):
            io.write_resource(file_path, json_res, is_static=False)

        return json_res
