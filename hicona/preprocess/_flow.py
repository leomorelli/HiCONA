"""Module to handle filtering and normalization functions for a table.

This module contains the `Flow` class, which is used to schedule
and retrieve the operations applied to a pixel table in order to filter and
normalize it.

"""

import importlib
import inspect
from typing import Iterable, Type, TYPE_CHECKING, Union

from hicona._ops import io
from hicona.preprocess import _DEFAULT_FLOWS, _FUNCS_MODULES

if TYPE_CHECKING:
    from hicona._dtypes import JsonDict, Operation, KwargsDict


__all__ = ["Flow"]


# TODO: Maybe make the Flow immutable when fetching a table,
#       requiring to create a copy of the object to modify it again.


def _fetch_funs(
    fun_names: list[str],
    fun_kwargs: list["KwargsDict"],
    sources: list[str],
) -> list["Operation"]:
    """Fetch preprocessing functions from their names."""

    funs: dict[str, Type["Operation"]] = {}
    for mod in sources:
        mod_obj = importlib.import_module(mod)
        mod_funs = inspect.getmembers(mod_obj)
        funs.update({k: v for k, v in mod_funs if k in mod_obj.__all__})

    outs: list["Operation"] = []
    for ind, name in enumerate(fun_names):
        fun_class: Union[Type["Operation"], None] = funs.get(name)

        if not fun_class:
            raise ValueError(f"{name} is not a default function.")

        outs.append(fun_class(**fun_kwargs[ind]))

    return outs


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

    def __init__(self, name: str, operations: Iterable["Operation"] | None = None):
        self._name: str = name
        self._ops: tuple[Operation, ...] = tuple(operations) if operations else tuple()

    def __eq__(self, other: "Flow") -> bool:
        """Check equality among ProcessingFlow objects."""
        return self.to_json() == other.to_json()

    def __len__(self) -> int:
        """Return the number of operations in the flow."""
        return len(self._ops)

    def __str__(self) -> str:
        """Return the operations flow as a string."""

        out_str = f"Flow '{self.name}':\n"

        for op in self._ops:
            out_str += f" - {op.name} -> {op.kwargs}\n"

        if len(self) == 0:
            out_str += " - No operations added yet."

        return out_str.strip()

    @property
    def name(self) -> str:
        """Return the name of the flow."""
        return self._name

    def rename(self, new_name: str) -> "Flow":
        """Rename the flow.

        Since the `Flow` object is immutable, this method returns a new
        instance of the object with the new name.

        Parameters
        ----------
        new_name: str
            The new name for the flow.

        Returns
        -------
        Flow
            Flow with the new name.

        Examples
        --------
        Rename the flow:

        >>> from hicona.preprocess import Flow
        >>> flow = Flow.from_default("hicona")
        >>> print(flow.name)
        'hicona'
        >>> flow = flow.rename("new_name")
        >>> print(flow.name)
        'new_name'

        """

        return Flow(name=new_name, operations=self._ops)

    @classmethod
    def from_dict(cls, name: str, flow_dict: "JsonDict") -> "Flow":
        """Create a flow from a dictionary."""

        # Sort flow_dict by keys to ensure order and check for gaps
        sort_dict: JsonDict = dict(sorted(flow_dict.items(), key=lambda x: x[0]))
        if not all(str(i) in sort_dict for i in range(len(sort_dict))):
            raise ValueError("The json data is not properly formatted.")

        names_list: list[str] = [op["name"] for op in sort_dict.values()]
        kwargs_list: list[KwargsDict] = [op["kwargs"] for op in sort_dict.values()]
        operations = _fetch_funs(names_list, kwargs_list, _FUNCS_MODULES)

        return cls(name, operations)

    @classmethod
    def from_json(cls, json: str) -> "Flow":
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

        json_data: dict[str, JsonDict] = io.read_resource(json)
        name, data = json_data.popitem()
        return cls.from_dict(name, data)

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
        return cls.from_dict(default_name, default_json[default_name])

    @property
    def ops(self) -> tuple["Operation", ...]:
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
        >>> [print(f.__name__, kwargs) for f, kwargs in flow.ops()]
        ('filt_self_looping', {})
        ('filt_inter_chroms', {})
        ('filt_genomic_dist', {'min_dist': None, 'max_dist': 200000000})
        ('norm_genomic_dist', {'apply_col': 'count'})
        ('filt_column_quant', {'lower_quant': 0.05, 'upper_quant': None ...  # etc
        """

        return self._ops

    def ops_reset(self) -> "Flow":
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

        return Flow(name=self.name)

    def ops_add(self, operation: "Operation") -> "Flow":
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

        return Flow(name=self.name, operations=[*self._ops, operation])

    def to_json(self, file_path=None | str) -> "JsonDict":
        """Return the flow in a json-like dictionary, optionally save it.

        Convert the flow to a json-like dictionary, where the keys are the
        order of the operations and the values are dictionaries with the
        operation name and its kwargs. This is done to preserve the order.

        Returns
        -------
        A dictionary with the operations flow in json-like format.
        """

        json_res: JsonDict = {
            str(order): {"name": op.name, "kwargs": op.kwargs}
            for order, op in enumerate(self._ops)
        }
        full_json = {self.name: json_res}

        if isinstance(file_path, str):
            io.write_resource(file_path, full_json, is_static=False)

        return full_json
