"""Module to handle filtering and normalization flow for a table.

This module contains the `Flow` class, which is used to schedule
and retrieve the operations applied to a pixel table in order to filter
and normalize it.

"""

import importlib
import inspect
from typing import Iterable, Type, TYPE_CHECKING, Union

from hicona._ops import io
from hicona.preprocess import _DEFAULT_FLOWS, _FUNCS_MODULES

if TYPE_CHECKING:
    from hicona._dtypes import JsonDict, Operation, KwargsDict


__all__ = ["Flow"]


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
    """Class to organize filtering and normalization steps for a table.

    This class is used to specify an ordered list of filtering, normalization
    and sparsification steps, which can be used to create or fetch a pixel table.

    The object can be initialized using its class constructor or one of the
    class methods. Unless a custom ``Flow`` is required, usually the objects
    will be initialized using the ``from_default`` method.

    Instances of this class are immutable to prevent accidental changes to the
    flow of processed tables, therefore all methods that modify the flow
    return a new instance of the object with the changes applied.


    See Also
    --------
    hicona.HiconaTable :
        The class to handle pixel tables.
    hicona.preprocess.Flow.from_default :
        Load a default flow from the ones available from HiCONA.

    Examples
    --------

    Create an empty flow using its constructor:

    >>> import hicona.preprocess as prep
    >>> flow = prep.Flow("my_flow")  # Empty flow to add functions to


    Load a flow using the class methods:

    >>> flow = prep.Flow.from_default("hicona")
    >>> flow = prep.Flow.from_file("path/to/file.json")

    """

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
            out_str += " - No operations."

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

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_default("hicona")
        >>> print(flow.name)
        'hicona'
        >>> flow = flow.rename("new_name")
        >>> print(flow.name)
        'new_name'

        """

        return Flow(name=new_name, operations=self._ops)

    @classmethod
    def from_json(cls, name: str, flow_dict: "JsonDict") -> "Flow":
        """Create a flow from a json-like dictionary.

        Create an instance of ``Flow`` from a json-like dictionary representation,
        that is, a dictionary in the format ``{order: {"name": "op_name", "kwargs": {}}}``.

        Parameters
        ----------
        name: str
            The name of the flow.
        flow_dict: dict
            The operations to add to the flow, in json format.

        Returns
        -------
        Flow
            Workflow with the provided operations already added.

        Examples
        --------
        Create a flow from a json-like dictionary:

        >>> import hicona.preprocess as prep
        >>> json = {0: {"name": "FiltInterChroms", "kwargs": {}}}
        >>> flow = prep.Flow.from_json("my_flow", json)

        """

        # Sort flow_dict by keys to ensure order and check for gaps
        sort_dict: JsonDict = dict(sorted(flow_dict.items(), key=lambda x: x[0]))
        if not all(str(i) in sort_dict for i in range(len(sort_dict))):
            raise ValueError("The json data is not properly formatted.")

        names_list: list[str] = [op["name"] for op in sort_dict.values()]
        kwargs_list: list[KwargsDict] = [op["kwargs"] for op in sort_dict.values()]
        operations = _fetch_funs(names_list, kwargs_list, _FUNCS_MODULES)

        return cls(name, operations)

    @classmethod
    def from_file(cls, json: str) -> "Flow":
        """Create an instance from a json file.

        Load a flow which was previously saved to a json file using the method
        ``to_file``.

        .. warning::
            Currently, the method only supports loading flows with all default
            functions. Custom functions are not supported yet but will be in the
            future.


        Parameters
        ----------
        json: str
            The path to the json file containing the flow.

        Returns
        -------
        Flow
            Workflow with the provided operations already added.

        Examples
        --------
        Create a flow from a json file:

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_file("path/to/file.json")

        """

        json_data: dict[str, JsonDict] = io.read_resource(json)
        name, data = json_data.popitem()
        return cls.from_json(name, data)

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

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_default("hicona")
        >>> print(flow)
        Flow 'hicona':
        - FiltSelfLooping -> {}
        - FiltInterChroms -> {}
        - FiltGenomicDist -> {'max_dist': 200000000}
        - NormGenomicDist -> {'apply_col': 'count'}
        - FiltColumnQuant -> {'apply_col': 'norm', 'lower_quant': 0.05}

        """

        default_json = io.read_resource(_DEFAULT_FLOWS, is_static=True)
        return cls.from_json(default_name, default_json[default_name])

    @property
    def ops(self) -> tuple["Operation", ...]:
        """Return the operations in the flow as a tuple of operations.

        Returns
        -------
        tuple of Operation objects
            The operations in the flow.

        Examples
        --------
        Get the operations in the flow as a tuple:

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_default("hicona")
        >>> print(flow.ops)
        (FiltSelfLooping, FiltInterChroms, FiltGenomicDist, NormGenomicDist, FiltColumnQuant)
        """

        return self._ops

    def ops_reset(self) -> "Flow":
        """Reset the operations flow.

        Reset the operations flow, removing all operations added so far.

        Returns
        -------
        Flow
            Empty flow.

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

        Given an instance of a class that inherits from either ``FiltOperation``,
        ``NormOperation`` or ``SparOperation``, add it as the last operation in the flow.

        .. warning::
            Currently the method only supports adding default functions. Custom
            functions are not supported yet but will be in the future.

        Parameters
        ----------
        operation: Operation
            The operation to add to the flow.

        Returns
        -------
        Flow
            Flow with the new operation added.

        Examples
        --------
        Add a new operation to the flow:

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_default("hicona")
        >>> print(len(flow))
        5
        >>> flow = flow.ops_add(prep.NormBinwise())
        >>> print(len(flow))
        6

        Adding operations can be done in a chain-like fashion:

        >>> flow = (
        ...     prep.Flow("custom_flow")
        ...     .ops_add(prep.FiltSelfLooping())
        ...     .ops_add(prep.FiltInterChroms())
        ...     .ops_add(prep.FiltGenomicDist(min_dist=1000000))
        ...     .ops_add(prep.NormGenomicDist(apply_col="count"))
        ...     .ops_add(prep.FiltColumnQuant(apply_col="norm", lower_quant=0.05))
        ... )
        >>> print(len(flow))
        5

        """

        return Flow(name=self.name, operations=[*self._ops, operation])

    def to_file(self, file_path: str) -> None:
        """Save a flow to a json file.

        Save a flow to a json file so that it can be loaded later.

        Parameters
        ----------
        file_path: str
            The path to save the flow to.

        Examples
        --------
        Save a flow to a json file after modifying it:

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow.from_default("hicona")
        >>> flow.to_file("path/to/file.json")

        """

        full_json = {self.name: self.to_json()}

        io.write_resource(file_path, full_json, is_static=False)

    def to_json(self) -> "JsonDict":
        """Return the flow as a json-like dictionary.

        Convert the flow to a json-like dictionary, that is, a dictionary in the
        format ``{order: {"name": "op_name", "kwargs": {}}}``.

        Returns
        -------
        dict
            The flow as a json-like dictionary.

        Examples
        --------
        Get the flow as a json-like dictionary:

        >>> import hicona.preprocess as prep
        >>> flow = prep.Flow("my_flow").add(prep.FiltInterChroms())
        >>> print(flow.to_json())
        {0: {"name": "FiltInterChroms", "kwargs": {}}}

        """

        json_res: JsonDict = {
            str(order): {"name": op.name, "kwargs": op.kwargs}
            for order, op in enumerate(self._ops)
        }

        return json_res
