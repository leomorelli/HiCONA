"""Module containing abstract base classes for filtering and normalization ops."""

import abc
import copy
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from hicona._core import Table
    from hicona._dtypes import PdChunks, KwargsDict


class DocStringInheritor(abc.ABCMeta):
    """Class to handle docstring inheritance.

    Allow derived classes to inherit docstrings from their base classes
    if the methods are not documented in the derived class.
    """

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        for key, value in attrs.items():
            if not value.__doc__:
                for base in bases:
                    base_method = getattr(base, key, None)
                    if base_method and base_method.__doc__:
                        value.__doc__ = base_method.__doc__
                        break
        return cls


class OperationABC(metaclass=DocStringInheritor):
    """Abstract Base Class for any filtering or normalization operation."""

    @abc.abstractmethod
    def run(self, table: "Table") -> "PdChunks":
        """Process the table and return processed table chunks.

        Parameters
        ----------
        table : Table
            Table instance to process.

        Returns
        -------
        Generator of pandas.DataFrame
            A generator of filtered pixel chunks.
        """

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Initialize the operation with the given parameters."""
        self._kwargs: KwargsDict = kwargs  # Never called, but needed to avoid errors.

    @property
    def kwargs(self) -> "KwargsDict":
        """Return the keyword arguments used to initialize the operation."""
        return self._kwargs

    @property
    def name(self) -> str:
        """Return the name of the operation."""
        return self.__class__.__name__

    def get_json(self) -> dict[str, Any]:
        """Return the operation as a json-like dictionary."""
        return {"name": self.__class__.__name__, "kwargs": self._kwargs}

    def __new__(cls, **kwargs):

        # TODO: Avoid positional arguments to be passed to the constructor.
        # TODO: Add check that all kwargs are serializable to json.

        # Storing given kwargs for usage in the json representation.
        instance = super().__new__(cls)
        setattr(instance, "_kwargs", copy.deepcopy(kwargs))

        return instance


class NormOperation(OperationABC):  # pylint: disable=abstract-method
    """Abstract class for any normalization operation.

    Any custom normalization operation should inherit from this class and
    override the `run` method with the function of interest. The run method
    should not take any parameters aside from the `Table` instance.

    If any parameter needs to be passed, override the class constructor.
    All parameters in the class constructor should be provided as keyword
    arguments and should be in a format that can be serialized to json.
    """

    def __init__(self):
        pass


class FiltOperation(OperationABC):  # pylint: disable=abstract-method
    """Abstract class for any filtering operation.

    Any custom filtering operation should inherit from this class and
    override the `run` method with the function of interest. The run method
    should not take any parameters aside from the `Table` instance.

    If any parameter needs to be passed, override the class constructor.
    All parameters in the class constructor should be provided as keyword
    arguments and should be in a format that can be serialized to json.
    """

    def __init__(self):
        pass


class SparOperation(OperationABC):  # pylint: disable=abstract-method
    """Abstract class for any sparsification operation.

    Any custom sparsification operation should inherit from this class and
    override the `run` method with the function of interest. The run method
    should not take any parameters aside from the `Table` instance.

    If any parameter needs to be passed, override the class constructor.
    All parameters in the class constructor should be provided as keyword
    arguments and should be in a format that can be serialized to json.
    """

    def __init__(self):
        pass
