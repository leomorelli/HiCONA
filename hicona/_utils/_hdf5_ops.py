"""Placeholder"""

from typing import Any

import h5py
import pandas as pd

from ._table_ops import get_dataf_mapping

# NOTE: during key selection, `is None` is used rather than `or` to avoid
#       `ValueError: The truth value of a Index is ambiguous`

# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// CHUNK OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def fetch_chunk(store, path, lower, upper, keys=None) -> pd.DataFrame:
    """Retrieve a subset of the table as a pandas dataframe."""

    with h5py.File(store, mode="r") as h5_handle:
        group = h5_handle[path]
        keys = group.keys() if keys is None else keys
        table = pd.DataFrame({f: group[f][lower:upper] for f in keys})

    return table


# TODO: Add lock
def write_chunk(store, path, chunk, lower, keys=None):
    """Write pandas dataframe data at the provided position in the table."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]
        upper = lower + len(chunk)

        if keys is None:
            keys = [k for k in group.keys() if k in chunk.columns]

        for key in keys:
            group[key][lower:upper] = chunk[key]


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// TABLE OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


# TODO: Add lock
def init_table(store, path, size, col_mapping):
    """Initialize dataframe columns as 1D arrays."""

    opts: dict[str, Any] = {"compression": "gzip"}
    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle.require_group(path)
        for name, dtype in col_mapping.items():
            group.require_dataset(name, shape=(size,), dtype=dtype, **opts)


# TODO: Add lock?
def write_table(store, path, iterator, keys=None) -> int:
    """Write chunk applied to iterator of chunks of the same table."""

    tab_size = 0
    for chunk in iterator:
        write_chunk(store, path, chunk, tab_size, keys)
        tab_size += len(chunk)

    return tab_size


# TODO: Add lock
def save_table(store, path, table):
    """Convenience shorthand to initialize and place table at once."""

    init_table(store, path, len(table), get_dataf_mapping(table))
    write_chunk(store, path, table, 0)


# TODO: Add lock
def resize_table(store, path, size):
    """Resize a table by cropping it at the provided index."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]
        for key in group.keys():
            group[key].resize((size,))


def get_table_size(store, path) -> int:
    """Fetch table size."""

    with h5py.File(store, mode="r") as h5_handle:
        group = h5_handle[path]
        size = set(len(group[k]) for k in group.keys())

        if len(size) != 1:
            raise ValueError("Inconsistent table size")

    return size.pop()


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// GROUP OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


# TODO: Add lock
def require_group(store, path, attrs_dict=None):
    """Requires a group and sets some attributes if non-existent."""

    attrs_dict = attrs_dict or {}

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle.require_group(path)

        for k, v in attrs_dict.items():
            if k not in group.attrs:
                group.attrs[k] = v


# TODO: Add lock
def set_attrs(store, path, attrs_dict):
    """Sets one or more group attributes (overwriting present)."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]

        for k, v in attrs_dict.items():
            group.attrs[k] = v


def get_attrs(store, path) -> dict[str, Any]:
    """Fetch table attributes."""

    with h5py.File(store, mode="r") as h5_handle:
        group = h5_handle[path]
        attrs = dict(group.attrs)

    return attrs


def del_keys(store, path, keys_list):
    """Deletes one or more table keys."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]

        for key in keys_list:
            del group[key]


def get_keys(store, path) -> list[str]:
    """Fetch table keys."""

    with h5py.File(store, mode="r") as h5_handle:
        group = h5_handle[path]
        keys = list(group.keys())

    return keys


# TODO: Remove
def get_subgroups_attrs(store, path) -> dict[str, dict[str, Any]]:
    """Fetch attributes of all subgroups in a group."""

    attrs = {}
    for table in get_keys(store, path):
        table_path = "/".join([path, table])
        attrs[table] = get_attrs(store, table_path)

    return attrs
