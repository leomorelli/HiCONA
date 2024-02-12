"""Placeholder"""

import h5py

from .table_ops import get_dataf_mapping


def fetch_chunk(store, path, lower, upper, keys=None):
    """Retrieve a subset of the table as a pandas dataframe."""

    with h5py.File(store, mode="r") as h5_handle:
        group = h5_handle[path]
        keys = keys or group.keys()
        table = pd.DataFrame({f: group[f][lower:upper] for f in keys})

    return table


# TODO: Add lock
def write_chunk(store, path, chunk, lower, keys=None):
    """Write pandas dataframe data at the provided position in the table."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]
        upper = len(chunk)
        keys = keys or group.keys()
        for key in keys:
            group[key][lower:upper] = chunk[key]


# TODO: Add lock
def resize_table(store, path, size):
    """Resize a table by cropping it at the provided index."""

    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle[path]
        keys = group.keys()
        for key in keys:
            group[key].resize((size,))


# TODO: Add lock
def init_table(store, path, size, col_mapping):
    """Initialize dataframe columns as 1D arrays."""

    opts = {"compression": "gzip"}
    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle.require_group(path)
        for name, dtype in col_mapping.items():
            group.require_dataset(name, shape=(size,), dtype=dtype, **opts)


# TODO: Add lock
def save_table(store, path, table):
    """Convenience shorthand to initialize and place table at once."""

    init_table(store, path, len(table), get_dataf_mapping(table))
    write_chunk(store, path, table, 0)


def group_info(store, path, info):
    """Return some group information (keys or attrs for example)."""

    items = []
    try:
        with h5py.File(store, mode="r") as h5_handle:
            group = h5_handle[path]

            match info:
                case "keys":
                    items.extend(group.keys())
                case "attrs":
                    items.extend(group.attrs)

    except KeyError:
        pass

    return items


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
