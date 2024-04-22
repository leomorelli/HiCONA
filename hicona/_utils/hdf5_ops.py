"""Placeholder"""

from typing import Any, Iterable

import h5py
import pandas as pd

from ._table_ops import get_dataf_mapping
from .dtypes import PdChunks
from .._core.uris import Uris

# NOTE: during key selection, `is None` is used rather than `or` to avoid
#       `ValueError: The truth value of a Index is ambiguous`


def _get_group(handle: h5py.File, path: str) -> h5py.Group:
    """Fetch a group from an open HDF5 file."""

    group = handle[path]
    if not isinstance(group, h5py.Group):
        raise ValueError(f"Got {type(group)}, expected h5py.Group.")
    return group


def _get_dataset(handle: h5py.Group, path: str) -> h5py.Dataset:
    """Fetch a dataset from an open HDF5 group."""

    dataset = handle[path]
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"Got {type(dataset)}, expected h5py.Dataset.")
    return dataset


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// CHUNK OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def fetch_chunk(
    uris: Uris,
    bounds: slice,
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Retrieve a subset of the table as a pandas dataframe."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        grp = _get_group(h5_handle, path)
        keys = list(grp.keys()) if keys is None else keys
        table = pd.DataFrame({f: _get_dataset(grp, f)[bounds] for f in keys})

    return table


def write_chunk(
    uris: Uris,
    chunk: pd.DataFrame,
    lower: int,
    keys: Iterable[str] | None = None,
) -> None:
    """Write pandas dataframe data at the provided position in the table."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = _get_group(h5_handle, path)
        bounds = slice(lower, lower + len(chunk))

        if keys is None:
            keys = [k for k in grp.keys() if k in chunk.columns]

        for key in keys:
            _get_dataset(grp, key)[bounds] = chunk[key]


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// TABLE OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def init_table(uris: Uris, size: int, col_mapping: dict[str, str]) -> None:
    """Initialize dataframe columns as 1D arrays."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        group = h5_handle.require_group(path)
        for name, dtype in col_mapping.items():
            group.require_dataset(
                name,
                shape=(size,),
                dtype=dtype,
                compression="gzip",
            )


def write_table(
    uris: Uris,
    iterator: PdChunks,
    keys: Iterable[str] | None = None,
) -> int:
    """Write chunk applied to iterator of chunks of the same table."""

    tab_size: int = 0
    for chunk in iterator:
        write_chunk(uris, chunk, tab_size, keys)
        tab_size += len(chunk)

    return tab_size


def save_table(uris: Uris, table: pd.DataFrame) -> None:
    """Convenience shorthand to initialize and place table at once."""

    init_table(uris, len(table), get_dataf_mapping(table))
    write_chunk(uris, table, 0)


def resize_table(uris: Uris, size: int) -> None:
    """Resize a table by cropping it at the provided index."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = _get_group(h5_handle, path)
        for key in grp.keys():
            _get_dataset(grp, key).resize((size,))


def get_table_size(uris: Uris) -> int:
    """Fetch table size."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        grp = _get_group(h5_handle, path)
        size = set(len(_get_dataset(grp, k)) for k in grp.keys())

        if len(size) != 1:
            raise ValueError("Inconsistent table size")

    return size.pop()


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// GROUP OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def require_group(uris: Uris, attrs: dict[str, Any] | None = None) -> None:
    """Requires a group and sets some attributes if non-existent."""

    attrs = attrs or {}
    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = h5_handle.require_group(path)

        for k, v in attrs.items():
            if k not in grp.attrs:
                grp.attrs[k] = v


def set_attrs(uris: Uris, attrs: dict[str, Any]) -> None:
    """Sets one or more group attributes (overwriting present)."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        group = _get_group(h5_handle, path)

        for k, v in attrs.items():
            group.attrs[k] = v


def get_attrs(uris: Uris) -> dict[str, Any]:
    """Fetch table attributes."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        group = _get_group(h5_handle, path)
        attrs = dict(group.attrs)

    return attrs


def del_keys(uris: Uris, keys: Iterable[str]) -> None:
    """Deletes one or more table keys."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        group = _get_group(h5_handle, path)

        for key in keys:
            del group[key]


def get_keys(uris: Uris) -> Iterable[str]:
    """Fetch table keys."""

    store, path = uris.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        group = _get_group(h5_handle, path)
        keys = list(group.keys())

    return keys
