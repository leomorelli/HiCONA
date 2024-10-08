"""Placeholder"""

from typing import Any, Iterable

import h5py
import pandas as pd
import polars as pl

from hicona._ops import uris
from hicona._dtypes import DfChunks


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
    uris_path: uris.Uris,
    bounds: slice,
    keys: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Retrieve a subset of the table as a dataframe."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        grp = _get_group(h5_handle, path)
        keys = list(grp.keys()) if keys is None else keys
        table = pd.DataFrame({f: _get_dataset(grp, f)[bounds] for f in keys})

    return pl.DataFrame(table)


def iter_chunks(
    uris_path: uris.Uris,
    chunk_size: int,
    keys: Iterable[str] | None = None,
) -> DfChunks:
    """Iterate over table chunks."""

    tab_size = get_table_size(uris_path)
    for lower in range(0, tab_size, chunk_size):
        yield fetch_chunk(uris_path, slice(lower, lower + chunk_size), keys)


def write_chunk(
    uris_path: uris.Uris,
    chunk: pl.DataFrame,
    lower: int,
    keys: Iterable[str] | None = None,
) -> None:
    """Write pandas dataframe data at the provided position in the table."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = _get_group(h5_handle, path)
        bounds = slice(lower, lower + len(chunk))

        if keys is None:
            keys = [k for k in grp.keys() if k in chunk.columns]

        for key in keys:
            _get_dataset(grp, key)[bounds] = chunk[key]


# def copy_table(source_uris: uris.Uris, dest_uris: uris.Uris) -> None:
#     """Copy a group containing multiple datasets to a new location."""

#     store, source = source_uris.hdf5_uris()
#     with h5py.File(store, mode="r+") as h5_handle:
#         h5_handle.copy(source, dest_uris.path)


# def iter_table(uris_path: uris.Uris) -> DfChunks:
#     """Iterate over table chunks."""

#     store, path = uris_path.hdf5_uris()
#     with h5py.File(store, mode="r") as h5_handle:
#         grp = _get_group(h5_handle, path)

#         tables = {k: _get_dataset(grp, k).iter_chunks() for k in grp.keys()}

#         yield pl.DataFrame({k: next(v)[:] for k, v in tables.items()})


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// TABLE OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def init_table(uris_path: uris.Uris, size: int, col_mapping: dict) -> None:
    """Initialize dataframe columns as 1D arrays."""

    store, path = uris_path.hdf5_uris()
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
    uris_path: uris.Uris,
    iterator: DfChunks,
    keys: Iterable[str] | None = None,
) -> int:
    """Write chunk applied to iterator of chunks of the same table."""

    tab_size: int = 0
    for chunk in iterator:
        write_chunk(uris_path, chunk, tab_size, keys)
        tab_size += len(chunk)

    return tab_size


def save_table(uris_path: uris.Uris, table: pl.DataFrame) -> None:
    """Convenience shorthand to initialize and place table at once."""

    mapping = {
        k: (v if v != "object" else f"|S{table[k].str.len_chars().max()}")
        for k, v in dict(table.to_pandas().dtypes).items()
    }
    init_table(uris_path, len(table), mapping)
    write_chunk(uris_path, table, 0)


def resize_table(uris_path: uris.Uris, size: int) -> None:
    """Resize a table by cropping it at the provided index."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = _get_group(h5_handle, path)
        for key in grp.keys():
            _get_dataset(grp, key).resize((size,))


def get_table_size(uris_path: uris.Uris) -> int:
    """Fetch table size."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        grp = _get_group(h5_handle, path)
        size = set(len(_get_dataset(grp, k)) for k in grp.keys())

        if len(size) != 1:
            raise ValueError("Inconsistent table size")

    return size.pop()


# ////////////////////////////////////////////////////////////////////////////
# ///////////////////////////// GROUP OPERATIONS /////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


def require_group(uris_path: uris.Uris, attrs: dict[str, Any] | None = None) -> None:
    """Requires a group and sets some attributes if non-existent."""

    attrs = attrs or {}
    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        grp = h5_handle.require_group(path)

        for k, v in attrs.items():
            if k not in grp.attrs:
                grp.attrs[k] = v


def set_attrs(uris_path: uris.Uris, attrs: dict[str, Any]) -> None:
    """Sets one or more group attributes (overwriting present)."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        group = _get_group(h5_handle, path)

        for k, v in attrs.items():
            group.attrs[k] = v


def get_attrs(uris_path: uris.Uris) -> dict[str, Any]:
    """Fetch table attributes."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        group = _get_group(h5_handle, path)
        attrs = dict(group.attrs)

    return attrs


def del_keys(uris_path: uris.Uris, keys: Iterable[str]) -> None:
    """Deletes one or more table keys."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r+") as h5_handle:
        group = _get_group(h5_handle, path)

        for key in keys:
            del group[key]


def get_keys(uris_path: uris.Uris) -> Iterable[str]:
    """Fetch table keys."""

    store, path = uris_path.hdf5_uris()
    with h5py.File(store, mode="r") as h5_handle:
        group = _get_group(h5_handle, path)
        keys = list(group.keys())

    return keys
