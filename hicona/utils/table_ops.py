"""Placeholder"""

import numpy as np
import pandas as pd

from ..settings import HICONA_SETTINGS

# TODO: somehow add category modality


def from_df_to_sarrays(data: pd.DataFrame):
    """Return each column of a dataframe as a numpy structured array.

    Transform the columns of a dataframe into numpy structured array and
    define the numpy datatype most appropriate for storage in HDF5 (especially
    minimum required string fixed length for categorical annotations).
    """

    # TODO: Currently only discriminating string/non string, improve

    dtypes = data.dtypes
    for col_name, dtype in zip(data, dtypes):
        if dtype == "object":
            str_len = int(data[col_name].str.len().max())
            str_len = str_len if str_len > 3 else 3
            dtype = np.dtype(f"S{str_len}")

        new_col = np.zeros(len(data), dtype)
        new_col[:] = data[col_name].values

        yield (col_name, new_col, dtype)


def get_dataf_mapping(dataf: pd.DataFrame) -> dict:
    """Placeholder"""

    # Fetch dtype for each column name
    mapping = dict(zip(dataf.columns, dataf.dtypes))

    # If dtype is object, convert to |SX where X is the max str length.
    # This is because hdf5 does not support object or generic string types.
    for k, v in mapping.items():
        if v == "object":
            conv_col = dataf[k].convert_dtypes()
            if conv_col.dtype == "string[python]":
                conv_col.fillna("nan", inplace=True)
                max_str_len = conv_col.map(len).max()
                mapping[k] = f"|S{max_str_len}"

            else:  # Cannot be converted to string
                raise TypeError("Object type annotations are not supported.")

    return mapping


def pd_to_h5_dtype(pd_dtype: str):
    """Placeholder"""

    h5_dtype = pd_dtype
    if pd_dtype == "object":
        h5_dtype = np.dtype("S10")

    return h5_dtype


def pd2gt_dtype(pd_dtype: str):
    """Convert pandas-like datatypes to graph-tools datatypes"""
    return HICONA_SETTINGS.conventions.dtype_conversion[pd_dtype]
