"""Module containing function to simplify I/O operations with resources."""

import importlib.resources as imp_res
import json
import os
import pathlib
from typing import Any

from hicona import _resources


def _get_path(res_path: str, is_static: bool) -> str:
    """Modify path if it belongs to a static resource, else return as is."""

    if is_static:
        folder: pathlib.PosixPath = imp_res.files(_resources)  # type: ignore
        return os.path.join(folder, res_path)

    return res_path


def read_resource(res_path: str, is_static: bool = False) -> Any:
    """Load a resource from file."""

    full_path = _get_path(res_path, is_static)
    with open(full_path, "r", encoding="utf-8") as handle:
        ext = os.path.splitext(full_path)[1]

        match ext:
            case ".json":
                out = json.load(handle)
            case _:
                raise ValueError("Currently unknown extension.")

    return out


def write_resource(res_path: str, content: Any, is_static: bool = False) -> None:
    """Write a resource to file."""

    full_path = _get_path(res_path, is_static)
    with open(full_path, "w", encoding="utf-8") as handle:
        ext = os.path.splitext(full_path)[1]

        match ext:
            case ".json":
                json.dump(content, handle, indent=4)
            case _:
                raise ValueError("Currently unknown extension.")
