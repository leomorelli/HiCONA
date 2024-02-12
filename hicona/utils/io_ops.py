"""Placeholder"""

import importlib.resources as imp_res
import os
import json

from .. import resources


def read_resource(res_path):
    """Load a static resource."""

    folder = imp_res.files(resources)
    full_path = os.path.join(folder, res_path)

    with open(full_path, "r", encoding="utf-8") as handle:
        ext = os.path.splitext(full_path)[1]

        match ext:
            case ".json":
                out = json.load(handle)
            case _:
                raise ValueError("Currently unknown extension.")

    return out


def write_resource(res_path, content):
    """Write to a static resource."""

    folder = imp_res.files(resources)
    full_path = path.join(folder, res_path)

    with open(full_path, "w", encoding="utf-8") as handle:
        ext = os.path.splitext(full_path)[1]

        match ext:
            case ".json":
                json.dump(settings_json, handle, indent=4)
            case _:
                raise ValueError("Currently unknown extension.")
