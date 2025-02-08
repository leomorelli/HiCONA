"""Temporary parquet-like folder torn down when the instance is deleted."""

import os
import shutil
import tempfile
from typing import Any

import polars as pl

from .df_dtypes import PlChunks


class TmpParquet:
    """Temporary parquet-like folder stored in the /tmp/ dir."""

    def __init__(self):
        self._tmp_ref = tempfile.TemporaryDirectory(prefix="hicona_parquet")
        self._path = os.path.abspath(self._tmp_ref.name)

    @property
    def path(self) -> str:
        """Full path to the folder."""
        return self._path

    def put(self, chunks: PlChunks):
        """Save chunks to the folder."""

        for i, chunk in enumerate(chunks):
            chunk_name: str = f"chunk_{str(i).zfill(4)}.parquet"
            chunk.write_parquet(os.path.join(self._path, chunk_name), statistics=False)

    def get(self) -> PlChunks:
        """Fetch iterable of chunks from the folder."""

        files = os.listdir(self._path)
        files.sort()

        for file in files:
            yield pl.read_parquet(os.path.join(self._path, file))

    def peek(self) -> dict[str, Any]:
        """Get first row of the table as a dictionary."""

        row_dict: dict[str, Any] | None = None
        for chunk in self.get():
            row_dict = chunk.row(0, named=True)
            break

        assert row_dict is not None, "Table is empty."
        return row_dict

    def save(self, path: str):
        """Save the temporary folder to a stable directory."""
        shutil.copytree(self._path, path)

    @classmethod
    def load(cls, path: str) -> "TmpParquet":
        """Create a temporary instance from a previously saved one."""

        instance = cls()
        shutil.copytree(path, instance.path, dirs_exist_ok=True)
        return instance
