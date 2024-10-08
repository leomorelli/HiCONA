"""Temporary folder torn down when the instance is deleted."""

import pathlib
import tempfile


class TmpStorage:
    """Class to handle storage in temporary directories."""

    def __init__(self, prefix: str = ""):
        self._tmp_store = tempfile.TemporaryDirectory(prefix=prefix)

    # TODO: Should not be necessary to implement __del__ method, but check
    # def __del__(self):
    #     if hasattr(self, "_tmp_store"):
    #         if self._tmp_store is not None:
    #             self._tmp_store.cleanup()

    @property
    def tmp_store(self) -> pathlib.Path:
        """Return the temporary folder path."""
        return pathlib.Path(self._tmp_store.name)
