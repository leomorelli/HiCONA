"""Placeholder"""


class Uris:
    """Placeholder"""

    def __init__(self, store: str, root: str, path: str | None = None):
        self._store = store
        self._root = root if root != "/" else ""
        self._path = path or ""

    def hdf5_uris(self):
        """Return store and path parts up to the last specified."""

        return (self._store, "/".join([self._root, self._path]))

    def cooler_uri(self):
        """Return a uri to use to create a cooler handle."""

        return "::".join([self._store, self._root])
