"""Placeholder"""


class Uris:
    """Placeholder"""

    def __init__(self, store: str, root: str = None, path: str = None):
        self._store = store
        self._root = root if root != "/" else ""
        self._path = path or ""

    def add_path_part(self, part: str):
        """Add path part as the last one in the parts."""

        self._path = "/".join([self._path, part])

    def get_hdf5_uris(self):
        """Return store and path parts up to the last specified."""

        return (self._store, "/".join([self._root, self._path]))

    def get_cooler_uri(self):
        """Return a uri to use to create a cooler handle."""

        return "::".join([self._store, self._root])
