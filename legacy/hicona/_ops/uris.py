"""Module to handle uri creation and manipulation."""


class Uris:
    """Class to handle uri creation and manipulation.

    Used to simplify the creation of uri strings from path parts and to easily
    pass the correct uri format string depending on the backend being used.

    Parameters
    ----------
    store : str
        The string specifying the system path of the cooler file.
    root : str
        The string used to specify the group containing the cooler when
        working with multi-resolutions coolers, else simply "/".
    path : str, optional
        The string specifying the part of the cooler to consider, usually
        the whole cooler or a pixel table). Default is None.
    """

    def __init__(self, store: str, root: str, path: str | None = None):
        self._store = store
        self._root = root if root != "/" else ""
        self._path = path or ""

    @property
    def store(self) -> str:
        """Return the store part of the uri."""
        return self._store

    @property
    def root(self) -> str:
        """Return the root part of the uri."""
        return self._root

    @property
    def path(self) -> str:
        """Return the path part of the uri."""
        return self._path

    def hdf5_uris(self) -> tuple[str, str]:
        """Return store and path parts up to the last specified."""
        return (self._store, "/".join([self._root, self._path]))

    def cooler_uri(self) -> str:
        """Return a uri to use to create a cooler handle."""
        return "::".join([self._store, self._root])

    def add_path(self, path: str) -> "Uris":
        """Add a part to the path uri and return a new instance."""
        return Uris(self._store, self._root, "/".join([self._path, path]))

    def is_valid(self) -> bool:
        """Check if the uri is valid."""
        return True  # TODO: Not actually checking for now, implement later
