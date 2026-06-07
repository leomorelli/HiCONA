from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("hicona")
except PackageNotFoundError:
    __version__ = "unknown"
