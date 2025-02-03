import contextlib
import os
import tempfile
import shutil


# Copied from https://github.com/open2c/cooler/blob/master/tests/_common.py
@contextlib.contextmanager
def isolated_filesystem():
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    t = tempfile.mkdtemp()
    os.chdir(t)
    try:
        yield t
    finally:
        os.chdir(cwd)
        try:
            shutil.rmtree(t)
        except OSError:
            pass
