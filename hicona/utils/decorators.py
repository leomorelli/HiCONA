"""Placeholder"""

import time
import functools

from decorator import decorate

from ..settings import HICONA_SETTINGS


def console_log(func):
    """A simple decorator to log information to the console."""
    # TODO: Make decorator toggleable

    def console_log_wrapper(*args, **kwargs):
        print(f"Starting to run: {func}")
        start_time = time.time()
        fun_return = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time-start_time}s")
        print("-" * 78)
        return fun_return

    return console_log_wrapper


def wait_hdf5_lock(func):
    """Placeholder"""

    def _wait_hdf5_lock(func, *args, **kwargs):
        while True:
            try:
                fun_return = func(*args, **kwargs)
                break
            except BlockingIOError:
                time.sleep(HICONA_SETTINGS.parameters.lock_delay)

        return fun_return

    return decorate(func, _wait_hdf5_lock)


def integration_cache(func):
    """Decorator to memoize alpha value integrals."""
    # TODO: make decorator toggleable
    int_cache = {}

    @functools.wraps(func)
    def integral_wrapper(*args, **kwargs):
        int_key = str(args) + str(kwargs)
        if int_key not in int_cache:
            int_cache[int_key] = func(*args, **kwargs)
        return int_cache[int_key]

    return integral_wrapper
