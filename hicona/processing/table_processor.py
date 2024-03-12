"""Object to process a table according to the given operations."""

import logging
import time

from ..hicona_table import RawTable
from .spar_funs import sparsify_chunk
from ..utils.chunked_ops import get_node_stats
from ..utils.hdf5_ops import resize_table, write_table, write_chunk


__all__ = ["TableProcessor"]


class TableProcessor:
    """Process a table according to the given operations."""

    def __init__(self, table: RawTable, log_level: int = logging.INFO):
        self._table = table
        self._log_level = log_level

    def _apply_functions(self):
        """ "Apply all filtering and normalization functions to the table."""

        store, path = self._table.uris.hdf5_uris()
        for op in self._table.flow.get_partials():
            logging.info("Starting to apply: %s", op.func.__name__)

            tab_size = write_table(store, path, op(self._table))
            resize_table(store, path, tab_size)

            logging.info("Finished applying: %s", op.func.__name__)
            logging.info("Table size: %s", tab_size)

    def _spar_table(self):
        """Sparsify the chunks and save the results."""

        logging.info("Starting to sparsify the table.")

        node_stats = get_node_stats(self._table.chunks(), "norm")
        tab_size = 0

        # NOTE: implemented this way to simplify breaking into parallel later
        store, path = self._table.uris.hdf5_uris()
        for i, chunk in enumerate(self._table.chunks()):

            logging.info("Starting to sparsify chunk %s.", i)
            start = time.time()

            spar_chunk = sparsify_chunk(chunk, node_stats)
            write_chunk(store, path, spar_chunk, tab_size)

            end = time.time()
            logging.info("Finished sparsifying chunk %s.", i)
            logging.info("Took %s seconds.", end - start)

            tab_size += len(spar_chunk)

    def create_table(self):
        """Begin actual table processing."""

        logging.basicConfig(level=self._log_level)
        logging.info("Starting table creation.")

        start_time = time.time()

        self._apply_functions()
        self._spar_table()

        end_time = time.time()
        logging.info("Table creation took %s seconds.", end_time - start_time)

        # TODO: Return table object
