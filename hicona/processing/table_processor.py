"""Object to process a table according to the given operations."""

from ..hicona_table import RawTable, HiconaTable
from .sparsification import sparsify_chunk
from ..utils.chunked_ops import get_node_stats
from ..utils.hdf5_ops import resize_table, write_table, write_chunk


__all__ = ["TableProcessor"]


class TableProcessor:
    """Process a table according to the given operations."""

    def __init__(self, table: RawTable):
        self._table = table

    def _apply_functions(self):
        """ "Apply all filtering and normalization functions to the table."""

        store, path = self._table.uris.hdf5_uris()
        for op in self._table.process_info.get_partials():
            tab_size = write_table(store, path, op(self._table))
            resize_table(store, path, tab_size)

    def _spar_table(self):
        """Sparsify the chunks and save the results."""

        node_stats = get_node_stats(self._table.chunks(), "norm")
        tab_size = 0

        # NOTE: implemented this way to simplify breaking into parallel later
        store, path = self._table.uris.hdf5_uris()
        for chunk in self._table.chunks():
            spar_chunk = sparsify_chunk(chunk, node_stats)
            write_chunk(store, path, spar_chunk, tab_size)
            tab_size += len(spar_chunk)

    def create_table(self) -> HiconaTable:
        """Begin actual table processing."""

        self._apply_functions()
        self._spar_table()

        # TODO: Return table object
