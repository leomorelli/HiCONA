"""Placeholder

Currently only one norm
"""

from .norm_funs import no_norm
from .sparsification import sparsify_chunk
from ..chunked_ops import get_node_stats
from ..utils.hdf5_ops import init_table, resize_table, write_table


class TableProcessor:
    """Placeholder"""

    def __init__(self, hico_cool, scheduler):
        self._hico_cool = hico_cool
        self._scheduler = scheduler
        self._table = None

    def _filt_table(self, partials):
        """Apply filters to the table."""

        store_uri, pixels_uri = self._table.store, self._table.pixels_uri
        for part in partials:
            tab_size = write_table(store_uri, pixels_uri, part(table))
            resize_table(store_uri, pixels_uri, tab_size)

    def _norm_table(self, partials):
        """Apply normalizations to the table."""

        # If no normalization is provided, copy filt data to norm column.
        partials = partials if partials else [no_norm]

        store_uri, pixels_uri = self._table.store, self._table.pixels_uri
        for part in partials:
            write_table(store_uri, pixels_uri, part(table))

    def _sparsify(self):
        """Placeholder."""

        store_uri, pixels_uri = self._table.store, self._table.pixels_uri
        node_stats = get_node_stats(iterator, weight_col)

        # NOTE: implemented this way to simplify breaking into parallel later
        tab_size = 0
        for chunk in self._table.get_chuks():
            spar_chunk = sparsify_chunk(chunk)
            write_chunk(store_uri, pixels_uri, spar_chunk, tab_size)
            tab_size += len(spar_chunk)

    def start(self):
        """Begin actual table processing."""

        # Init table
        # Add table attributes?

        self._filt_table(self._scheduler._pre_filters.get_partials())
        self._norm_table(self._scheduler._norm_method.get_partials())
        self._filt_table(self._scheduler._post_filters.get_partials())
        self._sparsify()


'''
def _create_table(self, scheduler):
    """Create the table matching the given set of parameters."""

    # Check there is no table with all matching keywords

    # Initialize table
    table_path = self._next_table_path()
    table_cols = HICONA_SETTINGS.conventions.table_columns
    init_table(self.root, table_path, self.info["nnz"], table_cols)

    # Create _TableProcessor instance and run it
    processor = TableProcessor(table)
    processor.start()


self._require_tables_root()
def _require_tables_root(self):
    """Initialize the tables root if it does not exist already."""

    require_group(self.store, self._tables_root, {"serial": 0})
'''
