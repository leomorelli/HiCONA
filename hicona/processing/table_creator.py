"""Placeholder

Currently only one norm
"""

from ..hicona_table import TableHandler
from .norm_funs import no_norm
from .sparsification import sparsify_chunk
from ..utils.chunked_ops import get_node_stats
from ..utils.hdf5_ops import (
    init_table,
    resize_table,
    require_group,
    write_table,
    get_attrs,
    set_attrs,
)
from ..settings import HICONA_SETTINGS


__all__ = ["TableCreator"]

# TODO: Maybe do not hardcode serial column name


class TableCreator(TableHandler):
    """Placeholder"""

    def _init_table(self, table_size):
        """Create the empty structure on disc and return table object."""

        def require_root(store, path):
            """Initialize the tables root if it does not exist already."""
            require_group(store, path, {"serial": 0})

        def next_tab_path(store, path):
            """Get the next available table path."""
            serial = get_attrs(store, path)["serial"]
            set_attrs(store, path, {"serial": serial + 1})
            return f"table_{str(serial).zfill(6)}"

        # Require hicona tables root
        require_root(*self._uris.get_hdf5_uris())

        # Get path for the nex table (next available serial number)
        table_path = next_tab_path(*self._uris.get_hdf5_uris())
        self._uris.add_path_part(table_path)

        # TODO: Check there is no table with all matching keywords
        # TODO: Add table attributes

        # Initialize table
        cols = HICONA_SETTINGS.conventions.table_columns
        init_table(*self._uris.get_hdf5_uris(), table_size, cols)

    def _filt_table(self, partials):
        """Apply filters to the table."""

        store_uri, pixels_uri = self._uris.get_hdf5_uris()
        for part in partials:
            tab_size = write_table(store_uri, pixels_uri, part(self))
            resize_table(store_uri, pixels_uri, tab_size)

    def _norm_table(self, partials):
        """Apply normalizations to the table."""

        # If no normalization is provided, copy filt data to norm column.
        partials = partials if partials else [no_norm]

        store_uri, pixels_uri = self._uris.get_hdf5_uris()
        for part in partials:
            write_table(store_uri, pixels_uri, part(self))

    def _spar_table(self):
        """Sparsify the chunks and save the results."""

        store_uri, pixels_uri = self._uris.get_hdf5_uris()
        node_stats = get_node_stats(iterator, weight_col)

        # NOTE: implemented this way to simplify breaking into parallel later
        tab_size = 0
        for chunk in self._table.chunks():
            spar_chunk = sparsify_chunk(chunk, node_stats)
            write_chunk(store_uri, pixels_uri, spar_chunk, tab_size)
            tab_size += len(spar_chunk)

    def create_table(self, table_size):
        """Begin actual table processing."""

        self._init_table(table_size)

        self._filt_table(self._process_info._pre_filters.get_partials())
        self._norm_table(self._process_info._norm_method.get_partials())
        self._filt_table(self._process_info._post_filters.get_partials())

        self._spar_table()

        # TODO: Return table object
