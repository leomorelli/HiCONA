"""Placeholder"""

from ..utils.hdf5_ops import resize_table, write_chunk


class _TableProcessor:
    """Placeholder"""

    def __init__(self, table, norm_method, pre_filters, post_filters):
        self._table = table
        self._norm_method = norm_method
        self._pre_filters = pre_filters
        self._post_filters = post_filters

    def _filter_table(self, pre_norm=True):
        """Apply filters to a table."""

        def apply_filter(table, filter_fun, **kwargs):
            """Apply a filtering function to a table."""

            tab_size = 0
            for chunk in filter_fun(table, **kwargs):
                write_chunk(table.store, table.pixels_uri, chunk, tab_size)
                tab_size += len(chunk)
            resize_table(table.store, table.pixels_uri, tab_size)

        filters = self._pre_filters if pre_norm else self._post_filters
        for filter_fun, fun_kwargs in filters:
            apply_filter(self._table, filter_fun, fun_kwargs)

    def _normalize_table(self):
        """Placeholder"""

        # Select appropriate normalization function
        norm_fun = None
        match self._norm_method:
            case "hicona":
                norm_fun = hicona_norm
            case "ice":
                norm_fun = ice_norm

        # Do the actual normalization

    def _sparsify(self):
        pass

    def start(self):
        """Begin actual table processing."""

        self._filter_table(pre_norm=True)
        self._normalize()
        self._filter_table(pre_norm=False)
        self._sparsify()
