"""Default functions for pixel table sparsification."""

import time
from typing import TYPE_CHECKING

from hicona._core import sparsification
from hicona._ops import chunked
from hicona.preprocess._abcs import SparOperation

if TYPE_CHECKING:
    from hicona._core import Table
    from hicona._dtypes import PdChunks


__all__ = ["SparWeighted"]


class SparWeighted(SparOperation):
    """Apply node weight based sparsification to a table.

    Sparsify a table based on the node weights. The sparsification is based on
    the alpha values computed according to Serrano et al. 2009."""

    def __init__(
        self,
        *,
        apply_col: str,
        bonferroni: bool = True,
    ):
        self._apply_col = apply_col
        self._bonferroni = bonferroni

    def run(self, table: "Table") -> "PdChunks":

        node_stats = chunked.get_node_stats(table.chunks(), self._apply_col)
        node_stats.to_csv("node_stats.csv")

        # NOTE: implemented this way to simplify breaking into parallel later
        for i, chunk in enumerate(table.chunks()):

            print(f"Starting to sparsify chunk {i}.")
            start = time.time()

            spar_chunk = sparsification.sparsify_chunk(
                chunk, node_stats, self._bonferroni
            )

            end = time.time()
            print(f"Finished sparsifying chunk {i}.")
            print(f"Took {end - start} seconds.")

            yield spar_chunk
