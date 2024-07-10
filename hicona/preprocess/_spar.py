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
    """Compute pixel table sparsification scores in a weighted matter.

    Compute the sparsification scores for a pixel table based on node weights.
    The alpha values are computed according to `Serrano et al. 2009`.

    Parameters
    ----------
    apply_col : str
        The column name of the node weights.
    bonferroni : bool, optional
        Whether to apply Bonferroni correction to the alpha values (multiply
        by the number of neighbors of the node). Default is True.
    """

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
