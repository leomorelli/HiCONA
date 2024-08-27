"""Default functions for pixel table sparsification."""

import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

from hicona._core import sparsification
from hicona._ops import chunked, dataf, logging
from hicona.preprocess._abcs import SparOperation

if TYPE_CHECKING:
    from hicona._core import Table
    from hicona._dtypes import DfChunks


__all__ = ["SparWeighted", "SparLocalDegree"]


def _general_sparsify(table: "Table", mode: str, logger, **kwargs) -> "DfChunks":
    """General sparsification function for pixel table chunks."""

    # NOTE: implemented this way to simplify breaking into parallel later
    # TODO: replace logging with progress bar maybe (but how to know number of chunks?)

    for i, chunk in enumerate(table.chunks()):

        logger.info("Working on chunk %s.", i)
        spar_chunk = sparsification.sparsify_chunk(chunk, mode, **kwargs)

        yield spar_chunk


class SparWeighted(SparOperation):
    """Compute pixel table sparsification scores using edge weight.

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

    def process(self, table: "Table") -> "DfChunks":

        logger = logging.get_console_logger("sparsify")

        logger.info("Computing node stats.")
        node_stats = chunked.get_node_stats(table.chunks(), self._apply_col)

        chunks = _general_sparsify(
            table,
            "weighted",
            logger,
            counts_col=self._apply_col,
            stats=node_stats,
            bonferroni=self._bonferroni,
        )

        return chunks


class SparLocalDegree(SparOperation):
    """Compute pixel table sparsification scores using local degree.

    Compute the sparsification scores for a pixel table based on local node
    degree. The algorithm is mostly the same as described in `Hamann et al.
    2016`, though ties are allowed in the ranking (rather than arbitrarily
    broken). This makes the algorithm deterministic and reproducible.

    Parameters
    ----------
    node_chunk : int, optional
        The number of nodes to process at once. Default is 10_000.

    Notes
    -----
    Currently complexity does not scale linearly with the number of nodes,
    therefore it is difficult to estimate memory usage depending on
    chunk size. Heuristically, 10,000 nodes per chunk should run in 6 GB RAM,
    while 30,000 nodes per chunk should run in 16 GB RAM. This may vary
    depending on the network structure.

    """

    def __init__(self, *, node_chunk: int = 10_000):
        self._node_chunk = node_chunk
        self._tmp_dir: pathlib.Path | None = None

    def process(self, table: "Table") -> "DfChunks":

        logger = logging.get_console_logger("sparsify")

        self._tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="hicona-"))

        # Retrieve node degrees and ranking to pass to sparsification function
        # Breaks are used to split the nodes into chunks to avoid RAM overload
        # TODO: Remove drop when get_node_stats is modularized

        max_id = max(c.get_column("bin2_id").max() for c in table.chunks())  # type: ignore
        values = range(0, max_id + 1, self._node_chunk)
        breaks = [(i, i + self._node_chunk) for i in values]

        # bin_id, degree, (quant)
        logger.info("Computing node stats.")
        degrees = chunked.get_node_stats(table.chunks(), "count").drop("weight")
        ranking = chunked.get_degree_ranking(
            table.chunks(), degrees, breaks, self._tmp_dir
        )


        return chunks

    def cleanup(self) -> None:
        if isinstance(self._tmp_dir, pathlib.Path):
            shutil.rmtree(self._tmp_dir)
