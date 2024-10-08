"""Subsampling of cooler files to emulate lower coverage."""

from typing import Iterator

import cooler
import numpy as np
import polars as pl
import pandas as pd


def subsample(
    in_file: str,
    out_file: str,
    target: float | str,
    chunk_size: int,
    rng_seed: int = 94206,
) -> None:
    """Uniformly subsample a cooler file.

    Subsample a cooler file to create a new one with lower coverage.
    In order to do so, the reads from all pixels are pooled together
    and randomly samples using a uniform distribution. The amount of
    reads to be kept is determined by the target parameter, which can
    be either a float between 0 and 1 (fraction of reads from the original
    file to be kept) or string (path to another cooler file to match the
    coverage with).

    Note
    ----
    This function seems, and should, be reproducible, though the parallelism
    in polars might affect the rng, so some further testing is needed.

    Parameters
    ----------
    in_file : str
        Path to the input cooler file.
    out_file : str
        Path to the output cooler file.
    target : float or str
        Fraction of reads to be kept or path to another cooler file.
    chunk_size : int
        Number of pixels to be read and process at once.
    rng_seed : int, optional
        Seed for the random number generator. Default is 94206.

    """

    def sampled_pixels(
        handle: cooler.Cooler,
        ratio: float,
        rng: np.random.Generator,
    ) -> Iterator[pd.DataFrame]:
        """Yield sampled pixels from a cooler file."""

        num_pix = handle.info["nnz"]
        for lower in range(0, num_pix, chunk_size):
            upper = min(lower + chunk_size, num_pix)
            chunk = pl.from_pandas(handle.pixels()[lower:upper])  # type: ignore

            chunk: pl.DataFrame = chunk.with_columns(
                pl.col("count").map_elements(lambda x: rng.binomial(x, ratio))
            )

            yield chunk.filter(pl.col("count") > 0).to_pandas()

    in_handle = cooler.Cooler(in_file)

    target_ratio: float = (
        cooler.Cooler(target).info["sum"] / in_handle.info["sum"]
        if isinstance(target, str)
        else target
    )
    assert isinstance(target_ratio, float) and 0 < target_ratio < 1

    rng = np.random.default_rng(rng_seed)
    pixels = sampled_pixels(in_handle, target_ratio, rng)
    bins = in_handle.bins()[:]  # type: ignore

    cooler.create_cooler(out_file, bins, pixels)
