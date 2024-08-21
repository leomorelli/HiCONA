"""Placeholder"""

import re

import polars as pl

from hicona._resources import regexes


class GenomicRegion:
    """Class to handle genomic regions conversion."""

    def __init__(self, genomic_str: str) -> None:

        self._chrom, self._start, self._end = self._parse_str(genomic_str)

    @staticmethod
    def _parse_str(region: str) -> tuple[str, int | None, int | None]:
        """Parse region string to obtain standard format parts."""

        if match := re.match(regexes.POS_LIKE_STR, region):
            return (match[1], int(match[2]) - 1, int(match[3]))
        if match := re.match(regexes.BED_LIKE_STR, region):
            return (match[1], int(match[2]), int(match[3]))
        if match := re.match(regexes.CHR_LIKE_STR, region):
            return (match[1], None, None)

        raise ValueError(f"Invalid region string: {region}")

    def to_bed_str(self) -> str:
        """Return region in BED format."""

        if self._start is None or self._end is None:
            return self._chrom
        return f"{self._chrom}\t{self._start}\t{self._end}"

    def to_pos_str(self) -> str:
        """Return region in POS format."""

        if self._start is None or self._end is None:
            return self._chrom
        return f"{self._chrom}:{self._start + 1}-{self._end}"

    def to_query(self, both: bool = False) -> pl.Expr:
        """Return region in query format."""

        def get_bin_exp(
            bin_id: int, chrom: str, start: int | None, end: int | None
        ) -> pl.Expr:

            expr = pl.col(f"chrom{bin_id}") == chrom
            if start is not None:
                expr &= pl.col(f"start{bin_id}") >= start
            if end is not None:
                expr &= pl.col(f"end{bin_id}") <= end

            return expr

        bin1_exp = get_bin_exp(1, self._chrom, self._start, self._end)
        bin2_exp = get_bin_exp(2, self._chrom, self._start, self._end)

        return bin1_exp & bin2_exp if both else bin1_exp | bin2_exp

    def snap_to_bin(self, bin_size: int) -> None:
        """Snap start and end to bin boundaries."""

        if self._start is not None:
            self._start = (self._start // bin_size) * bin_size
        if self._end is not None:
            self._end = ((self._end + bin_size - 1) // bin_size) * bin_size
