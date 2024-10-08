"""Placeholder"""

import re

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

    @property
    def chrom(self) -> str:
        """Region chromsome."""
        return self._chrom

    @property
    def start(self) -> int | None:
        """Region start. None if full chromosome."""
        return self._start

    @property
    def end(self) -> int | None:
        """Region end. None if full chromosome."""
        return self._end
