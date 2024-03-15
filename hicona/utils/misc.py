"""Miscellaneous utility functions for the hicona package."""

from itertools import chain, combinations
import re

from .regexes import BED_LIKE_STR, CHR_LIKE_STR, POS_LIKE_STR


def annotation_combinations(iterable, k_vals=(1, 2)):
    """Return iterable of all combinations for all k_vals"""

    comb = [combinations(iterable, k) for k in k_vals]
    return list(chain.from_iterable(comb))


class GenomicRegion:
    """Class to handle genomic regions conversion."""

    def __init__(self, genomic_str: str) -> None:

        self._chrom, self._start, self._end = self._parse_str(genomic_str)

    def _parse_str(self, region: str) -> tuple[str, int | None, int | None]:
        """Parse region string to obtain standard format parts."""

        if match := re.match(POS_LIKE_STR, region):
            return (match[1], int(match[2]) - 1, int(match[3]))
        if match := re.match(BED_LIKE_STR, region):
            return (match[1], int(match[2]), int(match[3]))
        if match := re.match(CHR_LIKE_STR, region):
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

    def to_query(self, both: bool = False) -> str:
        """Return region in query format."""

        operator = "and" if both else "or"

        temp = f"chrom[N] == '{self._chrom}'"
        if self._start or self._end:
            temp += f" and start[N] >= {self._start} and end[N] <= {self._end}"

        temps = [temp.replace("[N]", str(num)) for num in (1, 2)]
        return f"({temps[0]}) {operator} ({temps[1]})"

    def snap_to_bin(self, bin_size: int) -> None:
        """Snap start and end to bin boundaries."""

        if self._start is not None:
            self._start = (self._start // bin_size) * bin_size
        if self._end is not None:
            self._end = ((self._end + bin_size - 1) // bin_size) * bin_size
