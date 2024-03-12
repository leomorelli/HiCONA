"""Placeholder"""

from typing import Iterable

from .uris import Uris


class Intervals:
    """Placeholder"""

    def __init__(self, uris: Uris, intervals: Iterable[str]):

        self._uris = uris

        self._uscs: Iterable[str] = self._sanitize_ucsc(intervals)
        self._bins: Iterable[tuple[int, int]] = []
        self._pixels: Iterable[tuple[int, int]] = []

    @property
    def ucsc(self) -> Iterable[str]:
        """Return UCSC-style intervals."""
        return self._uscs

    @property
    def bins(self) -> Iterable[tuple[int, int]]:
        """Return intervals of bins ids."""
        return self._bins

    def _sanitize_ucsc(self, intervals: Iterable[str]) -> Iterable[str]:
        """Sanitize list of UCSC-style intervals."""

    def _ucsc_to_bins(self) -> Iterable[tuple[int, int]]:
        """Convert UCSC coordinates to intervals of bins ids."""


def parse_regions(regions, chroms):
    """Convert chromosome selection from regex/default str to iterable."""

    def fix_species_selection(selection):
        """Convert species selection string into list of chromosomes."""
        # "1-22,X,Y" -> ["chr1", "chr2", ..., "chr22", "chrX", "chrY"]

        out = []
        selection = selection.split(",")
        for s in selection:
            is_interval = len(s.split("-")) > 1
            if is_interval:
                lower, upper = s.split("-")
                for v in range(int(lower), int(upper) + 1):
                    out.append(f"chr{v}")
            else:
                out.append(f"chr{s}")

        return out

    intervals = []

    if isinstance(regions, str):
        regions = regions.strip()

        # "chrN:NNNN-NNNN" -> no formatting needed, return it
        if match := re.search(HICONA_SETTINGS.regexes.region, regions):
            intervals.append(regions)

        # "chrN" -> "chrN:NNNN-NNNN"
        elif match := re.search(HICONA_SETTINGS.regexes.chromosome, regions):
            c, s, e = chroms.query(f"chrom == '{match.group(0)}'").values[0].T
            intervals.append(f"{c}:{s}-{e}")

        # "organism" -> ["chrN:NNNN-NNNN", ...]
        elif match := HICONA_SETTINGS.conventions.chrom_lists.get(regions):
            for r in fix_species_selection(match):
                intervals.extend(parse_regions(r, chroms))

        # Incompatible string
        else:
            raise ValueError(f"{regions} is not a recognized genomic region.")

    # Assume it is an iterable of compatible objects
    else:
        for r in regions:
            intervals.extend(parse_regions(r, chroms))

    return intervals
