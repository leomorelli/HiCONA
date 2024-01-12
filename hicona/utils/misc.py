"""Miscellaneous utility functions for the hicona package."""

from itertools import chain, combinations
import re

from ..settings import HICONA_SETTINGS


def annotation_combinations(iterable, k_vals=(1, 2)):
    """Return iterable of all combinations for all k_vals"""

    comb = [combinations(iterable, k) for k in k_vals]
    return list(chain.from_iterable(comb))


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
