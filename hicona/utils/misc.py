"""Miscellaneous utility functions for the hicona package."""

from itertools import chain, combinations


def annotation_combinations(iterable, k_vals=(1, 2)):
    """Return iterable of all combinations for all k_vals"""

    comb = [combinations(iterable, k) for k in k_vals]
    return list(chain.from_iterable(comb))


def build_query(region: str, both: bool) -> str:
    """Placeholder"""

    # TODO: Somewhere add stringent checks for region format
    #       (e.g. "chrN:NNNN-NNNN" or "chrN" or "organism")
    # TODO: Snap to bin boundaries

    # TODO: Refactor this mess

    parts = region.split(" ")
    operator = "and" if both else "or"

    if len(parts) in (1, 3):
        temp = "chrom{num} == '{region}'"
    else:
        raise ValueError(f"Invalid region string: {region}")

    if len(parts) == 3:
        temp += " and start{num} >= {start} and end{num} <= {end}"

    queries = []
    for num in range(1, 3):
        if len(parts) == 1:
            query = temp.format(num=num, region=parts[0])
        else:
            query = temp.format(num=num, region=parts[0], start=parts[1], end=parts[2])
        queries.append(query)

    return f"({queries[0]}) {operator} ({queries[1]})"
