"""Everything that is assumed to be constant in the code."""

# Uri where the tables are stored in the hicona cooler.
TABLES_ROOT = "hicona_tables"

# Columns and dtypes of hicona tables.
TABLE_COLUMNS = {
    "bin1_id": "i8",
    "bin2_id": "i8",
    "count": "i4",
    "norm": "f8",
    "alpha_min": "f8",
    "alpha_max": "f8",
}

# Columns that are always present in the bins table.
BASE_BINS_COLS = (str("chrom"), str("start"), str("end"))
