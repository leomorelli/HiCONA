"""Module with custom data types for the hicona package type annotation."""

from typing import Generator

import pandas as pd


# Type alias for a generator that yields pandas DataFrames.
PdChunks = Generator[pd.DataFrame, None, None]
