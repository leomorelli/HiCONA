"""Placeholder."""

from typing import TYPE_CHECKING

import pandas as pd
import cooler
import polars as pl

from hicona.analysis import _plotting
from hicona._core import annotate

if TYPE_CHECKING:
    from hicona._table import HiconaTable


class TableComparison:
    """Placeholder"""

    def __init__(self, table_a: "HiconaTable", table_b: "HiconaTable"):
        self._table_a = table_a
        self._table_b = table_b
        self._dataframe = table_a.dataframe().join(
            table_a.dataframe(),
            on=["bin1_id", "bin2_id"],
            how="outer",
            suffix="_b",
        )

        mapping = {
            c: f"{c}_a"
            for c in self._dataframe.columns
            if ((not c.endswith("_b")) and (c != "bin1_id") and (c != "bin2_id"))
        }
        self._dataframe = self._dataframe.rename(mapping)

        # TODO: add some form of check that the tables are comparable

    def get_dataframe(self, choice: str = "all") -> pl.DataFrame:
        """Placeholder"""

        df = self._dataframe
        if choice == "all":
            pass
        elif choice == "a":
            df = df.drop_nulls(subset=["count_a"])
        elif choice == "b":
            df = df.drop_nulls(subset=["count_b"])
        elif choice == "both":
            df = df.drop_nulls(subset=["count_a", "count_b"])
        else:
            raise ValueError("Invalid choice")

        return df

    def plot(
        self,
        column: str,
        names: tuple[str, str] = ("A", "B"),
        img_path: str | None = None,
        show: bool = False,
    ):
        """Placeholder"""

        if column not in ["alpha_min", "alpha_max"]:
            raise ValueError("Currently only alpha_min and alpha_max are supported")

        points = self.get_dataframe("both")[[f"{column}_a", f"{column}_b"]]
        points = points.rename({f"{column}_a": "x", f"{column}_b": "y"})

        handle = cooler.Cooler(self._table_a.uris.cooler_uri())
        bins: pl.DataFrame = pl.from_pandas(handle.bins()[:])  # type: ignore
        highlight = annotate(self.get_dataframe("both"), bins)

        # part_a = highlight.query("HMM_annot1 == 'Prom' and HMM_annot2 == 'Enh'")
        # part_b = highlight.query("HMM_annot1 == 'Enh' and HMM_annot2 == 'Prom'")
        # highlight = pd.concat([part_a, part_b])[[f"{column}_a", f"{column}_b"]]
        # highlight = highlight.rename(columns={f"{column}_a": "x", f"{column}_b": "y"})

        # _plotting.plot_comparison(
        #     self._table_a.get_distribution(column),
        #     self._table_b.get_distribution(column),
        #     points,
        #     highlight,
        #     names,
        #     img_path,
        #     show,
        # )
