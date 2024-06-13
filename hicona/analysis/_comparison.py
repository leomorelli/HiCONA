"""Placeholder."""

from typing import TYPE_CHECKING

import pandas as pd
import cooler

from hicona.analysis import _plotting

if TYPE_CHECKING:
    from hicona._table import HiconaTable


class TableComparison:
    """Placeholder"""

    def __init__(self, table_a: "HiconaTable", table_b: "HiconaTable"):
        self._table_a = table_a
        self._table_b = table_b
        self._dataframe = pd.merge(
            table_a.dataframe(),
            table_b.dataframe(),
            how="outer",
            on=["bin1_id", "bin2_id"],
            suffixes=("_a", "_b"),
        )

        # TODO: add some form of check that the tables are comparable

    def get_dataframe(self, choice: str = "all") -> pd.DataFrame:
        """Placeholder"""

        df = self._dataframe
        if choice == "all":
            pass
        elif choice == "a":
            df = df[df["count_a"].notnull()]
        elif choice == "b":
            df = df[df["count_b"].notnull()]
        elif choice == "both":
            df = df[df["count_a"].notnull() & df["count_b"].notnull()]
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
        points = points.rename(columns={f"{column}_a": "x", f"{column}_b": "y"})

        handle = cooler.Cooler(self._table_a.uris.cooler_uri())
        highlight = cooler.annotate(self.get_dataframe("both"), handle.bins()[:])

        part_a = highlight.query("HMM_annot1 == 'Prom' and HMM_annot2 == 'Enh'")
        part_b = highlight.query("HMM_annot1 == 'Enh' and HMM_annot2 == 'Prom'")
        highlight = pd.concat([part_a, part_b])[[f"{column}_a", f"{column}_b"]]
        highlight = highlight.rename(columns={f"{column}_a": "x", f"{column}_b": "y"})

        _plotting.plot_comparison(
            self._table_a.get_distribution(column),
            self._table_b.get_distribution(column),
            points,
            highlight,
            names,
            img_path,
            show,
        )
