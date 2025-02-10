import polars as pl
import polars.testing as pt

from hicona import HiconaCooler
from hicona.misc import (
    get_expected_counts,
    expected_normalized_pixels,
    expected_normalized_cooler,
)
from tests._common import isolated_filesystem

PATH_EXP_COUNTS = "tests/data/expected_counts.tsv"


def test_get_expected_counts(mock_cooler):

    chroms = pl.from_pandas(mock_cooler.chroms()[:])
    exp_counts = get_expected_counts(mock_cooler)

    # Expected number is number of handshakes, or num chroms choose 2
    expected_inter_num = chroms.height * (chroms.height - 1) * 0.5
    observed_inter_num = exp_counts.filter(pl.col("chrom1") != pl.col("chrom2")).height
    assert expected_inter_num == observed_inter_num

    # Expected number is given by
    # sum over i of (pixels in chromosome i - 1) e.i. last id i - first id i
    # This conveniently simplifies to num bins
    expected_intra_num = mock_cooler.info["nbins"]
    observed_intra_num = exp_counts.filter(pl.col("chrom1") == pl.col("chrom2")).height
    assert expected_intra_num == observed_intra_num

    # NOTE: actual values are not checked, since it would require re-implementing
    # they are just compared to previously saved ones to ensure no backend changes
    old_expected = pl.read_csv(PATH_EXP_COUNTS, separator="\t")
    pt.assert_frame_equal(exp_counts, old_expected)

    # Check that the masking bug is properly addressed
    # (https://github.com/open2c/cooltools/issues/509)
    dist_less_2 = exp_counts.filter((0 <= pl.col("dist")) & (pl.col("dist") < 2))
    assert dist_less_2.height > 0
    assert dist_less_2.filter(pl.col("expected").is_not_null()).height == 0


def test_expected_normalized_pixels(mock_pix_table):

    exp_counts = pl.read_csv(PATH_EXP_COUNTS, separator="\t")
    pixels = mock_pix_table.get_dataframe(annotate=True)

    norm_pix = pl.concat(expected_normalized_pixels(pixels, exp_counts))
    assert norm_pix.height < pixels.height

    all_match = (
        norm_pix.join(
            pixels.select("bin1_id", "bin2_id", "count").rename({"count": "raw_count"}),
            on=["bin1_id", "bin2_id"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("chrom1") != pl.col("chrom2"))
            .then(pl.lit(-1))
            .otherwise(pl.col("bin2_id") - pl.col("bin1_id"))
            .alias("dist")
        )
        .join(exp_counts, on=["chrom1", "chrom2", "dist"], how="left")
        .with_columns(
            (pl.col("count") == (pl.col("raw_count") / pl.col("expected"))).alias(
                "match"
            )
        )
        .get_column("match")
        .all()
    )

    assert all_match


def test_expected_normalized_cooler(mock_cooler, mock_pix_table):
    NEW_COOL_PATH = "norm_cool.cool"

    exp_counts = pl.read_csv(PATH_EXP_COUNTS, separator="\t")
    pixels = mock_pix_table.get_dataframe(
        annotate=True,
        selection_kwargs={"balance": True},
    )
    old_pix = pl.concat(expected_normalized_pixels(pixels, exp_counts))

    with isolated_filesystem():
        expected_normalized_cooler(mock_cooler, NEW_COOL_PATH)
        new_cool = HiconaCooler(NEW_COOL_PATH)
        new_pix = new_cool.get_pixel_table().get_dataframe(annotate=True)

        pt.assert_frame_equal(new_pix, old_pix)
