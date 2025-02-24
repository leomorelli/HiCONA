import os

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pt
import pytest

from hicona import BinTable, PixelTable
from _common import isolated_filesystem
from _test_grids import bin_annot_test_grid

CLUST_REGION = "chr1:0-100000"


def test_bin_table_bin_size(mock_bin_table):
    assert mock_bin_table.bin_size == 10_000


def test_bin_table_store_size(mock_bin_table):
    resized_tab = BinTable(mock_bin_table.get_dataframe(), store_size=50_000)
    assert resized_tab.store_size == 50_000
    assert mock_bin_table.store_size == 10_000_000


@pytest.mark.parametrize(
    "query,expected",
    (
        ("chrII", (33, 115)),
        ("chrII:0-1000000000", (33, 115)),
        ("chrX:20000-40000", (600, 602)),
    ),
)
def test_bin_table_extent(mock_bin_table, query, expected):
    assert mock_bin_table.extent(query) == expected


def test_bin_table_copy(mock_bin_table):
    table_copy = mock_bin_table.copy()
    pt.assert_frame_equal(table_copy.get_dataframe(), mock_bin_table.get_dataframe())
    assert id(mock_bin_table) != id(table_copy)


@pytest.mark.parametrize(
    "query,expected_err",
    (
        ("chrII:9292", ValueError),
        ("chrII:-939393", ValueError),
    ),
)
def test_bin_table_extent_bad_queries(mock_bin_table, query, expected_err):
    with pytest.raises(expected_err):
        mock_bin_table.extent(query)


def test_bin_table_col_names(mock_bin_table):
    expected_names = ["bin_id", "chrom", "start", "end", "weight"]
    assert all([name in expected_names for name in mock_bin_table.col_names])
    assert len(mock_bin_table.col_names) == 5


def test_bin_table_get_dataframe(mock_bin_table):
    # Test return dtype
    assert isinstance(mock_bin_table.get_dataframe(), pl.DataFrame)
    assert isinstance(mock_bin_table.get_dataframe(dtype="polars"), pl.DataFrame)
    assert isinstance(mock_bin_table.get_dataframe(dtype="pandas"), pd.DataFrame)

    # Test table subsetting
    assert mock_bin_table.get_dataframe().height == 1226
    assert mock_bin_table.get_dataframe("chrII").height == 82
    assert mock_bin_table.get_dataframe("chrII:10000-70000").height == 6


@pytest.mark.parametrize("params,expected", bin_annot_test_grid())
def test_bin_table_add_annotation(mock_bin_table, mock_yeast_hmm, params, expected):

    test_region = "chrII:0-100000"
    base_cols = ["chrom", "start", "end"]
    anno_name = [c for c in mock_yeast_hmm.columns if c not in base_cols][0]

    # Save all mods == False
    if isinstance(expected, list):
        mock_bin_table.add_annotation(mock_yeast_hmm, **params)
        out_arr = (
            mock_bin_table.get_dataframe(test_region).get_column(anno_name).to_list()
        )
        assert out_arr == expected

    # Save all mods == True
    elif isinstance(expected, pl.DataFrame):
        mock_bin_table.add_annotation(mock_yeast_hmm, **params)
        out_arr = mock_bin_table.get_dataframe(test_region).drop(
            ["bin_id", "chrom", "start", "end", "weight"]
        )

        for col in out_arr.columns:
            arr_col = out_arr.get_column(col).round(2).to_list()
            exp_col = expected.get_column(col).to_list()
            assert arr_col == exp_col

    # Bad combination parameters
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            mock_bin_table.add_annotation(mock_yeast_hmm, **params)


def test_bin_table_save_and_load(mock_bin_table):
    TEST_PATH = "test_bin_table"

    with isolated_filesystem():
        mock_bin_table.save(TEST_PATH)
        new_table = BinTable.load(TEST_PATH)
        pt.assert_frame_equal(mock_bin_table.get_dataframe(), new_table.get_dataframe())

        with pytest.raises(OSError):
            mock_bin_table.save(TEST_PATH)

        with pytest.raises(OSError):
            BinTable.load("some_random_path_that_does_not_exist")

        os.makedirs("empty_folder")
        with pytest.raises(ValueError):
            BinTable.load("empty_folder")


def test_pix_table_bins(mock_pix_table):
    assert isinstance(mock_pix_table.bins, BinTable)


def test_pix_table_store_size(mock_pix_table):
    resized_tab = PixelTable(
        mock_pix_table.get_chunks(),
        bins=mock_pix_table.bins.get_dataframe(),
        store_size=50_000,
    )
    assert resized_tab.store_size == 50_000
    assert resized_tab.bins.store_size == 50_000
    assert mock_pix_table.store_size == 10_000_000
    assert mock_pix_table.bins.store_size == 10_000_000


def test_pix_table_copy(mock_pix_table):

    table_copy = mock_pix_table.copy()

    pt.assert_frame_equal(
        table_copy.bins.get_dataframe(),
        mock_pix_table.bins.get_dataframe(),
    )
    pt.assert_frame_equal(
        table_copy.get_dataframe(),
        mock_pix_table.get_dataframe(),
    )
    assert id(mock_pix_table) != id(table_copy)
    assert id(mock_pix_table.bins) != id(table_copy.bins)


def test_pix_table_colnames(mock_pix_table):
    expected_names = ["bin1_id", "bin2_id", "count"]
    assert len(mock_pix_table.col_names) == 3
    assert all(n in expected_names for n in mock_pix_table.col_names)


def test_pix_table_get_dataframe(mock_pix_table):
    # Test return dtype
    assert isinstance(mock_pix_table.get_dataframe(), pl.DataFrame)
    assert isinstance(mock_pix_table.get_dataframe(dtype="polars"), pl.DataFrame)
    assert isinstance(mock_pix_table.get_dataframe(dtype="pandas"), pd.DataFrame)

    # Test table subsetting
    assert mock_pix_table.get_dataframe().height == 728311
    assert mock_pix_table.get_dataframe("chrII").height == 3_355
    assert mock_pix_table.get_dataframe("chrII:10000-70000").height == 21

    # Test annotation
    assert len(mock_pix_table.get_dataframe("chrII").columns) == 3
    assert len(mock_pix_table.get_dataframe("chrII", annotate=True).columns) == 11

    # TODO: maybe test strategies here too, rather than only in the chunks


def test_pix_table_get_chunks(mock_pix_table):
    # Test return dtype
    assert isinstance(next(mock_pix_table.get_chunks()), pl.DataFrame)
    assert isinstance(next(mock_pix_table.get_chunks(dtype="polars")), pl.DataFrame)
    assert isinstance(next(mock_pix_table.get_chunks(dtype="pandas")), pd.DataFrame)

    # Test table subsetting
    assert next(mock_pix_table.get_chunks("chrII")).height == 3_355
    assert next(mock_pix_table.get_chunks("chrII:10000-70000")).height == 21
    assert next(mock_pix_table.get_chunks("chrII", chunk_size=10)).height == 10

    # Test chunk size
    assert next(mock_pix_table.get_chunks()).height == 728311
    assert next(mock_pix_table.get_chunks(chunk_size=10000)).height == 10000

    # Test annotation
    assert len(next(mock_pix_table.get_chunks("chrII")).columns) == 3
    assert len(next(mock_pix_table.get_chunks("chrII", annotate=True)).columns) == 11

    # Test balance
    balanced_val = next(
        mock_pix_table.get_chunks("chrI:0-10000", balance=True)
    ).get_column("count")[0]
    assert balanced_val == 0.15904249785884622


def test_pix_table_get_matrix(mock_pix_table):

    mat = mock_pix_table.get_matrix()
    full_region = "chrII:0-100000"
    full_mat = mock_pix_table.get_matrix(full_region)

    # Basic sanity checks
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (1226, 1226)
    assert isinstance(full_mat, np.ndarray)
    assert full_mat.shape == (10, 10)
    assert full_mat.any()

    # Mode check
    upper_mat = mock_pix_table.get_matrix(full_region, mode="upper")
    lower_mat = mock_pix_table.get_matrix(full_region, mode="lower")
    assert (full_mat == mock_pix_table.get_matrix(full_region, mode="full")).all()
    assert (full_mat == full_mat.T).all()
    assert (upper_mat != upper_mat.T).any()
    assert (lower_mat != lower_mat.T).any()
    with pytest.raises(ValueError):
        mock_pix_table.get_matrix(full_region, mode="random_mode_name")

    # Size consistency checks
    sub_region = "chrII:10000-100000"
    sub_mat = mock_pix_table.subset(sub_region).get_matrix(full_region)
    assert sub_mat.shape == (10, 10)
    assert np.isnan(sub_mat[:, 0]).all()
    assert np.isnan(sub_mat[0, :]).all()

    # Diagonal check
    with_diag = mock_pix_table.get_matrix(full_region, mask_diagonal=False)
    no_diag = mock_pix_table.get_matrix(full_region, mask_diagonal=True)
    assert (full_mat.diagonal() == with_diag.diagonal()).all()
    assert np.isnan(no_diag.diagonal()).all()

    # Check value col
    fake_annot_table = PixelTable(
        mock_pix_table.get_dataframe().with_columns(pl.lit(1).alias("mock_val")),
        bins=mock_pix_table.bins,
    )
    assert (fake_annot_table.get_matrix(full_region, value_col="mock_val") == 1.0).all()

    # TODO: Maybe test one or more selection kwargs from the chunks method


def test_pix_table_get_graph(mock_pix_table):

    test_region = "chrII:0-100000"
    rm_self_loops = pl.col("bin1_id") != pl.col("bin2_id")

    full_graph = mock_pix_table.get_graph()
    pt.assert_frame_equal(
        mock_pix_table.get_dataframe().filter(rm_self_loops),
        pl.concat(full_graph.get_pixels()),
    )
    pt.assert_frame_equal(
        mock_pix_table.bins.get_dataframe(),
        pl.concat(full_graph.get_bins()),
    )

    sub_graph = mock_pix_table.get_graph(test_region)
    pt.assert_frame_equal(
        mock_pix_table.get_dataframe(test_region).filter(rm_self_loops),
        pl.concat(sub_graph.get_pixels()),
    )
    pt.assert_frame_equal(
        mock_pix_table.bins.get_dataframe(test_region),
        pl.concat(sub_graph.get_bins()),
    )


def test_pix_table_subset(mock_pix_table):

    test_region = "chrII:0-100000"
    subset = mock_pix_table.subset(test_region)

    assert isinstance(subset, PixelTable)
    assert mock_pix_table.bins.get_dataframe().equals(subset.bins.get_dataframe())
    assert mock_pix_table.get_dataframe().height > subset.get_dataframe().height
    assert mock_pix_table.get_dataframe(test_region).equals(subset.get_dataframe())


def test_pix_table_apply(mock_pix_table):

    def mock_strat(chunks):
        for chunk in chunks:
            yield chunk.filter(pl.col("count") > 10)

    ref = mock_pix_table.get_dataframe().filter(pl.col("count") > 10)
    app = mock_pix_table.apply(mock_strat).get_dataframe()

    assert ref.equals(app)


@pytest.mark.parametrize("params,expected", bin_annot_test_grid())
def test_pix_table_add_bin_annotation(mock_pix_table, mock_yeast_hmm, params, expected):

    test_region = "chrII:0-100000"
    base_cols = ["chrom", "start", "end"]
    anno_name = [c for c in mock_yeast_hmm.columns if c not in base_cols][0]

    # Save all mods == False
    if isinstance(expected, list):
        mock_pix_table.add_bin_annotation(mock_yeast_hmm, **params)
        out_arr = (
            mock_pix_table.bins.get_dataframe(test_region)
            .get_column(anno_name)
            .to_list()
        )
        assert out_arr == expected

    # Save all mods == True
    elif isinstance(expected, pl.DataFrame):
        mock_pix_table.add_bin_annotation(mock_yeast_hmm, **params)
        out_arr = mock_pix_table.bins.get_dataframe(test_region).drop(
            ["bin_id", "chrom", "start", "end", "weight"]
        )

        for col in out_arr.columns:
            arr_col = out_arr.get_column(col).round(2).to_list()
            exp_col = expected.get_column(col).to_list()
            assert arr_col == exp_col

    # Bad combination parameters
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            mock_pix_table.add_bin_annotation(mock_yeast_hmm, **params)


def test_pix_table_add_pix_annotation(mock_pix_table):

    mock_pix_anno = (
        mock_pix_table.get_dataframe()
        .select("bin1_id", "bin2_id")
        .with_columns(
            pl.lit(5).alias("numeric_anno"),
            pl.lit("test").alias("string_anno"),
            pl.lit(np.NaN).alias("nan_anno"),
        )
    )

    mock_pix_table.add_pix_annotation(mock_pix_anno)
    assert (
        mock_pix_table.get_dataframe()
        .select("bin1_id", "bin2_id", "numeric_anno", "string_anno", "nan_anno")
        .equals(mock_pix_anno)
    )

    # Test for repetitive name annotations
    with pytest.raises(NotImplementedError):
        mock_pix_table.add_pix_annotation(mock_pix_anno.to_pandas())


def test_pix_table_save_and_load(mock_pix_table):
    TEST_PATH = "test_pix_table"

    with isolated_filesystem():
        mock_pix_table.save(TEST_PATH)
        new_table = PixelTable.load(TEST_PATH)
        pt.assert_frame_equal(mock_pix_table.get_dataframe(), new_table.get_dataframe())
        pt.assert_frame_equal(
            mock_pix_table.bins.get_dataframe(),
            new_table.bins.get_dataframe(),
        )

        with pytest.raises(OSError):
            mock_pix_table.save(TEST_PATH)

        with pytest.raises(OSError):
            PixelTable.load("some_random_path_that_does_not_exist")

        os.makedirs("empty_folder")
        with pytest.raises(ValueError):
            PixelTable.load("empty_folder")


@pytest.mark.parametrize("mode", ("no", "bins", "pixels"))
def test_pix_table_add_clustering_modes(mock_small_pix_table, mode):
    # NOTE: no check is performed on the actual values, no reasonable way to do it

    mock_small_pix_table.add_clustering(CLUST_REGION, marginals=mode)

    match mode:
        case "no":
            assert "pix_prob" not in mock_small_pix_table.col_names
            assert "pix_group" not in mock_small_pix_table.col_names
            assert "bin_prob" not in mock_small_pix_table.bins.col_names
        case "bins":
            assert "pix_prob" not in mock_small_pix_table.col_names
            assert "pix_group" not in mock_small_pix_table.col_names
            assert "bin_prob" in mock_small_pix_table.bins.col_names
        case "pixels":
            assert "pix_prob" in mock_small_pix_table.col_names
            assert "pix_group" in mock_small_pix_table.col_names
            assert "bin_prob" in mock_small_pix_table.bins.col_names


def test_pix_table_add_clustering_invalid_mode(mock_small_pix_table):
    INVALID_MODE = "Resistance is futile, you will be assimilated"

    with pytest.raises(ValueError):
        mock_small_pix_table.add_clustering(CLUST_REGION, marginals=INVALID_MODE)


def test_pix_table_add_clustering_seed(mock_small_pix_table):

    rep1 = mock_small_pix_table.copy()
    rep1.add_clustering(CLUST_REGION, seed=23, marginals="bins")

    rep2 = mock_small_pix_table.copy()
    rep2.add_clustering(CLUST_REGION, seed=23, marginals="bins")

    diff = mock_small_pix_table.copy()
    diff.add_clustering(CLUST_REGION, seed=42, marginals="bins")

    pt.assert_frame_equal(rep1.bins.get_dataframe(), rep2.bins.get_dataframe())
    pt.assert_frame_equal(rep1.get_dataframe(), rep1.get_dataframe())

    pt.assert_frame_not_equal(rep1.bins.get_dataframe(), diff.bins.get_dataframe())


def test_pix_table_add_clustering_float_counts(mock_pix_table):
    # NOTE: no check is performed on the actual values, no reasonable way to do it

    table = PixelTable(
        mock_pix_table.get_chunks(balance=True),
        bins=mock_pix_table.bins.get_dataframe(),
    )
    table.add_clustering("chrI:0-100000", marginals="pixels")

    assert "pix_prob" in table.col_names
    assert "pix_group" in table.col_names
    assert "bin_prob" in table.bins.col_names


def test_pix_table_add_clustering_all_bins_and_pixels_kept(mock_small_pix_table):

    init_bin = mock_small_pix_table.bins.get_dataframe()
    init_pix = mock_small_pix_table.get_dataframe()

    mock_small_pix_table.add_clustering(CLUST_REGION, marginals="pixels")

    after_bin = mock_small_pix_table.bins.get_dataframe()
    after_pix = mock_small_pix_table.get_dataframe()

    # All bins kept
    assert init_bin.height == after_bin.height
    # All bins assigned to a cluster
    assert init_bin.height == after_bin.select("level_(0)").drop_nans().height
    # All pixels kept
    assert (
        init_pix.height
        == init_pix.join(after_pix, on=("bin1_id", "bin2_id"), how="inner").height
    )
    # All pixels have a class
    assert after_pix.height == after_pix.select("pix_group").drop_nans().height
