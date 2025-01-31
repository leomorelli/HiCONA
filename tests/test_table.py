import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pt
import pytest

from hicona import BinTable, PixelTable
from _test_grids import bin_annot_test_grid


def test_bin_table_bin_size(mock_bin_table):
    assert mock_bin_table.bin_size == 10_000


@pytest.mark.parametrize(
    "query,expected",
    (
        ("chrII", (33, 115)),
        ("chrII:0-1000000000", (33, 115)),
        ("chrX:20000-40000", (600, 602)),
    ),
)
def test_bin_table_extent(mock_bin_table, query, expected):
    print(mock_bin_table.get_dataframe(query))
    assert mock_bin_table.extent(query) == expected


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
        annot_table = mock_bin_table.add_annotation(mock_yeast_hmm, **params)
        out_arr = annot_table.get_dataframe(test_region).get_column(anno_name).to_list()
        assert out_arr == expected

    # Save all mods == True
    elif isinstance(expected, pl.DataFrame):
        annot_table = mock_bin_table.add_annotation(mock_yeast_hmm, **params)
        out_arr = annot_table.get_dataframe(test_region).drop(
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


def test_pix_table_bins(mock_pix_table):
    assert isinstance(mock_pix_table.bins, BinTable)


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


def test_pix_table_add_clustering():
    pass
