import os.path as op
from shutil import copyfile

import cooler
import pandas as pd
import polars as pl
import pytest

from hicona import HiconaCooler, BinTable, PixelTable
from _common import isolated_filesystem
from _test_grids import bin_annot_test_grid

testdir = op.realpath(op.dirname(__file__))
datadir = op.join(testdir, "data")


def test_cooler_object(mock_cooler):

    assert isinstance(mock_cooler, cooler.Cooler)
    assert isinstance(mock_cooler.bins()[:10], pd.DataFrame)
    assert isinstance(mock_cooler.pixels()[:10], pd.DataFrame)
    assert isinstance(mock_cooler.chroms()[:10], pd.DataFrame)


@pytest.mark.parametrize("store_size", [100_000, 1_000_000])
def test_cooler_get_bin_table(mock_cooler, store_size):

    bin_table = mock_cooler.get_bin_table(store_size=store_size)
    assert isinstance(bin_table, BinTable)
    assert bin_table.store_size == store_size


@pytest.mark.parametrize("store_size", [100_000, 1_000_000])
def test_cooler_get_pixel_table(mock_cooler, store_size):
    TEST_REGION = "chrII:10000-40000"

    full_table = mock_cooler.get_pixel_table(store_size=store_size)
    assert isinstance(full_table, PixelTable)
    assert full_table.store_size == store_size

    num_chroms = full_table.get_dataframe(annotate=True).select("chrom1").n_unique()
    assert num_chroms == 18

    subset_table = mock_cooler.get_pixel_table(TEST_REGION, store_size=store_size)
    subset_df = subset_table.get_dataframe(annotate=True)
    unique_chroms = subset_df.select("chrom1").unique()
    assert unique_chroms.height == 1
    assert unique_chroms.rows(named=True)[0].get("chrom1") == "chrII"
    assert subset_df.min().rows(named=True)[0].get("start1") == 10_000
    assert subset_df.max().rows(named=True)[0].get("end1") == 40_000
    assert subset_df.min().rows(named=True)[0].get("start2") == 10_000
    assert subset_df.max().rows(named=True)[0].get("end2") == 40_000


@pytest.mark.parametrize("params,expected", bin_annot_test_grid())
def test_cooler_bin_annot_add(mock_yeast_hmm, params, expected):
    cool_file = "yeast.10kb.cool"
    test_region = "chrII:0-100000"

    base_cols = ["chrom", "start", "end"]
    anno_name = [c for c in mock_yeast_hmm.columns if c not in base_cols][0]

    with isolated_filesystem():
        cool_path = copyfile(op.join(datadir, cool_file), cool_file)
        cool_obj = HiconaCooler(cool_path)

        # Save all mods == False
        if isinstance(expected, list):
            cool_obj.bin_annot_add(mock_yeast_hmm, **params)
            out_arr = (
                cool_obj.get_bin_table()
                .get_dataframe(test_region)
                .head(10)
                .get_column(anno_name)
                .to_list()
            )
            assert out_arr == expected

        # Save all mods == True
        elif isinstance(expected, pl.DataFrame):
            cool_obj.bin_annot_add(mock_yeast_hmm, **params)
            out_arr = (
                cool_obj.get_bin_table()
                .get_dataframe(test_region)
                .head(10)
                .drop(["bin_id", "chrom", "start", "end", "weight"])
            )

            for col in out_arr.columns:
                arr_col = out_arr.get_column(col).round(2).to_list()
                exp_col = expected.get_column(col).to_list()
                assert arr_col == exp_col

        # Bad combination parameters
        elif isinstance(expected, ValueError):
            with pytest.raises(ValueError):
                cool_obj.bin_annot_add(mock_yeast_hmm, **params)


def test_cooler_bin_annot_del(mock_yeast_hmm):
    cool_file = "yeast.10kb.cool"

    with isolated_filesystem():
        cool_path = copyfile(op.join(datadir, cool_file), cool_file)
        cool_obj = HiconaCooler(cool_path)

        init_cols = list(cool_obj.bins()[:].columns)
        cool_obj.bin_annot_add(mock_yeast_hmm)
        cool_obj.bin_annot_del("hmm")
        end_cols = list(cool_obj.bins()[:].columns)

        assert init_cols == end_cols


def test_cooler_bin_annot_del_base_cols():
    cool_file = "yeast.10kb.cool"

    for col in ["chrom", "start", "end"]:

        with isolated_filesystem():
            cool_path = copyfile(op.join(datadir, cool_file), cool_file)
            cool_obj = HiconaCooler(cool_path)

            with pytest.raises(ValueError):
                cool_obj.bin_annot_del(col)
