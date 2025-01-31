import os.path as op

import pytest
import polars as pl

from hicona import BinTable, PixelTable, HiconaCooler, HiconaGraph

testdir = op.realpath(op.dirname(__file__))
datadir = op.join(testdir, "data")


@pytest.fixture
def mock_cooler():
    return HiconaCooler(op.join(datadir, "yeast.10kb.cool"))


@pytest.fixture
def mock_yeast_hmm():
    return pl.read_csv(op.join(datadir, "mock_yeast_hmm.bed"), separator="\t")


@pytest.fixture
def mock_bin_table():
    return BinTable(
        pl.read_csv(op.join(datadir, "yeast.10kb.bins.tsv"), separator="\t")
    )


@pytest.fixture
def mock_pix_table():
    return PixelTable(
        pl.read_csv(op.join(datadir, "yeast.10kb.pixels.tsv"), separator="\t"),
        bins=pl.read_csv(op.join(datadir, "yeast.10kb.bins.tsv"), separator="\t"),
    )


@pytest.fixture
def mock_graph(mock_pix_table):
    pixels = mock_pix_table.get_dataframe()
    bins = mock_pix_table.bins.get_dataframe()
    return HiconaGraph(bins=bins, pixels=pixels)
