import os.path as op

import pytest
import polars as pl

from hicona import HiconaCooler

testdir = op.realpath(op.dirname(__file__))
datadir = op.join(testdir, "data")


@pytest.fixture
def mock_cooler():
    return HiconaCooler(op.join(datadir, "yeast.10kb.cool"))


@pytest.fixture
def mock_yeast_hmm():
    return pl.read_csv(op.join(datadir, "mock_yeast_hmm.bed"), separator="\t")
