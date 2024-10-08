import pytest

from hicona import HiconaCooler


@pytest.fixture
def raw_test_cooler():
    """Raw cooler, first 1_000_000 lines of chr2 and chr10, Rao 2014 HUVEC"""
    return HiconaCooler("data/reduced_cooler.mcool::resolutions/10000")


# ////////////////////////////////////////////////////////////////////////////
# /////////////////////////// ANNOTATION FIXTURES ////////////////////////////
# ////////////////////////////////////////////////////////////////////////////


@pytest.fixture
def annotated_cooler():
    """Cooler annotated with 'compartment' and 'has_compartment' columns"""
    return HiconaCooler("data/annotated_cooler.mcool::resolutions/10000")


@pytest.fixture
def bin_annotation_header():
    """chr2 and chr10 compartment annotation for HMEC, with header"""
    return "data/bin_annotation_header.bed"


@pytest.fixture
def bin_annotation_headless():
    """chr2 and chr10 compartment annotation for HMEC, without header"""
    return "data/bin_annotation_headless.bed"
