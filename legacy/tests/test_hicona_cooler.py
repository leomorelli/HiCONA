"""Test properties and methods of HiconaCooler object"""

import cooler
import pytest

import hicona


def test_hicona_cooler_class_check(raw_test_cooler):
    """Test correct object inheritance"""
    assert isinstance(raw_test_cooler, hicona.HiconaCooler)
    assert isinstance(raw_test_cooler, cooler.Cooler)


# ////////////////////////////////////////////////////////////////////////////
# /////////////////////////// ANNOTATION FUNCTIONS ///////////////////////////
# ////////////////////////////////////////////////////////////////////////////

'''
@pytest.mark.parametrize(
    "ann_list, valid_list",
    [
        (None, []),
        ("chrom", []),
        ("has_compartment", ["has_compartment"]),
        (["chrom", "start", "end"], []),
        (["chrom", "start", "end", "compartment"], ["compartment"]),
    ],
)
def test_valid_bin_annotation(annotated_cooler, ann_list, valid_list):
    """Test that only valid bin annotations are returned"""
    assert annotated_cooler._valid_bin_annotations(ann_list) == valid_list
'''


def test_annotations_list_none(raw_test_cooler):
    """Test no bin annotations in a raw cooler (if weights are removed)"""
    assert raw_test_cooler.annotations_list() == []


def test_annotations_list_some(annotated_cooler):
    """Test corresct retrieval of annotations name from an annotated file"""
    test_cols = ["compartment", "has_compartment"]
    assert annotated_cooler.annotations_list() == test_cols


"""
def test_add_bin_annot():
    pass


def test_del_bin_annot():
    pass


def test_annotation_to_ohe():
    pass
"""
