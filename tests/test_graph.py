import os.path as op

import graph_tool.all as gt
import pandas as pd
import polars as pl
import polars.testing as pt
import pytest

from hicona import HiconaGraph, PixelTable

testdir = op.realpath(op.dirname(__file__))
datadir = op.join(testdir, "data")

CLUST_REGION = "chr1:0-100000"


def test_graph_init(mock_pix_table):
    pixels = mock_pix_table.get_dataframe()
    bins = mock_pix_table.bins.get_dataframe()

    graph = HiconaGraph(bins=bins, pixels=pixels)
    assert isinstance(graph.graph, gt.Graph)
    assert "genomic" in graph.graph.edge_properties.keys()

    no_self_loop_pix = pixels.filter(pl.col("bin1_id") != pl.col("bin2_id"))
    assert no_self_loop_pix.equals(pl.concat(graph.get_pixels()))
    pt.assert_frame_equal(bins, pl.concat(graph.get_bins()))

    # TODO: add explicit test for genomic link


def test_graph_get_bins(mock_graph, mock_bin_table):

    # Basic fetching
    table_bins = mock_bin_table.get_dataframe()
    graph_bins = pl.concat(mock_graph.get_bins())
    pt.assert_frame_equal(graph_bins, table_bins)

    # Node ids
    assert "node_id" not in graph_bins.columns
    assert "node_id" not in pl.concat(mock_graph.get_bins(drop_node_id=True)).columns
    assert "node_id" in pl.concat(mock_graph.get_bins(drop_node_id=False)).columns

    # Bare
    def only_base_cols(df):
        base_cols = ["bin_id", "chrom", "start", "end"]
        cond_1 = len([c for c in base_cols if c in df.columns]) == 4
        cond_2 = len(df.columns) == 4
        return cond_1 and cond_2

    assert not only_base_cols(graph_bins)
    assert only_base_cols(pl.concat(mock_graph.get_bins(bare=True)))
    assert not only_base_cols(pl.concat(mock_graph.get_bins(bare=False)))

    # Chunk_size
    def all_within_size(chunks, size):
        return all(map(lambda x: len(x) <= size, chunks))

    assert all_within_size(mock_graph.get_pixels(), 10_000_000)
    assert all_within_size(mock_graph.get_pixels(chunk_size=100), 100)

    # Dtype
    def all_of_dtype(chunks, dtype):
        return all(map(lambda x: type(x) == dtype, chunks))

    assert all_of_dtype(mock_graph.get_bins(), pl.DataFrame)
    assert all_of_dtype(mock_graph.get_bins(dtype="polars"), pl.DataFrame)
    assert all_of_dtype(mock_graph.get_bins(dtype="pandas"), pd.DataFrame)


def test_graph_get_pixels(mock_graph, mock_pix_table):

    # Basic test
    rm_self_loops = pl.col("bin1_id") != pl.col("bin2_id")
    table_pix = mock_pix_table.get_dataframe().filter(rm_self_loops)
    graph_pix = pl.concat(mock_graph.get_pixels())
    pt.assert_frame_equal(table_pix, graph_pix)

    # Keep genomic
    with_genomic = pl.concat(mock_graph.get_pixels(keep_genomic=True))
    rm_genomic = pl.concat(mock_graph.get_pixels(keep_genomic=False))

    assert "genomic" not in graph_pix.columns
    assert "genomic" not in rm_genomic.columns
    assert "genomic" in with_genomic.columns

    assert with_genomic.height > rm_genomic.height
    pt.assert_frame_equal(
        table_pix,
        with_genomic.filter(pl.col("genomic") == False).drop("genomic"),
    )

    # Chunk_size
    def all_within_size(chunks, size):
        return all(map(lambda x: len(x) <= size, chunks))

    assert all_within_size(mock_graph.get_pixels(), 10_000_000)
    assert all_within_size(mock_graph.get_pixels(chunk_size=50_000), 50_000)

    # Dtype
    def all_of_dtype(chunks, dtype):
        return all(map(lambda x: type(x) == dtype, chunks))

    assert all_of_dtype(mock_graph.get_pixels(), pl.DataFrame)
    assert all_of_dtype(mock_graph.get_pixels(dtype="polars"), pl.DataFrame)
    assert all_of_dtype(mock_graph.get_pixels(dtype="pandas"), pd.DataFrame)


def test_graph_from_cooler(mock_cooler):

    rm_self_loops = pl.col("bin1_id") != pl.col("bin2_id")

    pix_table = mock_cooler.get_pixel_table()
    bin_table = mock_cooler.get_bin_table()

    full_graph = HiconaGraph.from_cooler(mock_cooler)
    pt.assert_frame_equal(
        pl.concat(full_graph.get_bins()),
        bin_table.get_dataframe(),
    )
    pt.assert_frame_equal(
        pl.concat(full_graph.get_pixels()),
        pix_table.get_dataframe().filter(rm_self_loops),
    )

    subset_region = "chrII"
    part_graph = HiconaGraph.from_cooler(mock_cooler, region=subset_region)
    pt.assert_frame_equal(
        pl.concat(part_graph.get_bins()),
        bin_table.get_dataframe(subset_region),
    )
    pt.assert_frame_equal(
        pl.concat(part_graph.get_pixels()),
        pix_table.get_dataframe(subset_region).filter(rm_self_loops),
    )


def test_graph_from_pixel_table(mock_pix_table):

    rm_self_loops = pl.col("bin1_id") != pl.col("bin2_id")

    full_graph = HiconaGraph.from_pixel_table(mock_pix_table)
    pt.assert_frame_equal(
        pl.concat(full_graph.get_bins()),
        mock_pix_table.bins.get_dataframe(),
    )
    pt.assert_frame_equal(
        pl.concat(full_graph.get_pixels()),
        mock_pix_table.get_dataframe().filter(rm_self_loops),
    )

    subset_region = "chrII"
    part_graph = HiconaGraph.from_pixel_table(mock_pix_table, region=subset_region)
    pt.assert_frame_equal(
        pl.concat(part_graph.get_bins()),
        mock_pix_table.bins.get_dataframe(subset_region),
    )
    pt.assert_frame_equal(
        pl.concat(part_graph.get_pixels()),
        mock_pix_table.get_dataframe(subset_region).filter(rm_self_loops),
    )


@pytest.mark.parametrize("mode", ("no", "bins", "pixels"))
def test_graph_compute_clustering_modes(mock_small_graph, mode):
    # NOTE: no check is performed on the actual values, no reasonable way to do it

    mock_small_graph.compute_clustering(marginals=mode)
    table = PixelTable(mock_small_graph.get_pixels(), bins=mock_small_graph.get_bins())

    match mode:
        case "no":
            assert "pix_prob" not in table.col_names
            assert "pix_group" not in table.col_names
            assert "bin_prob" not in table.bins.col_names
        case "bins":
            assert "pix_prob" not in table.col_names
            assert "pix_group" not in table.col_names
            assert "bin_prob" in table.bins.col_names
        case "pixels":
            assert "pix_prob" in table.col_names
            assert "pix_group" in table.col_names
            assert "bin_prob" in table.bins.col_names


def test_graph_compute_clustering_invalid_mode(mock_small_graph):
    INVALID_MODE = "Resistance is futile, you will be assimilated"

    with pytest.raises(ValueError):
        mock_small_graph.compute_clustering(marginals=INVALID_MODE)


def test_graph_compute_clustering_seed(mock_small_graph):

    def graph_copy():
        return HiconaGraph(
            bins=mock_small_graph.get_bins(),
            pixels=mock_small_graph.get_pixels(),
        )

    rep1 = graph_copy()
    rep1.compute_clustering(seed=23, marginals="bins")

    rep2 = graph_copy()
    rep2.compute_clustering(seed=23, marginals="bins")

    diff = graph_copy()
    diff.compute_clustering(seed=42, marginals="bins")

    pt.assert_frame_equal(pl.concat(rep1.get_bins()), pl.concat(rep2.get_bins()))
    pt.assert_frame_equal(pl.concat(rep1.get_pixels()), pl.concat(rep2.get_pixels()))

    pt.assert_frame_not_equal(
        pl.concat(rep1.get_bins()),
        pl.concat(diff.get_bins()),
    )


def test_graph_compute_clustering_float_counts(mock_pix_table):
    # NOTE: no check is performed on the actual values, no reasonable way to do it

    table = PixelTable(
        mock_pix_table.get_chunks(balance=True),
        bins=mock_pix_table.bins.get_dataframe(),
    )
    graph = table.get_graph("chrI:0-100000")
    graph.compute_clustering(marginals="pixels")

    table = PixelTable(graph.get_pixels(), bins=graph.get_bins())

    assert "pix_prob" in table.col_names
    assert "pix_group" in table.col_names
    assert "bin_prob" in table.bins.col_names


def test_graph_compute_clustering_all_bins_and_pixels_kept(mock_small_graph):

    init_bin = pl.concat(mock_small_graph.get_bins())
    init_pix = pl.concat(mock_small_graph.get_pixels())

    mock_small_graph.compute_clustering(marginals="pixels")

    after_bin = pl.concat(mock_small_graph.get_bins())
    after_pix = pl.concat(mock_small_graph.get_pixels())

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
