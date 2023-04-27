"""Placeholder
Placeholder
"""

import re

from cooler import Cooler
from cooler.util import open_hdf5
from networkx import from_pandas_edgelist, to_pandas_edgelist
from numpy import log2, where
from pandas import concat, DataFrame

from utils import console_log, compute_alpha_val
from iterators import ChunkBordersIterator, ChromTablesIterator

__all__ = ["HiconaCooler"]

# TODO: check page for relative import problem
# https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time/14132912#14132912

# Dictionary of standard regular expressions to simplify chromosome fetching
_DEFAULT_CHROM_RE = {"humanCanonical": "^chr([1-9]|[1][0-9]|[2][0-2]|[XY])$"}
_GROUP_TEMPLATE = "countThr_{}_distThr_{}_stat_{}"


class HiconaCooler(Cooler):
    """Placeholder
    Placeholder
    """

    def pixel_chunks(self, chrom_id: str, chunk_size: int):
        """Generator function yielding pixel chunks of specified size."""

        extent = self.extent(chrom_id)
        borders = ChunkBordersIterator(self.store, self.root, extent, chunk_size)
        for lower, upper in borders:
            yield self.pixels()[lower:upper]

    def filter_pixels(
        self,
        pix_df: DataFrame,
        chrom_id: str,
        count_thr: int = 1,
        dist_thr: int = 10_000_000,
    ) -> None:
        """Filter out pixels non conformant to some condition.

        Remove all pixels that do NOT satisfy at least one of these filters:
        - distance among the bins is below maximal genomic distance allowed
        - number of counts of the interaction is above the minimal threshold
        - the bins are distinc (non-self looping)
        - the second bin does match the provided chromosome id

        Parameters
        ----------
        pix_df : :py:class:`DataFrame`
            Dataframe of pixels to filter.
        chrom_id: str
            Id of the chromosome whose internal pixels should be kept.
        count_thr : int, optional
            Threshold for pixel count; only keep pixels with count greater
            than this value. (default is 1)
        dist_thr : int, optional
            Threshold for genomic distance; only keep pixels whose bins are
            closer to each other than this distance.
        """

        max_diff = -(-dist_thr // self.binsize)
        id_upper_bound = self.extent(chrom_id)[1]

        indexer = pix_df[
            (pix_df["bin2_id"] - pix_df["bin1_id"] >= max_diff)
            | (pix_df["count"] <= count_thr)
            | (pix_df["bin1_id"] == pix_df["bin2_id"])
            | (pix_df["bin2_id"] >= id_upper_bound)
        ].index
        pix_df.drop(indexer, inplace=True)

        return pix_df

    @console_log
    def get_filtered_pixels(self, chrom_id, count_thr, dist_thr, chunk_size=10_000_000):
        """Filter pixels in chunks and return a single dataframe."""

        c_iter = self.pixel_chunks(chrom_id, chunk_size=chunk_size)
        pix_df = [self.filter_pixels(c, chrom_id, count_thr, dist_thr) for c in c_iter]
        pix_df = concat(pix_df, axis=0)

        return pix_df

    @console_log
    def drop_duplicate_pixels(self, pix_df):
        """pandas.drop_duplicates wrapper for logging purposes."""
        pix_df.drop_duplicates(subset=["bin1_id", "bin2_id"], inplace=True)

    @console_log
    def compute_decay(self, pix_df: DataFrame, stat: str = "median") -> None:
        """Add log2(observed/expected) counts ratio column to the pixels dataframe."""

        pix_df["bin_difference"] = pix_df["bin2_id"] - pix_df["bin1_id"]
        group_counts = pix_df.groupby("bin_difference")["count"]
        pix_df["exp_ratio"] = log2(pix_df["count"] / group_counts.transform(stat) + 1)
        pix_df.drop("bin_difference", axis=1, inplace=True)

    @console_log
    def add_sparsity_val(self, pix_df: DataFrame) -> None:
        """Add alpha value column to dataframe (computed as per Serrano et al. 2009).

        Add a column to the dataframe containing the alpha values for the edges,
        meaning the confidence level of a weighted edge given local fluctuations
        in the network, see Serrano et al. 2009 for in depth explaination. A graph
         is built starting from the list of edges, then the procedure is iterated
        over the nodes; the resulting network is converted back to edge list and
        sorted, since graph traversal does not have inherent order.

        WARNING: the edge in a disconnected doublet always gets alpha = 1,
        therefore it will be subsequently filtered. This behaviour might not be
        the desired one and might be changed.
        """

        # Add default alpha value
        pix_df["spar_alpha"] = 1
        graph = from_pandas_edgelist(
            pix_df,
            source="bin1_id",
            target="bin2_id",
            edge_attr=["exp_ratio", "spar_alpha"],
        )

        for node in graph:
            num_neighbours = len(graph[node])

            # The approach is not able to compute an alpha value if the number
            # of neighbours is one, therefore assign the default alpha value
            if num_neighbours == 1:
                continue

            weigths_sum = sum(graph[node][n]["exp_ratio"] for n in graph[node])
            for neigh in graph[node]:
                norm_weight = graph[node][neigh]["exp_ratio"] / weigths_sum
                new_alpha = compute_alpha_val(num_neighbours, norm_weight)
                old_aplha = graph[node][neigh]["spar_alpha"]
                graph[node][neigh]["spar_alpha"] = min(old_aplha, new_alpha)

        # TODO: cannot find if to_pandas_edgelist is already sorted or not
        graph = to_pandas_edgelist(graph, source="bin1_id", target="bin2_id")

        # Graph object does not preserve pair order
        # TODO: test for performance since it does not feel great
        graph["bin1_id"], graph["bin2_id"] = where(
            graph["bin1_id"] < graph["bin2_id"],
            (graph["bin1_id"], graph["bin2_id"]),
            (graph["bin2_id"], graph["bin1_id"]),
        )

        graph.sort_values(["bin1_id", "bin2_id"], inplace=True)
        pix_df["spar_alpha"] = graph["spar_alpha"].values

    def _create_chrom_table(self, chrom_id: str, table_root: str) -> None:
        """Create chromosome-level table given the set of parameters."""

        # Retrieve parameters from the parent group
        count_thr = table_root.attrs["count-threshold"]
        dist_thr = table_root.attrs["distance-threshold"]
        decay_stat = table_root.attrs["decay-statistic"]

        if chrom_id not in table_root:
            # Process the chromosome pixels
            chrom_pix = self.get_filtered_pixels(chrom_id, count_thr, dist_thr)
            self.drop_duplicate_pixels(chrom_pix)
            self.compute_decay(chrom_pix, decay_stat)
            self.add_sparsity_val(chrom_pix)

            # Save the dataframe columns as individual 1D-arrays (for storage)
            chrom_table = table_root.create_group(chrom_id)
            for column in chrom_pix.columns:
                chrom_table.create_dataset(column, data=chrom_pix[column])
            # TODO: Add duplicate of object properties for ease of retrieval?
        else:
            print(f"WARNING: {chrom_id} already processed with these params, skipping.")

    def _chrom_regex_to_iter(self, chrom_selection):
        """Convert chromosome selection from regex to iterable object"""

        if isinstance(chrom_selection, str):
            if _DEFAULT_CHROM_RE.get(chrom_selection):
                chrom_selection = _DEFAULT_CHROM_RE.get(chrom_selection)
            regex = re.compile(chrom_selection)
            chrom_selection = [c for c in self.chromnames if regex.match(c)]

        return chrom_selection

    def create_tables(
        self,
        chrom_selection: str = "humanCanonical",
        count_thr: int = 1,
        dist_thr: int = 2_000_000,
        decay_stat: str = "median",
    ) -> None:
        """Process and create chromosome-level tables to use for network construction.

        Given a set of chromosome and some parameters for the processing, create
        individual groups, each one corresponding to a chromosome and containing
        1D-arrays corresponding to the columns of the processed dataframe.

        Parameters
        ----------
        chrom_selection: str or iterable, optional
            Iterable of ids of the chromosome to process or regular expression to build
            one. Some strings are also accepted as proxy for common regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY" (default value)
            * other to be defined
        count_thr: int, optional
            Remove pixels whose row count is not greater than this value. (default is 1)
        dist_thr: int, optional
            Remove pixels whose genomic distance among bins is greater or equal to this
            value (in bp). (default is 2Mb)
        decay_stat: str, optional
            Statistic to use to summarize pixels with a certain distance while computing
            the expected counts. Must be compatible with pandas.transform. (default is
            "median")
        """

        # Convert chromosome selection to an iterable
        chrom_selection = self._chrom_regex_to_iter(chrom_selection)

        with open_hdf5(self.store, mode="a") as h5_handle:
            table_root = _GROUP_TEMPLATE.format(count_thr, dist_thr, decay_stat)
            table_root = self.root + "/chrom_tables/" + table_root

            # Create container group if not already existent
            if table_root not in h5_handle:
                table_grp = h5_handle.create_group(table_root)
                table_grp.attrs["count-threshold"] = count_thr
                table_grp.attrs["distance-threshold"] = dist_thr
                table_grp.attrs["decay-statistic"] = decay_stat
            else:
                table_grp = h5_handle[table_root]

            # Create chromosome-level groups and datasets
            for chrom_id in chrom_selection:
                print(f"STARTING {chrom_id}")
                self._create_chrom_table(chrom_id, table_grp)

    def tables_info(self):
        """Placeholder"""

        with open_hdf5(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/chrom_tables"]
            param_str = "PARAMETER SETS:"
            for param_grp in tables_grp.values():
                param_str += "\n" + "-" * 80
                param_list = [f"\n-{k}: {v}" for k, v in param_grp.attrs.items()]
                param_str += "".join(param_list)
                param_str += "\n-chromosomes:"
                param_str += "".join([f"\n\t--{k}" for k in param_grp.keys()])
            param_str += "\n" + "-" * 80
        print(param_str)

    def tables(
        self,
        chrom_selection: str = "humanCanonical",
        count_thr: int = None,
        dist_thr: int = None,
        decay_stat: str = None,
    ) -> ChromTablesIterator:
        """Return an iterator of selected tables and respective information.

        Use the input parameters to define a list of partial URIs strings
        corresponding to the groups where the column tables are stored, then
        return an iterator object where each item is a tuple in the form
        (corresponding pandas Dataframe, dictionary of dataframe information).

        Parameters
        ----------
        chrom_selection: str or iterable, optional
            Iterable of ids of the chromosome to fetch or regular expression to build
            one. Some strings are also accepted as proxy for common regular espressions:
            * humanCanonical: "chr1" to "chr22" plus "chrX" and "chrY" (default value)
            * other to be defined
        count_thr: int, optional
            Fetch tables created using this value as count threshold. If None, consider
            all tables regardless of the used value.
        dist_thr: int, optional
            Fetch tables created using this value as distance threshold. If None, consider
            all tables regardless of the used value.
        decay_stat: str, optional
            Fetch tables created using this statistic to compute count decay. If None,
            consider all tables regardless of the used statistic.
        """

        # Create valid groups regex according to input parameters
        chrom_selection = self._chrom_regex_to_iter(chrom_selection)
        count_thr = r"\S+" if not count_thr else count_thr
        dist_thr = r"\S+" if not dist_thr else dist_thr
        decay_stat = r"\S+" if not decay_stat else decay_stat
        grp_regex = re.compile(_GROUP_TEMPLATE.format(count_thr, dist_thr, decay_stat))

        # Define a list of partial URIs to valid tables
        with open_hdf5(self.store, mode="r") as h5_handle:
            tables_grp = h5_handle[self.root + "/chrom_tables"]
            valid_groups = [g for g in tables_grp if grp_regex.match(g)]
            valid_tables = []
            for grp in valid_groups:
                tabs = [grp + "/" + c for c in tables_grp[grp] if c in chrom_selection]
                valid_tables.extend(tabs)

        return ChromTablesIterator(self.store, self.root, valid_tables)


if __name__ == "__main__":
    COOL_PATH = "../test_files/small.mcool::resolutions/10000"
    hc = HiconaCooler(COOL_PATH)
    hc.tables_info()

    # hc.create_tables(chrom_selection=["chr1", "chr2", "chr3", "chr4"])
