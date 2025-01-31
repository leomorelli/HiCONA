import polars as pl


def bin_annot_test_grid():

    all_mod_frac_overlap_dict = {
        "A_frac_overlap": [0.9, None, None, 0.6, None, 1.0, 1.0, 1.0, 1.0, 0.05],
        "B_frac_overlap": [None, 1.0, 1.0, 0.4, None, None, None, None, None, None],
        "null_frac_overlap": [0.1, None, None, None, 1.0, None, None, None, None, 0.95],
    }

    all_mod_chrom_enrich_dict = {
        "A_chrom_enrich": [3.83, None, None, 3.29, None, 3.97, 3.97, 3.97, 3.97, 0.79],
        "B_chrom_enrich": [None, 5.12, 5.12, 3.86, None, None, None, None, None, None],
        "null_chrom_enrich": [
            0.15,
            None,
            None,
            None,
            1.08,
            None,
            None,
            None,
            None,
            1.04,
        ],
    }

    return (
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": "auto",
                "save_all_mods": False,
            },
            ["A", "B", "B", "A", "None", "A", "A", "A", "A", "A"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": True,
                "save_all_mods": False,
            },
            ["A", "B", "B", "A", "None", "A", "A", "A", "A", "A"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": False,
                "save_all_mods": False,
            },
            ["A", "B", "B", "A", "None", "A", "A", "A", "A", "None"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": "auto",
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "A"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": True,
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "A"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": False,
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "None"],
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": "auto",
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_frac_overlap_dict),
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": True,
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_frac_overlap_dict),
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": True,
                "ignore_null_mode": False,
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_frac_overlap_dict),
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": "auto",
                "save_all_mods": True,
            },
            ValueError(),
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": True,
                "save_all_mods": True,
            },
            ValueError(),
        ),
        (
            {
                "metric": "frac_overlap",
                "consolidate": False,
                "ignore_null_mode": False,
                "save_all_mods": True,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": "auto",
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "None"],
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": True,
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "A"],
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": False,
                "save_all_mods": False,
            },
            ["A", "B", "B", "B", "None", "A", "A", "A", "A", "None"],
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": "auto",
                "save_all_mods": False,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": True,
                "save_all_mods": False,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": False,
                "save_all_mods": False,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": "auto",
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_chrom_enrich_dict),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": True,
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_chrom_enrich_dict),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": True,
                "ignore_null_mode": False,
                "save_all_mods": True,
            },
            pl.DataFrame(all_mod_chrom_enrich_dict),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": "auto",
                "save_all_mods": True,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": True,
                "save_all_mods": True,
            },
            ValueError(),
        ),
        (
            {
                "metric": "chrom_enrich",
                "consolidate": False,
                "ignore_null_mode": False,
                "save_all_mods": True,
            },
            ValueError(),
        ),
    )
