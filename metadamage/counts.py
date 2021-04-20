# Scientific Library
import numpy as np
import pandas as pd

# Standard Library
import logging
import warnings

# Third Party
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from psutil import cpu_count
from tqdm.auto import tqdm

# First Party
from metadamage import io, utils
from metadamage.progressbar import console, progress


logger = logging.getLogger(__name__)

#%%

ACTG = ["A", "C", "G", "T"]

ref_obs_bases = []
for ref in ACTG:
    for obs in ACTG:
        ref_obs_bases.append(f"{ref}{obs}")


# fmt: off
# fmt: on

columns = [
    "tax_id",
    "tax_name",
    "tax_rank",
    "N_alignments",
    "strand",
    "position",
    *ref_obs_bases,
]


#%%


def clean_up_after_dask():
    utils.delete_folder("./dask-worker-space")


def get_subsitution_bases_to_keep(cfg):
    # ["C", "CT", "G", "GA"]
    forward = cfg.substitution_bases_forward
    reverse = cfg.substitution_bases_reverse
    bases_to_keep = [forward[0], forward, reverse[0], reverse]
    return bases_to_keep


def get_base_columns(df):
    base_columns = []
    for column in df.columns:
        if len(column) == 2 and column[0] in ACTG and column[1] in ACTG:
            base_columns.append(column)
    return base_columns


def add_N_counts(df):
    """ Adds the toal number of bases """
    base_columns = get_base_columns(df)
    df["N_counts"] = df[base_columns].sum(axis=1)
    return df


def get_reference_columns(df, ref):
    ref_columns = []
    for column in get_base_columns(df):
        if column[0] == ref:
            ref_columns.append(column)
    return ref_columns


def add_reference_counts(df, ref):
    reference_columns = get_reference_columns(df, ref)
    df[ref] = df[reference_columns].sum(axis=1)
    return df


def compute_fraction_and_uncertainty(x, N, set_zero_to_nan=False):
    f = x / N
    if set_zero_to_nan:
        f = f.mask(x == 0, np.nan)
    sf = np.sqrt(f * (1 - f) / N)
    return f, sf


def compute_error_rates(df, ref, obs):
    s_ref_obs = ref + obs
    x = df[s_ref_obs]
    N_ref = df[ref]
    # N_ref = df[ref_columns].sum(axis=1)
    f, sf = compute_fraction_and_uncertainty(x, N_ref)
    return f, sf


def add_error_rates(df, ref, obs, include_uncertainties=False):
    f, sf = compute_error_rates(df, ref, obs)
    df[f"f_{ref}{obs}"] = f
    if include_uncertainties:
        df[f"sf_{ref}{obs}"] = sf
    return df


def make_position_1_indexed(df):
    "Make the position, z, one-indexed (opposed to zero-indexed)"
    df["position"] += 1
    return df


def make_reverse_position_negative(df):
    pos = df["position"]
    is_reverse = ~utils.is_forward(df)
    pos_reverse = pos[~utils.is_forward(df)]
    # pos_reverse *= -1
    df["position"] = df["position"].mask(is_reverse, -pos_reverse)
    return df


def delayed_list_unknown_length(lst):
    @dask.delayed()
    def delayed_list_tmp(lst):
        out = []
        for l in lst:
            out.append(l)
        return out

    return delayed_list_tmp(lst)


def remove_base_columns(df):
    columns_to_keep = []
    columns = df.columns
    base_columns = get_base_columns(df)
    for col in columns:
        if col not in base_columns:
            columns_to_keep.append(col)
    return df[columns_to_keep]


def keep_only_base_columns(df, cfg):
    base_columns_to_keep = get_subsitution_bases_to_keep(cfg)
    columns_to_keep = []
    columns = df.columns
    base_columns = get_base_columns(df)
    base_cols_to_discard = [
        base for base in base_columns if base not in base_columns_to_keep
    ]
    for col in columns:
        if col not in base_cols_to_discard:
            columns_to_keep.append(col)
    return df[columns_to_keep]


def sort_by_alignments(df_top_N):
    pos = df_top_N["position"]
    df_top_N["order"] = pos.mask(pos > 0, 1 / pos)
    return df_top_N.sort_values(
        by=["N_alignments", "tax_id", "order"], ascending=False
    ).drop(columns=["order"])


def replace_nans_with_zeroes(df):
    return df.fillna(0)


def compute_k_sum_total(group, cfg):
    k_sum_total = 0
    forward_bases = (
        cfg.substitution_bases_forward[0] + cfg.substitution_bases_forward[1]
    )
    k_sum_total += group[group.position > 0][forward_bases].sum()
    reverse_bases = (
        cfg.substitution_bases_reverse[0] + cfg.substitution_bases_reverse[1]
    )
    k_sum_total += group[group.position < 0][reverse_bases].sum()
    return k_sum_total


def add_k_sum_counts(df, cfg):

    meta = pd.Series(
        [],
        name="k_sum_total",
        index=pd.Index([], name="tax_id", dtype=int),
        dtype=int,
    )

    ds = df.groupby("tax_id").apply(compute_k_sum_total, cfg, meta=meta)
    ds = ds.reset_index()
    df = dd.merge(df, ds, on=["tax_id"])
    return df


def compute_min_N_in_group(group, cfg):
    min_N_forward = group[group.position > 0][cfg.substitution_bases_forward[0]].min()
    min_N_reverse = group[group.position < 0][cfg.substitution_bases_reverse[0]].min()
    return min(min_N_forward, min_N_reverse)


def add_min_N_in_group(df, cfg):

    meta = pd.Series(
        [],
        name="min_N_in_group",
        index=pd.Index([], name="tax_id", dtype=int),
        dtype=int,
    )

    ds = df.groupby("tax_id").apply(compute_min_N_in_group, cfg, meta=meta)
    ds = ds.reset_index()
    df = dd.merge(df, ds, on=["tax_id"])
    return df


def filter_cut_based_on_cfg(df, cfg):
    query = f"(N_alignments >= {cfg.min_alignments}) "
    query += f"& (k_sum_total >= {cfg.min_k_sum})"
    query += f"& (min_N_in_group >= {cfg.min_N_at_each_pos})"
    return df.query(query)


def compute_counts_with_dask(cfg, use_processes=True):

    # Standard Library
    filename = cfg.filename

    # do not allow dask to use all the cores.
    # important to remove bugs on HEP
    n_workers = int(min(cfg.N_cores, cpu_count() * 0.6))
    logger.info(f"Dask: number of workers = {n_workers}.")

    with LocalCluster(
        n_workers=n_workers,
        processes=use_processes,
        silence_logs=logging.CRITICAL,
        dashboard_address=f":{cfg.dask_port}"
        # threads_per_worker=1,
    ) as cluster, Client(cluster) as client:

        df = (
            dd.read_csv(
                filename,
                sep="\t",
                header=None,
                names=columns,
            )
            # compute error rates
            .pipe(add_reference_counts, ref=cfg.substitution_bases_forward[0])
            .pipe(add_reference_counts, ref=cfg.substitution_bases_reverse[0])
            # # error rates forward
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_forward[0],
                obs=cfg.substitution_bases_forward[1],
            )
            # # error rates reverse
            .pipe(
                add_error_rates,
                ref=cfg.substitution_bases_reverse[0],
                obs=cfg.substitution_bases_reverse[1],
            )
            # add other information
            .pipe(make_position_1_indexed)
            .pipe(make_reverse_position_negative)
            .pipe(replace_nans_with_zeroes)
            .pipe(add_k_sum_counts, cfg=cfg)
            .pipe(add_min_N_in_group, cfg=cfg)
            .pipe(filter_cut_based_on_cfg, cfg)
            # turns dask dataframe into pandas dataframe
            .compute()
            # .pipe(remove_tax_ids_with_too_few_k_sum_total, cfg)
            # .pipe(cut_NANs_away)  # remove any tax_ids containing nans
            .reset_index(drop=True)
            .pipe(sort_by_alignments)
            .reset_index(drop=True)
        )

    # client.shutdown()
    # cluster.close()
    clean_up_after_dask()

    df["shortname"] = cfg.shortname
    categories = ["tax_id", "tax_name", "tax_rank", "strand", "shortname"]
    df2 = utils.downcast_dataframe(df, categories, fully_automatic=False)
    return df2


def load_counts(cfg):

    # reload(io)
    parquet = io.Parquet(cfg.filename_counts)

    if parquet.exists(cfg.forced):

        metadata_file = parquet.load_metadata()
        metadata_cfg = cfg.to_dict()

        include = [
            "min_alignments",
            "min_k_sum",
            "min_N_at_each_pos",
            "substitution_bases_forward",
            "substitution_bases_reverse",
            "shortname",
            "filename",
        ]

        if utils.metadata_is_similar(metadata_file, metadata_cfg, include=include):
            logger.info(f"Loading DataFrame from parquet-file.")
            df_counts = parquet.load()
            cfg.set_number_of_fits(df_counts)
            return df_counts

    logger.info(f"Creating DataFrame, please wait.")
    df_counts = compute_counts_with_dask(cfg, use_processes=True)
    parquet.save(df_counts, metadata=cfg.to_dict())

    cfg.set_number_of_fits(df_counts)
    return df_counts
