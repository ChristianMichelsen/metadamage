from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from scipy.stats import betabinom as sp_betabinom
from tqdm.auto import tqdm

import metadamage as meta
from metadamage.progressbar import progress


#%%

logger = logging.getLogger(__name__)

#%%


def split(strng, sep, pos):
    strng = strng.split(sep)
    return sep.join(strng[:pos]), sep.join(strng[pos:])


def get_number_of_lines(filename):
    with open(filename, "r") as f:
        counter = 0
        for _ in f:
            counter += 1
    return counter


def read_LCA_file(filename_LCA, use_only_these_taxids=None):

    N_lines = get_number_of_lines(filename_LCA)

    d_combined = {}
    with open(filename_LCA, "r") as f:

        task_LCA = progress.add_task(
            "task_LCA",
            progress_type="LCA",
            status="LCA    ",
            total=N_lines + 1,
        )

        for irow, row in enumerate(f):
            progress.advance(task_LCA)

            if irow == 0:
                continue

            read_id, rest = split(row.strip(), sep=":", pos=7)

            numerical, lca = split(rest, sep="\t", pos=1)

            tax_id = int(lca.split(":")[0])

            # tax_name = lca.split("\t")[0].split(":")[1]
            # tax_rank = lca.split("\t")[0].split(":")[2].strip('"')

            # remove seq
            numericals = numerical.split(":")[1:]

            combined = (
                [read_id]
                + [
                    tax_id,
                    # tax_name,
                    # tax_rank,
                ]
                + numericals
                + [lca.strip()]
            )
            d_combined[irow] = combined

    return d_combined, task_LCA


def compute_df_mismatches_wide(df_mismatches):

    max_pos = df_mismatches.position.max()

    d_direction = {
        "forward": {
            "query": "position > 0",
            "symbol": "+",
        },
        "reverse": {
            "query": "position < 0",
            "symbol": "-",
        },
    }

    df_mismatches_wide = []
    for direction in ["forward", "reverse"]:
        for variable in ["k", "N", "f"]:
            col_names = [
                f"{variable}{d_direction[direction]['symbol']}{i}"
                for i in range(1, max_pos + 1)
            ]
            columns = {i + 1: col for i, col in enumerate(col_names)}

            df_mismatches_wide.append(
                df_mismatches.query(d_direction[direction]["query"])
                .pivot(index="tax_id", columns="|z|", values=variable)
                .rename(columns=columns)
            )

    df_mismatches_wide = pd.concat(df_mismatches_wide, axis="columns")

    return df_mismatches_wide


def summarize_reads(df_LCA_read, df_mismatches_wide):

    df_LCA = (
        df_LCA_read.groupby("tax_id")
        .first()
        .drop(columns=["read_id", "read_L", "read_alignments", "read_GC"])
    )

    # get tax_id as column and not index
    df_LCA = df_LCA.reset_index()

    mismatch_counts_columns = list(
        filter(lambda s: not s.startswith("f"), df_mismatches_wide.columns)
    )

    dtypes_non_float = {
        "tax_id": "int",
        "tax_name": "str",
        "tax_rank": "str",
        "shortname": "str",
        "LCA": "str",
        "N_reads": "int",
        "N_alignments": "int",
        **{col: "int" for col in mismatch_counts_columns},
    }

    for column in df_LCA.columns:
        if column not in dtypes_non_float.keys():
            dtypes_non_float[column] = "float"

    df_LCA = df_LCA.astype(dtypes_non_float)

    return df_LCA


def compute_results(cfg, df_mismatches, df_fit_results):

    logger.info(f"Results: Loading LCA.")

    d_combined, task_LCA = read_LCA_file(cfg.filename_LCA)

    columns_lca = [
        "read_id",
        "tax_id",
        "read_L",
        "read_alignments",
        "read_GC",
        "LCA",
    ]

    dtypes_lca = {
        "tax_id": "uint32",
        "read_L": "uint32",
        "read_alignments": "uint64",
        "read_GC": "float",
    }

    df_LCA_read = pd.DataFrame.from_dict(
        d_combined,
        orient="index",
        columns=columns_lca,
    ).astype(dtypes_lca)

    #%%

    # merge fit results into the dataframe
    df_LCA_read = pd.merge(df_LCA_read, df_fit_results, on=["tax_id"])

    # merge the mismatch counts (as a wide dataframe) into the dataframe
    df_mismatches_wide = compute_df_mismatches_wide(df_mismatches)
    df_LCA_read = pd.merge(df_LCA_read, df_mismatches_wide, on=["tax_id"])

    non_categories = ["read_id", "read_L", "read_alignments", "read_GC"]
    categories = [col for col in df_LCA_read.columns if col not in non_categories]

    df_LCA_read = df_LCA_read.astype({cat: "category" for cat in categories})

    columns_order = [
        "tax_id",
        "tax_name",
        "tax_rank",
        "read_id",
        "read_L",
        "read_alignments",
        "read_GC",
        "shortname",
        "N_reads",
        "N_alignments",
        #
        "lambda_LR",
        "D_max",
        "mean_L",
        "mean_GC",
        "q",
        "A",
        "c",
        "phi",
        "rho_Ac",
        "valid",
        "asymmetry",
    ]

    columns_order += [col for col in df_fit_results.columns if not col in columns_order]
    columns_order += list(df_mismatches_wide.columns) + ["LCA"]

    df_LCA_read = df_LCA_read[columns_order]

    df_LCA = summarize_reads(df_LCA_read, df_mismatches_wide)

    progress.advance(task_LCA)
    return df_LCA, df_LCA_read


def load(cfg, df_mismatches, df_fit_results):

    parquet_results_LCA = meta.io.Parquet(cfg.filename_results)
    parquet_results_LCA_reads = meta.io.Parquet(cfg.filename_results_read)

    if parquet_results_LCA.exists(cfg.forced) and parquet_results_LCA_reads.exists(
        cfg.forced
    ):

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_results_LCA_reads.load_metadata()

        if meta.utils.metadata_is_similar(
            metadata_file_fit_results,
            metadata_cfg,
            # include=include,
        ):

            logger.info(f"Fit: Loading fits from parquet-file.")
            df_results = parquet_results_LCA.load()
            df_results_read = parquet_results_LCA_reads.load()
            return df_results, df_results_read

    logger.info(f"Fit: Generating results and saving to file.")

    df_results, df_results_read = compute_results(cfg, df_mismatches, df_fit_results)

    parquet_results_LCA.save(df_results, metadata=cfg.to_dict())
    parquet_results_LCA_reads.save(df_results_read, metadata=cfg.to_dict())

    return df_results, df_results_read
