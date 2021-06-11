from collections import defaultdict
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from scipy.stats import betabinom as sp_betabinom
from tqdm.auto import tqdm

import clitest.meta_fit as meta
from clitest.read_ids_database import DB as database
from clitest.rich import progress


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


def read_LCA_file(file_lca):

    N_lines = get_number_of_lines(file_lca)

    d_tax_id_to_read_ids = defaultdict(set)
    d_tax_id_to_LCA = dict()
    lcas = set()
    with open(file_lca, "r") as f:

        task_LCA = progress.add_task(
            "task_LCA",
            progress_type="unknown_total",
            status="Merging",
            total=N_lines + 1,
        )

        for irow, row in enumerate(f):
            progress.advance(task_LCA)

            if irow == 0:
                continue

            # break

            read_id, rest = split(row.strip(), sep=":", pos=7)

            seq_numerical, lca = split(rest, sep="\t", pos=1)

            tax_id = int(lca.split(":")[0])

            # tax_name = lca.split("\t")[0].split(":")[1]
            # tax_rank = lca.split("\t")[0].split(":")[2].strip('"')

            d_tax_id_to_read_ids[tax_id].add(read_id)
            d_tax_id_to_LCA[tax_id] = lca
            lcas.add(lca)

            # if keep_read_length_alignments_GC:

            # # remove seq
            # numericals = seq_numerical.split(":")[1:]
            # numericals[0] = int(numericals[0])
            # numericals[1] = int(numericals[1])
            # numericals[2] = float(numericals[2])

            # combined = (
            #     [read_id]
            #     + [
            #         tax_id,
            #         # tax_name,
            #         # tax_rank,
            #     ]
            #     + numericals
            #     + [lca.strip()]
            # )
            # d_combined[irow] = combined

    d_tax_id_to_read_ids = dict(d_tax_id_to_read_ids)

    for key, val in d_tax_id_to_read_ids.items():
        d_tax_id_to_read_ids[key] = list(val)

    return d_tax_id_to_read_ids, d_tax_id_to_LCA, task_LCA


def compute_df_mismatch_wide(df_mismatch):

    max_pos = df_mismatch.position.max()

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

    df_mismatch_wide = []
    for direction in ["forward", "reverse"]:
        for variable in ["k", "N", "f"]:
            col_names = [
                f"{variable}{d_direction[direction]['symbol']}{i}"
                for i in range(1, max_pos + 1)
            ]
            columns = {i + 1: col for i, col in enumerate(col_names)}

            df_mismatch_wide.append(
                df_mismatch.query(d_direction[direction]["query"])
                .pivot(index="tax_id", columns="|z|", values=variable)
                .rename(columns=columns)
            )

    df_mismatch_wide = pd.concat(df_mismatch_wide, axis="columns")

    return df_mismatch_wide


# def summarize_reads(df_LCA_read, df_mismatch_wide):

#     df_LCA = (
#         df_LCA_read.groupby("tax_id")
#         .first()
#         .drop(columns=["read_id", "read_L", "read_alignments", "read_GC"])
#     )

#     # get tax_id as column and not index
#     df_LCA = df_LCA.reset_index()

#     mismatch_counts_columns = list(
#         filter(lambda s: not s.startswith("f"), df_mismatch_wide.columns)
#     )

#     dtypes_non_float = {
#         "tax_id": "int",
#         "tax_name": "str",
#         "tax_rank": "str",
#         "shortname": "str",
#         "LCA": "str",
#         "N_reads": "int",
#         "N_alignments": "int",
#         **{col: "int" for col in mismatch_counts_columns},
#     }

#     for column in df_LCA.columns:
#         if column not in dtypes_non_float.keys():
#             dtypes_non_float[column] = "float"

#     df_LCA = df_LCA.astype(dtypes_non_float)

#     return df_LCA


# from tinydb import TinyDB, where
# from tinydb.table import Document

# d_small = {}
# for key, val in d_tax_id_to_read_ids.items():
#     d_small[key] = val[:10]


# db_name = 'db.json'
# table_name = 'read_ids'
# db = TinyDB(db_name).table(table_name)

# for tax_id, read_ids in d_tax_id_to_read_ids.items():
#     db.insert(Document({"read_ids": read_ids}, doc_id=tax_id))

# db = TinyDB(db_name).table(table_name)
# %timeit len(db.get(doc_id=9597)["read_ids"])


def compute_results(cfg, df_mismatch, df_fit_results):

    logger.info(f"Results: Loading LCA.")

    d_tax_id_to_read_ids, d_tax_id_to_LCA, task_LCA = read_LCA_file(cfg.file_lca)

    df_tax_id_LCA = (
        pd.DataFrame.from_dict(
            d_tax_id_to_LCA,
            orient="index",
            columns=["LCA"],
        )
        .reset_index()
        .rename(columns={"index": "tax_id"})
    )

    #%%

    # merge fit results into the dataframe
    df_results = pd.merge(df_fit_results, df_tax_id_LCA, on=["tax_id"])

    # merge the mismatch counts (as a wide dataframe) into the dataframe
    df_mismatch_wide = compute_df_mismatch_wide(df_mismatch)
    df_results = pd.merge(df_results, df_mismatch_wide, on=["tax_id"])

    columns_order = [
        "tax_id",
        "tax_name",
        "tax_rank",
        # "read_id",
        # "read_L",
        # "read_alignments",
        # "read_GC",
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
    columns_order += list(df_mismatch_wide.columns) + ["LCA"]

    df_results = df_results[columns_order]

    # import joblib

    # joblib.dump(d_tax_id_to_read_ids, "db.joblib")

    db = database(cfg)
    db.save(d_tax_id_to_read_ids)

    progress.advance(task_LCA)
    return df_results


def load(cfg, df_mismatch, df_fit_results):

    parquet_results = meta.io.Parquet(cfg.file_result)

    if parquet_results.exists(cfg.forced):

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_results.load_metadata()

        include = ["shortname"]

        if meta.utils.metadata_is_similar(
            metadata_file_fit_results,
            metadata_cfg,
            include=include,
        ):

            logger.info(f"Loading results from parquet-file.")
            df_results = parquet_results.load()
            return df_results

    logger.info(f"Generating results and saving to file.")

    df_results = compute_results(cfg, df_mismatch, df_fit_results)

    parquet_results.save(df_results, metadata=cfg.to_dict())

    return df_results
