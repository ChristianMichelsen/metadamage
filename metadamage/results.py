import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from copy import deepcopy
import logging


# First Party
from metadamage import counts, fits_Bayesian, fits_frequentist, io, utils
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


def read_LCA_file(filename_LCA, use_only_these_taxids=None, use_tqdm=False):

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

            if use_only_these_taxids is not None:
                if not tax_id in use_only_these_taxids:
                    continue

            tax_name = lca.split("\t")[0].split(":")[1]
            tax_rank = lca.split("\t")[0].split(":")[2].strip('"')

            numericals = numerical.split(":")[1:]  # remove seq

            combined = (
                [read_id] + [tax_id, tax_name, tax_rank] + numericals + [lca.strip()]
            )
            d_combined[irow] = combined

    return d_combined, task_LCA


def compute_results(cfg, df_counts, df_fit_results):

    columns = [
        "read_id",
        "tax_id",
        "tax_name",
        "tax_rank",
        "seq_length_read",
        "alignments_read",
        "GC_read",
        "LCA",
    ]

    dtypes = {
        "tax_id": "uint32",
        "seq_length_read": "uint32",
        "alignments_read": "uint64",
        "GC_read": "float",
    }

    use_only_these_taxids = None
    use_only_these_taxids = set(df_fit_results.tax_id.unique())

    logger.info(f"Results: Loading LCA.")

    d_combined, task_LCA = read_LCA_file(
        cfg.filename_LCA,
        use_only_these_taxids=use_only_these_taxids,
    )

    df = pd.DataFrame.from_dict(
        d_combined,
        orient="index",
        columns=columns,
    ).astype(dtypes)

    df_grouped = df.groupby("tax_id")

    df["N_reads"] = df_grouped["read_id"].transform(len)
    df["N_alignments"] = df_grouped["alignments_read"].transform(np.sum)

    df["seq_length_mean"] = df_grouped["seq_length_read"].transform(np.mean)
    df["seq_length_std"] = df_grouped["seq_length_read"].transform(np.std)

    df["seq_length_1%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 1)
    )
    df["seq_length_25%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 25)
    )
    df["seq_length_median"] = df_grouped["seq_length_read"].transform(np.median)
    df["seq_length_75%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 75)
    )
    df["seq_length_99%"] = df_grouped["seq_length_read"].transform(
        lambda x: np.percentile(x, 99)
    )

    df["gc_mean"] = df_grouped["GC_read"].transform(np.mean)
    df["gc_std"] = df_grouped["GC_read"].transform(np.std)

    df["gc_1%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 1))
    df["gc_25%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 25))
    df["gc_median"] = df_grouped["GC_read"].transform(np.median)
    df["gc_75%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 75))
    df["gc_99%"] = df_grouped["GC_read"].transform(lambda x: np.percentile(x, 99))

    #%%

    fit_result_columns = [
        "tax_id",
        "LR",
        "D_max",
        "q",
        "phi",
        "A",
        "c",
        "rho_Ac",
        "valid",
        "asymmetry",
    ]

    # merge LR and D_max into the dataframe
    # XXX IMPORTANT: Here we lose a lot of rows from the LCA
    df = pd.merge(df, df_fit_results[fit_result_columns], on=["tax_id"])

    max_pos = df_counts.position.max()
    CT_cols = [f"CT+{i}" for i in range(1, max_pos + 1)]
    GA_cols = [f"GA-{i}" for i in range(1, max_pos + 1)]

    # merge the CT transitions into the dataframe
    df = pd.merge(
        df,
        (
            df_counts.query("position > 0")
            .pivot(index="tax_id", columns="position", values="f_CT")
            .rename(columns={i + 1: col for i, col in enumerate(CT_cols)})
        ),
        on=["tax_id"],
    )

    df = pd.merge(
        df,
        (
            df_counts.query("position < 0")
            .pivot(index="tax_id", columns="position", values="f_GA")
            .sort_index(axis=1, ascending=False)
            .rename(columns={-(i + 1): col for i, col in enumerate(GA_cols)})
        ),
        on=["tax_id"],
    )

    non_categories = ["read_id", "seq_length_read", "alignments_read", "GC_read"]
    categories = [col for col in df.columns if col not in non_categories]

    df = df.astype({cat: "category" for cat in categories})

    columns_order = [
        "tax_id",
        "tax_name",
        "tax_rank",
        "read_id",
        "seq_length_read",
        "alignments_read",
        "GC_read",
        "N_alignments",
        "N_reads",
        "LR",
        "D_max",
        "seq_length_mean",
        "gc_mean",
        "q",
        "phi",
        "A",
        "c",
        "rho_Ac",
        "valid",
        "asymmetry",
        "seq_length_std",
        "seq_length_1%",
        "seq_length_25%",
        "seq_length_median",
        "seq_length_75%",
        "seq_length_99%",
        "gc_std",
        "gc_1%",
        "gc_25%",
        "gc_median",
        "gc_75%",
        "gc_99%",
    ]
    columns_order += CT_cols + GA_cols + ["LCA"]

    df = df[columns_order]

    df_small = (
        df.groupby("tax_id")
        .first()
        .drop(columns=["read_id", "seq_length_read", "alignments_read", "GC_read"])
    )

    dtypes_small = {
        "tax_name": "str",
        "tax_rank": "str",
        "LCA": "str",
        "N_reads": "int",
        "N_alignments": "int",
        "seq_length_median": "int",
        "seq_length_1%": "int",
        "seq_length_25%": "int",
        "seq_length_75%": "int",
        "seq_length_99%": "int",
    }

    for column in df_small.columns:
        if column not in dtypes_small.keys():
            dtypes_small[column] = "float"

    df_small = df_small.astype(dtypes_small)

    progress.advance(task_LCA)
    return df, df_small


def get_results(cfg, df_counts, df_fit_results):

    parquet_results_small = io.Parquet(cfg.filename_results_small)
    parquet_results_large = io.Parquet(cfg.filename_results_large)

    if parquet_results_large.exists(cfg.forced):

        include = [
            "min_alignments",
            "min_k_sum",
            "substitution_bases_forward",
            "substitution_bases_reverse",
            "N_fits",
            "shortname",
        ]

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_results_large.load_metadata()

        if utils.metadata_is_similar(
            metadata_file_fit_results, metadata_cfg, include=include
        ):

            logger.info(f"Fit: Loading fits from parquet-file.")
            df_results_small = parquet_results_small.load()
            df_results_large = parquet_results_large.load()
            return df_results_small, df_results_large

    logger.info(f"Fit: Generating results and saving to file.")

    df_results_large, df_results_small = compute_results(cfg, df_counts, df_fit_results)

    parquet_results_small.save(df_results_small, metadata=cfg.to_dict())
    parquet_results_large.save(df_results_large, metadata=cfg.to_dict())

    return df_results_large, df_results_small
