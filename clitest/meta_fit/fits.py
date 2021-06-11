from collections import defaultdict
import logging
from multiprocessing import current_process, Manager, Pool, Process, Queue
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timeout_decorator
from timeout_decorator import TimeoutError

from clitest import meta_fit
from clitest.rich import progress


logger = logging.getLogger(__name__)

#%%

timeout_first_fit = 5 * 60  # 5 minutes, very first fit
timeout_subsequent_fits = 60  # 1 minute

#%%


def get_groupby(df_mismatch):
    return df_mismatch.groupby("tax_id", sort=False, observed=True)


def group_to_numpyro_data(cfg, group):

    forward = "CT"
    forward_ref = forward[0]
    reverse = "GA"
    reverse_ref = reverse[0]

    z = np.array(group.iloc[:15]["position"], dtype=int)

    k_forward = np.array(group.iloc[:15][forward], dtype=int)
    N_forward = np.array(group.iloc[:15][forward_ref], dtype=int)

    k_reverse = np.array(group.iloc[-15:][reverse], dtype=int)
    N_reverse = np.array(group.iloc[-15:][reverse_ref], dtype=int)

    data = {
        "z": np.concatenate([z, -z]),
        "k": np.concatenate([k_forward, k_reverse]),
        "N": np.concatenate([N_forward, N_reverse]),
    }

    return data


#%%


def add_count_information(fit_result, group, data):
    fit_result["N_z1_forward"] = data["N"][0]
    fit_result["N_z1_reverse"] = data["N"][15]

    fit_result["N_sum_total"] = data["N"].sum()
    fit_result["N_sum_forward"] = data["N"][:15].sum()
    fit_result["N_sum_reverse"] = data["N"][15:].sum()

    fit_result["N_min"] = data["N"].min()

    fit_result["k_sum_total"] = data["k"].sum()
    fit_result["k_sum_forward"] = data["k"][:15].sum()
    fit_result["k_sum_reverse"] = data["k"][15:].sum()


#%%


def fit_single_group_without_timeout(
    cfg,
    group,
    mcmc_PMD=None,
    mcmc_null=None,
):

    fit_result = {}

    data = group_to_numpyro_data(cfg, group)

    # add_tax_information(fit_result, group)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        f, f_forward, f_reverse = meta_fit.fits_frequentist.make_fits(fit_result, data)

    add_count_information(fit_result, group, data)

    if mcmc_PMD is not None and mcmc_null is not None:
        meta_fit.fits_Bayesian.make_fits(fit_result, data, mcmc_PMD, mcmc_null)

    return fit_result


def get_fit_single_group_with_timeout(timeout=60):
    """timeout in seconds"""
    return timeout_decorator.timeout(timeout)(fit_single_group_without_timeout)


def compute_fits_seriel(cfg, df_mismatch):

    # initializez not MCMC if cfg.bayesian is False
    mcmc_PMD, mcmc_null = meta_fit.fits_Bayesian.init_mcmcs(cfg)

    groupby = get_groupby(df_mismatch)

    d_fit_results = {}

    fit_single_group_first_fit = get_fit_single_group_with_timeout(timeout_first_fit)
    fit_single_group_subsequent_fits = get_fit_single_group_with_timeout(
        timeout_subsequent_fits
    )

    fit_single_group = fit_single_group_first_fit

    logger.info(f"Fit: Initializing fit in seriel.")

    task_fit = progress.add_task(
        "task_status_fitting",
        progress_type="known_total",
        status="Fitting",
        name="Fits: ",
        total=len(groupby),
    )

    for tax_id, group in groupby:
        # break

        try:
            fit_result = fit_single_group(cfg, group, mcmc_PMD, mcmc_null)
            d_fit_results[tax_id] = fit_result

        except TimeoutError:
            logger.warning(f"Fit: Timeout at tax_id {tax_id}. Skipping for now")

        progress.advance(task_fit)
        fit_single_group = fit_single_group_subsequent_fits

    return d_fit_results


def worker(cfg, queue_in, queue_out):

    # initializez not MCMC if cfg.bayesian is False
    mcmc_PMD, mcmc_null = meta_fit.fits_Bayesian.init_mcmcs(cfg)

    fit_single_group_first_fit = get_fit_single_group_with_timeout(timeout_first_fit)
    fit_single_group_subsequent_fits = get_fit_single_group_with_timeout(
        timeout_subsequent_fits
    )

    # first run is patient
    fit_single_group = fit_single_group_first_fit

    while True:
        # block=True means make a blocking call to wait for items in queue
        tax_id_group = queue_in.get(block=True)
        if tax_id_group is None:
            break
        tax_id, group = tax_id_group

        try:
            fit_result = fit_single_group(cfg, group, mcmc_PMD, mcmc_null)
            queue_out.put((tax_id, fit_result))

        except TimeoutError:
            queue_out.put((tax_id, TimeoutError))

        fit_single_group = fit_single_group_subsequent_fits


def compute_fits_parallel_with_progressbar(cfg, df_mismatch):

    # logger.info(f"Fit: Initializing fit in parallel with progressbar")

    groupby = get_groupby(df_mismatch)
    N_groupby = len(groupby)

    N_cores = cfg.N_cores if cfg.N_cores < N_groupby else N_groupby

    manager = Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()
    the_pool = Pool(N_cores, worker, (cfg, queue_in, queue_out))

    d_fit_results = {}
    task_fit = progress.add_task(
        "task_status_fitting",
        progress_type="known_total",
        status="Fitting",
        name="Fits: ",
        total=N_groupby,
    )

    for tax_id, group in groupby:
        queue_in.put((tax_id, group))

    # Get and print results
    for _ in range(N_groupby):
        tax_id, fit_result = queue_out.get()
        if fit_result is not TimeoutError:
            d_fit_results[tax_id] = fit_result
        else:
            logger.warning(f"Fit: Timeout at tax_id {tax_id}. Skipping for now")
        progress.advance(task_fit)

    for _ in range(N_groupby):
        queue_in.put(None)

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()

    return d_fit_results


#%%


def match_tax_id_order_in_df_fit_results(df_fit_results, df_mismatch):
    tax_ids_all = pd.unique(df_mismatch.tax_id)
    ordered = [tax_id for tax_id in tax_ids_all if tax_id in df_fit_results.index]
    return df_fit_results.loc[ordered]


def move_column_inplace(df, col, pos=0):
    col = df.pop(col)
    df.insert(pos, col.name, col)


def make_df_fit_results_from_fit_results(cfg, d_fit_results, df_mismatch):
    df_fit_results = pd.DataFrame.from_dict(d_fit_results, orient="index")
    df_fit_results["tax_id"] = df_fit_results.index
    # move_column_inplace(df_fit_results, "tax_id", pos=0)

    df_fit_results = match_tax_id_order_in_df_fit_results(df_fit_results, df_mismatch)
    df_fit_results["shortname"] = cfg.shortname

    # categories = ["tax_id", "shortname"]
    categories = []
    df_fit_results = meta_fit.utils.downcast_dataframe(
        df_fit_results, categories, fully_automatic=False
    )

    df_fit_results = df_fit_results.reset_index(drop=True)

    return df_fit_results


#%%


def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_fits_parallel_with_progressbar_chunks(cfg, df_mismatch, chunk_max=1000):
    logger.info(
        f"Fit: Initializing fit in parallel with progressbar "
        f"in chunks of size {chunk_max}."
    )

    d_fits_all_chunks = {}
    tax_ids_unique = np.array(pd.unique(df_mismatch.tax_id))
    chunks = get_chunks(tax_ids_unique, chunk_max)
    for chunk in chunks:
        d_fits_chunk = compute_fits_parallel_with_progressbar(
            cfg,
            df_mismatch.query("tax_id in @chunk"),
        )
        d_fits_all_chunks.update(d_fits_chunk)
    return d_fits_all_chunks


def compute_duplicates(df_mismatch):

    groupby = get_groupby(df_mismatch)

    duplicate_entries = defaultdict(list)
    for group in groupby:
        key = joblib.hash(group[1][meta_fit.utils.ref_obs_bases].values)
        duplicate_entries[key].append(group[0])
    duplicate_entries = dict(duplicate_entries)

    unique = [tax_ids[0] for tax_ids in duplicate_entries.values()]
    duplicates = {tax_ids[0]: tax_ids[1:] for tax_ids in duplicate_entries.values()}

    return unique, duplicates


def de_duplicate_fit_results(d_fit_results, duplicates):
    for tax_id_unique, tax_ids_non_unique in duplicates.items():
        for tax_id_non_unique in tax_ids_non_unique:
            d_fit_results[tax_id_non_unique] = d_fit_results[tax_id_unique]


def compute_fits(cfg, df_mismatch):

    # find unique tax_ids (when compairing the mismatch matrices)
    # such that only those are fitted
    unique, duplicates = compute_duplicates(df_mismatch)

    df_mismatch_unique = df_mismatch.query("tax_id in @unique")

    if cfg.N_cores == 1:  #  or len(groupby) < 10:
        d_fit_results = compute_fits_seriel(cfg, df_mismatch_unique)

    else:

        if not cfg.bayesian:
            d_fit_results = compute_fits_parallel_with_progressbar(
                cfg,
                df_mismatch_unique,
            )

        else:
            d_fit_results = compute_fits_parallel_with_progressbar_chunks(
                cfg,
                df_mismatch_unique,
                chunk_max=1000,
            )

    de_duplicate_fit_results(d_fit_results, duplicates)

    df_fit_results = make_df_fit_results_from_fit_results(
        cfg,
        d_fit_results,
        df_mismatch,
    )

    df_stats = read_stats(cfg)

    df_fit_results = pd.merge(df_fit_results, df_stats, on="tax_id")

    cols_ordered = [
        "tax_id",
        "tax_name",
        "tax_rank",
        "N_reads",
        "N_alignments",
        "lambda_LR",
        "D_max",
        "mean_L",
        "std_L",
        "mean_GC",
        "std_GC",
        *df_fit_results.loc[:, "D_max_std":"shortname"].columns.drop("tax_id"),
    ]

    df_fit_results = df_fit_results[cols_ordered]

    return df_fit_results


def read_stats(cfg):

    d_rename = {
        "#taxid": "tax_id",
        "name": "tax_name",
        "rank": "tax_rank",
        "nalign": "N_alignments",
        "nreads": "N_reads",
        "mean_rlen": "mean_L",
        "var_rlen": "var_L",
        "mean_gc": "mean_GC",
        "var_gc": "var_GC",
    }
    # fmt: off
    df_stats = (pd.read_csv(cfg.file_mismatch_stat, sep="\t")
                  .rename(columns=d_rename)
                )
    # fmt: on
    df_stats["std_L"] = np.sqrt(df_stats["var_L"])
    df_stats["std_GC"] = np.sqrt(df_stats["var_GC"])
    return df_stats


def load(cfg, df_mismatch):

    """
    Computes fits for df_mismatch. If fits are already computed, just load them.
    """

    parquet_fit_results = meta_fit.io.Parquet(cfg.file_fit_result)

    if parquet_fit_results.exists(cfg.forced):

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_fit_results.load_metadata()

        include = ["shortname"]

        if meta_fit.utils.metadata_is_similar(
            metadata_file_fit_results,
            metadata_cfg,
            include=include,
        ):

            logger.info(f"Fit: Loading fits from parquet-file.")
            df_fit_results = parquet_fit_results.load()
            return df_fit_results

    logger.info(f"Fit: Generating fits and saving to file.")

    # df_mismatch_top_N = get_top_max_fits(cfg, df_mismatch.N_fits)

    df_fit_results = compute_fits(cfg, df_mismatch)

    parquet_fit_results.save(df_fit_results, metadata=cfg.to_dict())

    return df_fit_results
