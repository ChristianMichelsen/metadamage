# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from jax.scipy.special import logsumexp
from scipy.special import logsumexp

# Standard Library
import logging
from multiprocessing import current_process, Manager, Pool, Process, Queue
import time
import warnings

# Third Party
import jax
import jax.numpy as jnp
from jax.random import PRNGKey as Key
from joblib import delayed, Parallel
import numpyro
from numpyro import distributions as dist
from numpyro.infer import log_likelihood, MCMC, NUTS, Predictive
import timeout_decorator
from timeout_decorator import TimeoutError
from tqdm.auto import tqdm

# First Party
from metadamage import counts, fits_Bayesian, fits_frequentist, io, utils
from metadamage.progressbar import progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%

timeout_first_fit = 5 * 60  # 5 minutes, very first fit
timeout_subsequent_fits = 60  # 1 minute

#%%


def group_to_numpyro_data(group, cfg):

    forward = cfg.substitution_bases_forward
    forward_ref = forward[0]
    reverse = cfg.substitution_bases_reverse
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


def add_tax_information(fit_result, group):
    fit_result["tax_id"] = group["tax_id"].iloc[0]
    # fit_result["tax_name"] = group["tax_name"].iloc[0]
    # fit_result["tax_rank"] = group["tax_rank"].iloc[0]


def add_count_information(fit_result, group, data):
    fit_result["N_alignments"] = group.N_alignments.iloc[0]

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
    group,
    cfg,
    mcmc_PMD=None,
    mcmc_null=None,
):

    fit_result = {}

    data = group_to_numpyro_data(group, cfg)

    add_tax_information(fit_result, group)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        f, f_forward, f_reverse = fits_frequentist.make_fits(fit_result, data)

    add_count_information(fit_result, group, data)

    if mcmc_PMD is not None and mcmc_null is not None:
        fits_Bayesian.make_fits(fit_result, data, mcmc_PMD, mcmc_null)

    return fit_result


def get_fit_single_group_with_timeout(timeout=60):
    """ timeout in seconds """
    return timeout_decorator.timeout(timeout)(fit_single_group_without_timeout)


def compute_fits_seriel(df_counts, cfg):

    # initializez not MCMC if cfg.bayesian is False
    mcmc_PMD, mcmc_null = fits_Bayesian.init_mcmcs(cfg)

    groupby = df_counts.groupby("tax_id", sort=False, observed=True)

    d_fit_results = {}

    fit_single_group_first_fit = get_fit_single_group_with_timeout(timeout_first_fit)
    fit_single_group_subsequent_fits = get_fit_single_group_with_timeout(
        timeout_subsequent_fits
    )

    fit_single_group = fit_single_group_first_fit

    logger.info(f"Fit: Initializing fit in seriel.")

    task_fit = progress.add_task(
        "task_status_fitting",
        progress_type="status",
        status="Fitting",
        name="Fits: ",
        total=len(groupby),
    )

    for tax_id, group in groupby:
        # break

        try:
            fit_result = fit_single_group(group, cfg, mcmc_PMD, mcmc_null)
            d_fit_results[tax_id] = fit_result

        except TimeoutError:
            logger.warning(f"Fit: Timeout at tax_id {tax_id}. Skipping for now")

        progress.advance(task_fit)
        fit_single_group = fit_single_group_subsequent_fits

    return d_fit_results


def worker(queue_in, queue_out, cfg):

    # initializez not MCMC if cfg.bayesian is False
    mcmc_PMD, mcmc_null = fits_Bayesian.init_mcmcs(cfg)

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
            fit_result = fit_single_group(group, cfg, mcmc_PMD, mcmc_null)
            queue_out.put((tax_id, fit_result))

        except TimeoutError:
            queue_out.put((tax_id, TimeoutError))

        fit_single_group = fit_single_group_subsequent_fits


def compute_fits_parallel_with_progressbar(df, cfg):

    # logger.info(f"Fit: Initializing fit in parallel with progressbar")

    groupby = df.groupby("tax_id", sort=False, observed=True)
    N_groupby = len(groupby)

    N_cores = cfg.N_cores if cfg.N_cores < N_groupby else N_groupby

    manager = Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()
    the_pool = Pool(N_cores, worker, (queue_in, queue_out, cfg))

    d_fit_results = {}
    task_fit = progress.add_task(
        "task_status_fitting",
        progress_type="status",
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

    # prevent adding anything more to the queue and wait for queue to empty
    # queue_in.close()
    # queue_in.join_thread()

    # # join the queue until we're finished processing results
    # queue_out.join()
    # # not closing the Queues caused me untold heartache and suffering
    # queue_in.close()
    # queue_out.close()

    # prevent adding anything more to the process pool and wait for all processes to finish
    the_pool.close()
    the_pool.join()

    return d_fit_results


#%%


# def make_df_fit_predictions_from_d_fits(d_fits, cfg):

#     z = np.arange(15) + 1
#     position = np.concatenate([z, -z])

#     # d_fit_predictions = {}
#     d_fit_predictions = []
#     for key, d_val in d_fits.items():

#         data = {
#             "tax_id": key,
#             "position": position,
#             "mean": d_val["mean"],
#             "std": d_val["std"],
#             # "median": median,
#             # "hdpi_lower": hpdi[0, :],
#             # "hdpi_upper": hpdi[1, :],
#         }

#         df_tmp = pd.DataFrame(data=data)
#         # d_fit_predictions[key] = df_tmp
#         d_fit_predictions.append(df_tmp)

#     df_fit_predictions = pd.concat(d_fit_predictions, axis="index", ignore_index=True)
#     df_fit_predictions["shortname"] = cfg.shortname

#     categories = ["tax_id", "shortname"]
#     df_fit_predictions = utils.downcast_dataframe(
#         df_fit_predictions, categories, fully_automatic=False
#     )

#     return df_fit_predictions


def match_tax_id_order_in_df_fit_results(df_fit_results, df):
    tax_ids_all = pd.unique(df.tax_id)
    ordered = [tax_id for tax_id in tax_ids_all if tax_id in df_fit_results.index]
    return df_fit_results.loc[ordered]


def make_df_fit_results_from_fit_results(fit_results, df_counts, cfg):
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
    df_fit_results = match_tax_id_order_in_df_fit_results(df_fit_results, df_counts)
    df_fit_results["shortname"] = cfg.shortname

    # categories = ["tax_id", "tax_name", "tax_rank", "shortname"]
    categories = ["tax_id", "shortname"]
    df_fit_results = utils.downcast_dataframe(
        df_fit_results, categories, fully_automatic=False
    )

    df_fit_results = df_fit_results.reset_index(drop=True)

    return df_fit_results


#%%


def get_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def compute_fits_parallel_with_progressbar_chunks(df, cfg, chunk_max=1000):
    logger.info(
        f"Fit: Initializing fit in parallel with progressbar "
        f"in chunks of size {chunk_max}."
    )

    d_fits_all_chunks = {}
    tax_ids_unique = np.array(pd.unique(df.tax_id))
    chunks = get_chunks(tax_ids_unique, chunk_max)
    for chunk in chunks:
        d_fits_chunk = compute_fits_parallel_with_progressbar(
            df.query("tax_id in @chunk"), cfg
        )
        d_fits_all_chunks.update(d_fits_chunk)
    return d_fits_all_chunks


def compute_fits(df_counts, cfg):

    if cfg.N_cores == 1:  #  or len(groupby) < 10:
        d_fit_results = compute_fits_seriel(df_counts, cfg)

    else:

        if not cfg.bayesian:
            d_fit_results = compute_fits_parallel_with_progressbar(df_counts, cfg)

        else:
            d_fit_results = compute_fits_parallel_with_progressbar_chunks(
                df_counts, cfg, chunk_max=1000
            )

    df_fit_results = make_df_fit_results_from_fit_results(d_fit_results, df_counts, cfg)
    return df_fit_results


#%%


def extract_top_max_fits(df_counts, max_fits):
    top_max_fits = (
        df_counts.groupby("tax_id", observed=True)["N_alignments"]
        .sum()
        .nlargest(max_fits)
        .index
    )
    df_counts_top_N = df_counts.query("tax_id in @top_max_fits")
    return df_counts_top_N


def get_top_max_fits(df_counts, N_fits):
    if N_fits is not None and N_fits > 0:
        return df_counts.pipe(extract_top_max_fits, N_fits)
    else:
        return df_counts


def get_fits(df_counts, cfg):

    parquet_fit_results = io.Parquet(cfg.filename_fit_results)
    # parquet_fit_predictions = io.Parquet(cfg.filename_fit_predictions)

    if parquet_fit_results.exists(cfg.forced):
        #  and parquet_fit_predictions.exists(cfg.forced)

        include = [
            "min_alignments",
            "min_k_sum",
            "substitution_bases_forward",
            "substitution_bases_reverse",
            "N_fits",
            "shortname",
            # "filename",
        ]

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_fit_results.load_metadata()
        # metadata_file_fit_predictions = parquet_fit_predictions.load_metadata()

        if utils.metadata_is_similar(
            metadata_file_fit_results, metadata_cfg, include=include
        ):
            # and utils.metadata_is_similar(metadata_file_fit_predictions, ...)

            logger.info(f"Fit: Loading fits from parquet-file.")
            df_fit_results = parquet_fit_results.load()
            # df_fit_predictions = parquet_fit_predictions.load()
            return df_fit_results  # , df_fit_predictions

    logger.info(f"Fit: Generating fits and saving to file.")

    df_counts_top_N = get_top_max_fits(df_counts, cfg.N_fits)

    # df = df_counts = df_counts_top_N
    df_fit_results = compute_fits(df_counts_top_N, cfg)  # df_fit_predictions

    parquet_fit_results.save(df_fit_results, metadata=cfg.to_dict())
    # parquet_fit_predictions.save(df_fit_predictions, metadata=cfg.to_dict())

    return df_fit_results  # , df_fit_predictions


#%%


# import arviz as az

# data_no_k = filter_out_k(data)

# def get_InferenceData(mcmc, model):

#     posterior_samples = mcmc.get_samples()
#     posterior_predictive = Predictive(model, posterior_samples)(Key(1), **data_no_k)
#     prior = Predictive(model, num_samples=500)(Key(2), **data_no_k)

#     numpyro_data = az.from_numpyro(
#         mcmc,
#         prior=prior,
#         posterior_predictive=posterior_predictive,
#         # coords={"school": np.arange(eight_school_data["J"])},
#         # dims={"theta": ["school"]},
#     )

#     return numpyro_data

# data_PMD = get_InferenceData(mcmc_PMD, model_PMD)
# data_null = get_InferenceData(mcmc_null, model_null)

# var_names = ["A", "D_max", "q", "c", "phi"]

# az.plot_trace(data_PMD, var_names=var_names)
# az.plot_dist_comparison(data_PMD, var_names=var_names)
# az.plot_posterior(data_PMD, var_names=var_names)

# model_compare = az.compare({"PMD": data_PMD, "Null": data_null}, ic="waic", scale='deviance')

# model_compare[['rank', 'waic', 'd_waic', 'dse']]

# az.plot_compare(model_compare, insample_dev=False)
