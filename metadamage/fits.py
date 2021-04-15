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
from metadamage import counts, io, utils, fits_frequentist
from metadamage.progressbar import console, progress


numpyro.enable_x64()
logger = logging.getLogger(__name__)

#%%

timeout_first_fit = 5 * 60  # 5 minutes, very first fit
timeout_subsequent_fits = 60  # 1 minute


# beta dist (alpha, beta)
q_prior = (1, 4)  # mean = 0.2, shape = 5
A_prior = (1, 4)  # mean = 0.2, shape = 5
c_prior = (1.0, 9.0)  # mean = 0.1, shape = 10

# exponential dist (loc, scale)
phi_prior = (2, 1000)

#%%


def model_PMD(z, N, y=None):
    z = jnp.abs(z)

    q = numpyro.sample("q", dist.Beta(q_prior[0], q_prior[1]))
    A = numpyro.sample("A", dist.Beta(A_prior[0], A_prior[1]))
    c = numpyro.sample("c", dist.Beta(c_prior[0], c_prior[1]))
    # Dz = numpyro.deterministic("Dz", A * (1 - q) ** (z - 1) + c)
    Dz = jnp.clip(numpyro.deterministic("Dz", A * (1 - q) ** (z - 1) + c), 0, 1)
    D_max = numpyro.deterministic("D_max", A + c)  # pylint: disable=unused-variable

    delta = numpyro.sample("delta", dist.Exponential(1 / phi_prior[1]))
    phi = numpyro.deterministic("phi", delta + phi_prior[0])

    alpha = numpyro.deterministic("alpha", Dz * phi)
    beta = numpyro.deterministic("beta", (1 - Dz) * phi)

    numpyro.sample("obs", dist.BetaBinomial(alpha, beta, N), obs=y)


def model_null(z, N, y=None):
    q = numpyro.sample("q", dist.Beta(c_prior[0], c_prior[1]))  # mean = 0.2, shape = 4
    D_max = numpyro.deterministic("D_max", q)
    delta = numpyro.sample("delta", dist.Exponential(1 / phi_prior[1]))
    phi = numpyro.deterministic("phi", delta + phi_prior[0])
    numpyro.sample("obs", dist.BetaBinomial(q * phi, (1 - q) * phi, N), obs=y)


#%%


def filter_out_y(data):
    return {key: value for key, value in data.items() if key != "y"}


def is_model_PMD(model):
    name = model.__name__.lower()
    if "pmd" in name:
        return True
    elif "null" in name:
        return False
    raise AssertionError(f"Model should be PMD or null, got {model}")


#%%


@jax.jit
def _get_posterior_PMD(rng_key, samples, *args, **kwargs):
    return Predictive(model_PMD, samples)(rng_key, *args, **kwargs)["obs"]


@jax.jit
def _get_posterior_null(rng_key, samples, *args, **kwargs):
    return Predictive(model_null, samples)(rng_key, *args, **kwargs)["obs"]


def get_posterior_predictive(mcmc, data):
    posterior_samples = mcmc.get_samples()
    rng_key = Key(0)
    data_no_y = filter_out_y(data)
    if is_model_PMD(mcmc.sampler.model):
        return _get_posterior_PMD(rng_key, posterior_samples, **data_no_y)
    else:
        return _get_posterior_null(rng_key, posterior_samples, **data_no_y)


#%%


def get_y_average_and_hpdi(mcmc, data, func=np.median, return_hpdi=True):
    """func = central tendency function, e.g. np.mean or np.median"""
    posterior_predictive = get_posterior_predictive(mcmc, data)
    predictions_fraction = posterior_predictive / data["N"]
    y_average = func(predictions_fraction, axis=0)
    if not return_hpdi:
        return y_average
    y_hpdi = numpyro.diagnostics.hpdi(predictions_fraction, prob=0.68)
    return y_average, y_hpdi


#%%


@jax.jit
def _compute_log_likelihood_PMD(posterior_samples, data):
    return log_likelihood(model_PMD, posterior_samples, **data)["obs"]


@jax.jit
def _compute_log_likelihood_null(posterior_samples, data):
    return log_likelihood(model_null, posterior_samples, **data)["obs"]


def compute_log_likelihood(mcmc, data):
    posterior_samples = mcmc.get_samples()
    if is_model_PMD(mcmc.sampler.model):
        return _compute_log_likelihood_PMD(posterior_samples, data)
    else:
        return _compute_log_likelihood_null(posterior_samples, data)


#%%


def get_lppd_and_waic(mcmc, data):
    d_results = {}
    # get the log likehood for each (point, num_samples)
    logprob = np.asarray(compute_log_likelihood(mcmc, data))
    # lppd for each observation
    lppd_i = logsumexp(logprob, 0) - np.log(logprob.shape[0])
    d_results["lppd_i"] = lppd_i
    # lppd
    lppd = lppd_i.sum()
    d_results["lppd"] = lppd
    # waic penalty for each observation
    pWAIC_i = np.var(logprob, 0)
    d_results["pWAIC_i"] = pWAIC_i
    # waic penalty # the effective number of parameters penalty
    pWAIC = pWAIC_i.sum()
    d_results["pWAIC"] = pWAIC
    # waic  for each observation
    waic_i = -2 * (lppd_i - pWAIC_i)
    d_results["waic_i"] = waic_i
    # waic # prediction of  out-of-sample deviance
    waic = waic_i.sum()
    d_results["waic"] = waic
    # standard error of waic
    # waic_vec = -2 * (lppd_i - pWAIC_i)
    # waic_uncertainty = jnp.sqrt(logprob.shape[1] * jnp.var(waic_vec))
    return d_results


def SDOM(x, axis=0):
    return np.std(x, axis=axis) / np.sqrt(x.shape[axis])


def get_mean_of_variable(mcmc, variable, axis=0):
    return np.mean(mcmc.get_samples()[variable], axis=axis).item()


def get_SDOM_of_variable(mcmc, variable, axis=0):
    return SDOM(mcmc.get_samples()[variable], axis=axis).item()


def compute_waic_weight(d_results_PMD, d_results_null):
    waic = np.asarray([d_results_PMD["waic"], d_results_null["waic"]])
    delta = waic - waic.min()
    waic_weights = np.exp(-0.5 * delta) / np.exp(-0.5 * delta).sum()
    return waic_weights


def compute_n_sigma(d_results_PMD, d_results_null):
    n = len(d_results_PMD["waic_i"])
    waic_i_PMD = d_results_PMD["waic_i"]
    waic_i_null = d_results_null["waic_i"]
    dse = np.sqrt(n * np.var(waic_i_PMD - waic_i_null))
    d_waic = d_results_null["waic"] - d_results_PMD["waic"]
    n_sigma = d_waic / dse
    return n_sigma


def compute_assymmetry_combined_vs_forwardreverse(
    d_results_PMD,
    d_results_PMD_forward,
    d_results_PMD_reverse,
):

    n = len(d_results_PMD["waic_i"])

    waic_i_combined = d_results_PMD["waic_i"]
    waic_i_forwardreverse = np.concatenate(
        [
            d_results_PMD_forward["waic_i"],
            d_results_PMD_reverse["waic_i"],
        ]
    )

    dse = np.sqrt(n * np.var(waic_i_combined - waic_i_forwardreverse))

    waic_combined = d_results_PMD["waic"]
    waic_forwardreverse = d_results_PMD_forward["waic"] + d_results_PMD_reverse["waic"]

    d_waic = waic_forwardreverse - waic_combined
    n_sigma = d_waic / dse
    return n_sigma


def compute_fit_results(
    mcmc_PMD,
    mcmc_null,
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
    data,
    group,
):

    d_results_PMD = get_lppd_and_waic(mcmc_PMD, data)
    d_results_null = get_lppd_and_waic(mcmc_null, data)

    fit_result = {}

    fit_result["tax_id"] = group["tax_id"].iloc[0]
    fit_result["tax_name"] = group["tax_name"].iloc[0]
    fit_result["tax_rank"] = group["tax_rank"].iloc[0]

    # y_frac = y/N, on the first position
    y_frac_median, y_frac_hpdi = get_y_average_and_hpdi(mcmc_PMD, data, func=np.median)
    fit_result["D_max"] = y_frac_median[0].item()

    n_sigma = compute_n_sigma(d_results_PMD, d_results_null)
    fit_result["n_sigma"] = n_sigma.item()

    # waic_weights = compute_waic_weight(d_results_PMD, d_results_null)
    # fit_result["waic_weight"] = waic_weights[0]
    # fit_result["waic_weight_null"] = waic_weights[1]

    # y_frac = y/N, on the first position, HPDIs
    fit_result["D_max_lower_hpdi"] = y_frac_hpdi[0, 0].item()
    fit_result["D_max_upper_hpdi"] = y_frac_hpdi[1, 0].item()

    # marginalized values:

    # probability of a single succes
    fit_result["q_mean"] = get_mean_of_variable(mcmc_PMD, "q")
    # concentration or shape of beta/bino
    fit_result["concentration_mean"] = get_mean_of_variable(mcmc_PMD, "phi")
    # marginalized D_max
    fit_result["D_max_marginalized_mean"] = get_mean_of_variable(mcmc_PMD, "D_max")

    fit_result["N_alignments"] = group.N_alignments.iloc[0]

    fit_result["N_z1_forward"] = data["N"][0]
    fit_result["N_z1_reverse"] = data["N"][15]

    fit_result["N_sum_forward"] = data["N"][:15].sum()
    fit_result["N_sum_reverse"] = data["N"][15:].sum()
    fit_result["N_sum_total"] = data["N"].sum()

    fit_result["y_sum_forward"] = data["y"][:15].sum()
    fit_result["y_sum_reverse"] = data["y"][15:].sum()
    fit_result["y_sum_total"] = data["y"].sum()

    add_assymetry_results_to_fit_results(
        mcmc_PMD_forward_reverse,
        mcmc_null_forward_reverse,
        data,
        fit_result,
        d_results_PMD,
    )

    add_noise_estimates(group, fit_result)

    add_frequentist_fit_results(data, fit_result)

    return fit_result


def add_assymetry_results_to_fit_results(
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
    data,
    fit_result,
    d_results_PMD,
):
    """computes the assymetry between a fit to forward data and reverse data
    the assymmetry is here defined as the n_sigma (WAIC) between the two fits
    """

    # FORWARD

    data_forward = {key: val[data["z"] > 0] for key, val in data.items()}
    fit_mcmc(mcmc_PMD_forward_reverse, data_forward)
    fit_mcmc(mcmc_null_forward_reverse, data_forward)
    d_results_PMD_forward = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_forward)
    d_results_null_forward = get_lppd_and_waic(mcmc_null_forward_reverse, data_forward)

    fit_result["n_sigma_forward"] = compute_n_sigma(
        d_results_PMD_forward,
        d_results_null_forward,
    )

    fit_result["D_max_forward"] = get_y_average_and_hpdi(
        mcmc_PMD_forward_reverse,
        data_forward,
        func=np.median,
        return_hpdi=False,
    )[0]

    fit_result["q_mean_forward"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

    # REVERSE

    data_reverse = {key: val[data["z"] < 0] for key, val in data.items()}
    fit_mcmc(mcmc_PMD_forward_reverse, data_reverse)
    fit_mcmc(mcmc_null_forward_reverse, data_reverse)
    d_results_PMD_reverse = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_reverse)
    d_results_null_reverse = get_lppd_and_waic(mcmc_null_forward_reverse, data_reverse)

    fit_result["n_sigma_reverse"] = compute_n_sigma(
        d_results_PMD_reverse,
        d_results_null_reverse,
    )
    fit_result["D_max_reverse"] = get_y_average_and_hpdi(
        mcmc_PMD_forward_reverse,
        data_forward,
        func=np.median,
        return_hpdi=False,
    )[0]

    fit_result["q_mean_reverse"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

    fit_result["asymmetry"] = compute_assymmetry_combined_vs_forwardreverse(
        d_results_PMD,
        d_results_PMD_forward,
        d_results_PMD_reverse,
    )


def add_noise_estimates(group, fit_result):

    base_columns = [col for col in group.columns if len(col) == 2 and col[0] != col[1]]

    f_ij = group[base_columns].copy()

    f_ij.loc[f_ij.index[:15], "CT"] = np.nan
    f_ij.loc[f_ij.index[15:], "GA"] = np.nan

    f_mean = f_ij.mean(axis=0)
    noise_z = f_ij / f_mean

    # with np.errstate(divide="ignore", invalid="ignore"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
        fit_result["normalized_noise"] = np.nanstd(noise_z.values)
        fit_result["normalized_noise_forward"] = np.nanstd(noise_z.iloc[:15].values)
        fit_result["normalized_noise_reverse"] = np.nanstd(noise_z.iloc[15:].values)


def add_frequentist_fit_results(data, fit_result):
    # reload(fits_frequentist)
    # frequentist_likelihood = fits_frequentist.Frequentist(data, method='likelihood')
    # frequentist_posterior_q = fits_frequentist.Frequentist(data, method="posterior_q")
    # frequentist_posterior = fits_frequentist.Frequentist(data, method="posterior")

    # if False:
    #     frequentist.PMD.m
    #     print(frequentist)
    #     frequentist.plot()

    #     frequentist_likelihood.PMD.m
    #     frequentist_posterior_q.PMD.m
    #     frequentist_posterior.PMD.m

    #     print(frequentist_likelihood)
    #     print(frequentist_posterior_q)
    #     print(frequentist_posterior)

    #     frequentist_likelihood.plot()
    #     frequentist_posterior_q.plot()
    #     frequentist_posterior.plot()

    frequentist = fits_frequentist.Frequentist(data, method="likelihood")
    for var in ["D_max", "A", "q", "c", "phi", "LR", "LR_P", "LR_n_sigma", "valid"]:
        fit_result[f"frequentist_{var}"] = getattr(frequentist, var)


#%%


def init_mcmc(model, **kwargs):
    return MCMC(NUTS(model), jit_model_args=True, **kwargs)


def fit_mcmc(mcmc, data, seed=0):
    mcmc.run(Key(seed), **data)


def use_last_state_as_warmup_state(mcmc):
    # https://github.com/pyro-ppl/numpyro/issues/539
    mcmc._warmup_state = mcmc._last_state


#%%


def group_to_numpyro_data(group, cfg):

    forward = cfg.substitution_bases_forward
    forward_ref = forward[0]
    reverse = cfg.substitution_bases_reverse
    reverse_ref = reverse[0]

    z = np.array(group.iloc[:15]["position"], dtype=int)

    y_forward = np.array(group.iloc[:15][forward], dtype=int)
    N_forward = np.array(group.iloc[:15][forward_ref], dtype=int)

    y_reverse = np.array(group.iloc[-15:][reverse], dtype=int)
    N_reverse = np.array(group.iloc[-15:][reverse_ref], dtype=int)

    data = {
        "z": np.concatenate([z, -z]),
        "y": np.concatenate([y_forward, y_reverse]),
        "N": np.concatenate([N_forward, N_reverse]),
    }

    return data


def match_tax_id_order_in_df_fit_results(df_fit_results, df):
    tax_ids_all = pd.unique(df.tax_id)
    ordered = [tax_id for tax_id in tax_ids_all if tax_id in df_fit_results.index]
    return df_fit_results.loc[ordered]


def fit_single_group_without_timeout(
    group,
    cfg,
    mcmc_PMD,
    mcmc_null,
    mcmc_PMD_forward_reverse,
    mcmc_null_forward_reverse,
):

    data = group_to_numpyro_data(group, cfg)
    fit_mcmc(mcmc_PMD, data)
    fit_mcmc(mcmc_null, data)
    # mcmc_PMD.print_summary(prob=0.68)

    y_median_PMD, y_hpdi_PMD = get_y_average_and_hpdi(
        mcmc_PMD,
        data,
        func=np.median,
    )

    fit_result = compute_fit_results(
        mcmc_PMD,
        mcmc_null,
        mcmc_PMD_forward_reverse,
        mcmc_null_forward_reverse,
        data,
        group,
    )

    d_fit = {
        "median": y_median_PMD,
        "hpdi": y_hpdi_PMD,
        "fit_result": fit_result,
    }

    # if False:
    #     posterior_predictive_PMD = get_posterior_predictive(mcmc_PMD, data)
    #     posterior_predictive_null = get_posterior_predictive(mcmc_null, data)
    #     use_last_state_as_warmup_state(mcmc_PMD)
    #     use_last_state_as_warmup_state(mcmc_null)

    return d_fit


def get_fit_single_group_with_timeout(timeout=60):
    # timeout in seconds
    return timeout_decorator.timeout(timeout)(fit_single_group_without_timeout)


def compute_fits_seriel(df_counts, mcmc_kwargs, cfg):

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)
    mcmc_PMD_forward_reverse = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null_forward_reverse = init_mcmc(model_null, **mcmc_kwargs)

    groupby = df_counts.groupby("tax_id", sort=False, observed=True)
    d_fits = {}

    fit_single_group_first_fit = get_fit_single_group_with_timeout(timeout_first_fit)
    fit_single_group_subsequent_fits = get_fit_single_group_with_timeout(
        timeout_subsequent_fits
    )

    fit_single_group = fit_single_group_first_fit

    logger.info(f"Fit: Initializing fit in seriel.")

    with progress:
        task_fit = progress.add_task(
            "task_status_fitting",
            progress_type="status",
            status="Fitting ",
            name="Fits: ",
            total=len(groupby),
        )

        for tax_id, group in groupby:
            # break

            try:
                d_fit = fit_single_group(
                    group,
                    cfg,
                    mcmc_PMD,
                    mcmc_null,
                    mcmc_PMD_forward_reverse,
                    mcmc_null_forward_reverse,
                )

                d_fits[tax_id] = d_fit

            except TimeoutError:
                logger.warning(f"Fit: Timeout at tax_id {tax_id}. Skipping for now")

            progress.advance(task_fit)
            fit_single_group = fit_single_group_subsequent_fits

    return d_fits


def worker(queue_in, queue_out, mcmc_kwargs, cfg):

    mcmc_PMD = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null = init_mcmc(model_null, **mcmc_kwargs)
    mcmc_PMD_forward_reverse = init_mcmc(model_PMD, **mcmc_kwargs)
    mcmc_null_forward_reverse = init_mcmc(model_null, **mcmc_kwargs)

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
            d_fit = fit_single_group(
                group,
                cfg,
                mcmc_PMD,
                mcmc_null,
                mcmc_PMD_forward_reverse,
                mcmc_null_forward_reverse,
            )

            queue_out.put((tax_id, d_fit))

        except TimeoutError:
            queue_out.put((tax_id, TimeoutError))

        fit_single_group = fit_single_group_subsequent_fits


def compute_fits_parallel_with_progressbar(
    df,
    cfg,
    mcmc_kwargs,
):

    # logger.info(f"Fit: Initializing fit in parallel with progressbar")

    groupby = df.groupby("tax_id", sort=False, observed=True)
    N_groupby = len(groupby)

    N_cores = cfg.N_cores if cfg.N_cores < N_groupby else N_groupby

    manager = Manager()
    queue_in = manager.Queue()
    queue_out = manager.Queue()
    the_pool = Pool(N_cores, worker, (queue_in, queue_out, mcmc_kwargs, cfg))

    d_fits = {}
    with progress:
        task_fit = progress.add_task(
            "task_status_fitting",
            progress_type="status",
            status="Fitting ",
            name="Fits: ",
            total=N_groupby,
        )

        for tax_id, group in groupby:
            queue_in.put((tax_id, group))

        # Get and print results
        for _ in range(N_groupby):
            tax_id, d_fit = queue_out.get()
            if d_fit is not TimeoutError:
                d_fits[tax_id] = d_fit
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

    return d_fits


#%%


def make_df_fit_predictions_from_d_fits(d_fits, cfg):

    z = np.arange(15) + 1
    position = np.concatenate([z, -z])

    # d_fit_predictions = {}
    d_fit_predictions = []
    for key, d_val in d_fits.items():

        median = d_val["median"]
        hpdi = d_val["hpdi"]
        tax_id = key

        data = {
            "tax_id": tax_id,
            "position": position,
            "median": median,
            "hdpi_lower": hpdi[0, :],
            "hdpi_upper": hpdi[1, :],
        }

        df_tmp = pd.DataFrame(data=data)
        # d_fit_predictions[key] = df_tmp
        d_fit_predictions.append(df_tmp)

    df_fit_predictions = pd.concat(d_fit_predictions, axis="index", ignore_index=True)
    df_fit_predictions["shortname"] = cfg.shortname

    categories = ["tax_id", "shortname"]
    df_fit_predictions = utils.downcast_dataframe(
        df_fit_predictions, categories, fully_automatic=False
    )

    return df_fit_predictions


def make_df_fit_results_from_fit_results(fit_results, df_counts, cfg):
    df_fit_results = pd.DataFrame.from_dict(fit_results, orient="index")
    df_fit_results = match_tax_id_order_in_df_fit_results(df_fit_results, df_counts)
    df_fit_results["shortname"] = cfg.shortname

    categories = ["tax_id", "tax_name", "tax_rank", "shortname"]
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


def compute_fits_parallel_with_progressbar_chunks(df, cfg, mcmc_kwargs, chunk_max=1000):
    logger.info(
        f"Fit: Initializing fit in parallel with progressbar "
        "in chunks of size {chunk_max}."
    )

    d_fits_all_chunks = {}
    tax_ids_unique = np.array(pd.unique(df.tax_id))
    chunks = get_chunks(tax_ids_unique, chunk_max)
    for chunk in chunks:
        d_fits_chunk = compute_fits_parallel_with_progressbar(
            df.query("tax_id in @chunk"), cfg, mcmc_kwargs
        )
        d_fits_all_chunks.update(d_fits_chunk)
    return d_fits_all_chunks


def compute_fits(df_counts, cfg, mcmc_kwargs):

    # groupby = df_counts.groupby("tax_id", sort=False, observed=True)

    if cfg.N_cores == 1:  #  or len(groupby) < 10:
        d_fits = compute_fits_seriel(df_counts, mcmc_kwargs, cfg)

    else:
        # d_fits = compute_fits_parallel_with_progressbar(df_counts, cfg, mcmc_kwargs)
        d_fits = compute_fits_parallel_with_progressbar_chunks(
            df_counts,
            cfg,
            mcmc_kwargs,
            chunk_max=1000,
        )

    fit_results = {tax_id: d["fit_result"] for tax_id, d in d_fits.items()}

    df_fit_results = make_df_fit_results_from_fit_results(fit_results, df_counts, cfg)
    df_fit_predictions = make_df_fit_predictions_from_d_fits(d_fits, cfg)

    return df_fit_results, df_fit_predictions


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
    parquet_fit_predictions = io.Parquet(cfg.filename_fit_predictions)

    if parquet_fit_results.exists(cfg.forced) and parquet_fit_predictions.exists(
        cfg.forced
    ):

        include = [
            "min_alignments",
            "min_y_sum",
            "substitution_bases_forward",
            "substitution_bases_reverse",
            "N_fits",
            "shortname",
            "filename",
        ]

        metadata_cfg = cfg.to_dict()

        metadata_file_fit_results = parquet_fit_results.load_metadata()
        metadata_file_fit_predictions = parquet_fit_predictions.load_metadata()

        if utils.metadata_is_similar(
            metadata_file_fit_results, metadata_cfg, include=include
        ) and utils.metadata_is_similar(
            metadata_file_fit_predictions, metadata_cfg, include=include
        ):
            logger.info(f"Fit: Loading fits from parquet-file.")
            df_fit_results = parquet_fit_results.load()
            df_fit_predictions = parquet_fit_predictions.load()
            return df_fit_results, df_fit_predictions

    logger.info(f"Fit: Generating fits and saving to file.")

    df_counts_top_N = get_top_max_fits(df_counts, cfg.N_fits)

    mcmc_kwargs = dict(
        progress_bar=False,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,  # problem when setting to 2
        chain_method="sequential",
        # http://num.pyro.ai/en/stable/_modules/numpyro/infer/mcmc.html#MCMC
    )

    # df = df_counts = df_counts_top_N
    df_fit_results, df_fit_predictions = compute_fits(df_counts_top_N, cfg, mcmc_kwargs)

    parquet_fit_results.save(df_fit_results, metadata=cfg.to_dict())
    parquet_fit_predictions.save(df_fit_predictions, metadata=cfg.to_dict())

    return df_fit_results, df_fit_predictions


#%%


# import arviz as az

# data_no_y = filter_out_y(data)

# def get_InferenceData(mcmc, model):

#     posterior_samples = mcmc.get_samples()
#     posterior_predictive = Predictive(model, posterior_samples)(Key(1), **data_no_y)
#     prior = Predictive(model, num_samples=500)(Key(2), **data_no_y)

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
