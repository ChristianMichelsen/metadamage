# Scientific Library
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfinv
from scipy.stats import beta as sp_beta, chi2 as sp_chi2

# Standard Library
import math

# Third Party
from iminuit import describe
from numba import njit


def get_priors():

    # beta
    q_prior = mu_phi_to_alpha_beta(mu=0.2, phi=5)  # mean = 0.2, concentration = 5
    A_prior = mu_phi_to_alpha_beta(mu=0.2, phi=5)  # mean = 0.2, concentration = 5
    c_prior = mu_phi_to_alpha_beta(mu=0.1, phi=10)  # mean = 0.1, concentration = 10

    # exponential (min, scale)
    phi_prior = (2, 1000)

    return {"q": q_prior, "A": A_prior, "c": c_prior, "phi": phi_prior}


#%%


def prob_to_n_sigma(p):
    return np.sqrt(2) * erfinv(p)


def n_sigma_to_prob(n_sigma):
    return erf(n_sigma / np.sqrt(2))


def compute_likelihood_ratio(frequentist_PMD, frequentist_null):
    LR = -2 * (frequentist_PMD.log_likelihood - frequentist_null.log_likelihood)

    df = len(describe(frequentist_PMD)) - len(describe(frequentist_null))
    LR_P = sp_chi2.sf(x=LR, df=df)
    LR_n_sigma = prob_to_n_sigma(1 - LR_P)

    return LR, LR_P, LR_n_sigma


def sample_from_param_grid(param_grid, random_state=None):
    np.random.seed(42)
    parameters = {}
    for key, dist in param_grid.items():
        parameters[key] = dist.rvs(random_state=random_state)
    return parameters


def alpha_beta_to_mu_phi(alpha, beta):
    mu = alpha / (alpha + beta)
    phi = alpha + beta
    return mu, phi


def mu_phi_to_alpha_beta(mu, phi):
    alpha = mu * phi
    beta = phi * (1 - mu)
    return alpha, beta


#%%


@njit
def gammaln_scalar(x):
    return math.lgamma(x)


@njit
def gammaln_vec(xs):
    out = np.empty(len(xs), dtype="float")
    for i, x in enumerate(xs):
        out[i] = math.lgamma(x)
    return out


@njit
def log_betabinom_PMD(k, N, alpha, beta):
    return (
        gammaln_vec(N + 1)
        + gammaln_vec(k + alpha)
        + gammaln_vec(N - k + beta)
        + gammaln_vec(alpha + beta)
        - (
            gammaln_vec(k + 1)
            + gammaln_vec(N - k + 1)
            + gammaln_vec(alpha)
            + gammaln_vec(beta)
            + gammaln_vec(N + alpha + beta)
        )
    )


@njit
def xlog1py(x, y):
    if x == 0:
        return 0

    return x * np.log1p(y)


@njit
def xlogy(x, y):
    if x == 0:
        return 0

    return x * np.log(y)


@njit
def betaln(x, y):
    return gammaln_scalar(x) + gammaln_scalar(y) - gammaln_scalar(x + y)


@njit
def log_beta(x, alpha, beta):
    lPx = xlog1py(beta - 1.0, -x) + xlogy(alpha - 1.0, x)
    lPx -= betaln(alpha, beta)
    return lPx


@njit
def log_exponential(x, loc, scale):
    if x < loc:
        return -np.inf
    return -(x - loc) / scale - np.log(scale)


#%%


@njit
def log_betabinom_null(k, N, alpha, beta):
    return (
        gammaln_vec(N + 1)
        + gammaln_vec(k + alpha)
        + gammaln_vec(N - k + beta)
        + gammaln_scalar(alpha + beta)
        - (
            gammaln_vec(k + 1)
            + gammaln_vec(N - k + 1)
            + gammaln_scalar(alpha)
            + gammaln_scalar(beta)
            + gammaln_vec(N + alpha + beta)
        )
    )


#%%


def plot_beta_distribution(alpha, beta):
    fig, ax = plt.subplots()

    if isinstance(alpha, (int, float)) and isinstance(beta, (int, float)):
        alpha = [alpha]
        beta = [beta]

    for a, b in zip(alpha, beta):
        x = np.linspace(0, 1, 1000)
        y = sp_beta.pdf(x, a=a, b=b)

        mu, phi = alpha_beta_to_mu_phi(a, b)
        label = f"α={a}, β={b}, μ={mu}, φ={phi}"
        ax.plot(x, y, "-", label=label)
    ax.legend()
    return fig, ax


def plot_beta_distribution_mu_phi(mu, phi):
    fig, ax = plt.subplots()

    if isinstance(mu, (int, float)) and isinstance(phi, (int, float)):
        mu = [mu]
        phi = [phi]

    for μ, φ in zip(mu, phi):
        x = np.linspace(0, 1, 1000)
        a, b = mu_phi_to_alpha_beta(μ, φ)
        y = sp_beta.pdf(x, a=a, b=b)
        label = f"μ={μ}, φ={φ}, α={a}, β={b}"
        ax.plot(x, y, "-", label=label)
    ax.legend()
    return fig, ax


# plot_beta_distribution_mu_phi([0.2, 0.1], [5, 10])

# fig, ax = plt.subplots()
# x = np.linspace(0, 1, 1000)
# ax.plot(
#     x,
#     sp_beta.pdf(x, *mu_phi_to_alpha_beta(mu=0.2, phi=5)),
#     "-",
#     label=f"μ={0.2}, φ={5}",
#     lw=2,
#     color="C0",
# )
# ax.plot(
#     x,
#     sp_beta.pdf(x, *mu_phi_to_alpha_beta(mu=0.1, phi=10)),
#     "-",
#     label=f"μ={0.1}, φ={10}",
#     lw=2,
#     color="C3",
# )
# ax.plot(
#     x,
#     sp_beta.pdf(x, *mu_phi_to_alpha_beta(mu=0.2, phi=10)),
#     "-",
#     label=f"μ={0.2}, φ={10}",
#     alpha=0.5,
#     color="C1",
# )
# ax.plot(
#     x,
#     sp_beta.pdf(x, *mu_phi_to_alpha_beta(mu=0.1, phi=5)),
#     "-",
#     label=f"μ={0.1}, φ={5}",
#     alpha=0.5,
#     color="C2",
# )
# ax.legend()
# ax.set(xlabel="x", ylabel="PDF", ylim=(0, 10))


#%%


# from scipy.stats import beta as sp_beta
# from scipy.optimize import fmin

# def HDR_beta(a, b, alpha=0.68):
#     # freeze distribution with given arguments
#     # initial guess for HDIlowTailPr

#     dist = sp_beta(a=a, b=b)

#     def intervalWidth(lowTailPr):
#         return dist.ppf(alpha + lowTailPr) - dist.ppf(lowTailPr)

#     # find lowTailPr that minimizes intervalWidth
#     HDIlowTailPr = fmin(intervalWidth, 1.0 - alpha, ftol=1e-8, disp=False)[0]
#     # return interval as array([low, high])
#     return dist.ppf([HDIlowTailPr, alpha + HDIlowTailPr])


# def HDR_beta_vec(a, b, alpha=0.68):
#     if isinstance(a, (float, int)) and isinstance(b, (float, int)):
#         return HDR_beta(a, b, alpha=alpha)
#     else:
#         it = zip(a[::10], b[::10])
#         out = np.array([HDR_beta(ai, bi, alpha=alpha) for ai, bi in it])
#         return out


# def add_assymetry_results_to_fit_results(
#     mcmc_PMD_forward_reverse,
#     mcmc_null_forward_reverse,
#     data,
#     fit_result,
#     d_results_PMD,
# ):
#     """computes the assymetry between a fit to forward data and reverse data
#     the assymmetry is here defined as the n_sigma (WAIC) between the two fits
#     """

#     # FORWARD

#     data_forward = {key: val[data["z"] > 0] for key, val in data.items()}
#     fit_mcmc(mcmc_PMD_forward_reverse, data_forward)
#     fit_mcmc(mcmc_null_forward_reverse, data_forward)
#     d_results_PMD_forward = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_forward)
#     d_results_null_forward = get_lppd_and_waic(mcmc_null_forward_reverse, data_forward)

#     fit_result["n_sigma_forward"] = compute_n_sigma(
#         d_results_PMD_forward,
#         d_results_null_forward,
#     )

#     fit_result["D_max_forward"] = compute_posterior(
#         mcmc_PMD_forward_reverse,
#         data_forward,
#         func=np.median,
#         return_hpdi=False,
#     )[0]

#     fit_result["q_mean_forward"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

#     # REVERSE

#     data_reverse = {key: val[data["z"] < 0] for key, val in data.items()}
#     fit_mcmc(mcmc_PMD_forward_reverse, data_reverse)
#     fit_mcmc(mcmc_null_forward_reverse, data_reverse)
#     d_results_PMD_reverse = get_lppd_and_waic(mcmc_PMD_forward_reverse, data_reverse)
#     d_results_null_reverse = get_lppd_and_waic(mcmc_null_forward_reverse, data_reverse)

#     fit_result["n_sigma_reverse"] = compute_n_sigma(
#         d_results_PMD_reverse,
#         d_results_null_reverse,
#     )
#     fit_result["D_max_reverse"] = compute_posterior(
#         mcmc_PMD_forward_reverse,
#         data_forward,
#         func=np.median,
#         return_hpdi=False,
#     )[0]

#     fit_result["q_mean_reverse"] = get_mean_of_variable(mcmc_PMD_forward_reverse, "q")

#     fit_result["asymmetry"] = compute_assymmetry_combined_vs_forwardreverse(
#         d_results_PMD,
#         d_results_PMD_forward,
#         d_results_PMD_reverse,
#     )


# def add_noise_estimates(group, fit_result):

#     base_columns = [col for col in group.columns if len(col) == 2 and col[0] != col[1]]

#     f_ij = group[base_columns].copy()

#     f_ij.loc[f_ij.index[:15], "CT"] = np.nan
#     f_ij.loc[f_ij.index[15:], "GA"] = np.nan

#     f_mean = f_ij.mean(axis=0)
#     noise_z = f_ij / f_mean

#     # with np.errstate(divide="ignore", invalid="ignore"):
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
#         fit_result["normalized_noise"] = np.nanstd(noise_z.values)
#         fit_result["normalized_noise_forward"] = np.nanstd(noise_z.iloc[:15].values)
#         fit_result["normalized_noise_reverse"] = np.nanstd(noise_z.iloc[15:].values)
