import math

from iminuit import describe
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.special import erf, erfinv
from scipy.stats import beta as sp_beta, chi2 as sp_chi2


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
