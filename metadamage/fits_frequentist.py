import numpy as np
from scipy.stats import chi2 as sp_chi2
from scipy.stats import norm as sp_norm
from iminuit import describe
import matplotlib.pyplot as plt
from scipy.special import erfinv, erf

from scipy.stats import betabinom
from iminuit import Minuit

#%%

# # beta
# q_prior = dict(a=0.8, b=3.2)  # mean = 0.2, shape = 4
# A_prior = dict(a=0.8, b=3.2)  # mean = 0.2, shape = 4
# c_prior = dict(a=1.0, b=9.0)  # mean = 0.1, shape = 10

# # exponential
# phi_prior = dict(loc=2, scale=1000)


from scipy.stats import uniform as sp_uniform
from scipy.stats import beta as sp_beta
from scipy.stats import expon as sp_exponential


# beta
q_prior = (1, 4)  # mean = 0.2, shape = 4
A_prior = (1, 4)  # mean = 0.2, shape = 4
c_prior = (1.0, 9.0)  # mean = 0.1, shape = 10

# exponential
phi_prior = (2, 1000)

#%%


def plot_beta(alpha, beta):
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


# plot_beta([0.8, 1, 1], [3.2, 4, 9])

#%%

from numba import njit
import math


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
def log_betabinom_PMD(k, n, a, b):
    return (
        gammaln_vec(n + 1)
        + gammaln_vec(k + a)
        + gammaln_vec(n - k + b)
        + gammaln_vec(a + b)
        - (
            gammaln_vec(k + 1)
            + gammaln_vec(n - k + 1)
            + gammaln_vec(a)
            + gammaln_vec(b)
            + gammaln_vec(n + a + b)
        )
    )


@njit
def f_frequentist_PMD(q, A, c, phi, z, k, n):
    Dz = A * (1 - q) ** (np.abs(z) - 1) + c
    a = Dz * phi
    b = (1 - Dz) * phi
    return -log_betabinom_PMD(k=k, n=n, a=a, b=b).sum()
    # return -betabinom.logpmf(k=k, n=n, a=a, b=b).sum()


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
def log_beta(x, a, b):
    lPx = xlog1py(b - 1.0, -x) + xlogy(a - 1.0, x)
    lPx -= betaln(a, b)
    return lPx


@njit
def log_exponential(x, loc, scale):
    if x < loc:
        return -np.inf
    return -(x - loc) / scale - np.log(scale)


@njit
def log_prior(q, A, c, phi):
    lp = log_beta(q, *q_prior)
    lp += log_beta(A, a=A_prior[0], b=A_prior[1])
    lp += log_beta(c, a=c_prior[0], b=c_prior[1])
    lp += log_exponential(phi, loc=phi_prior[0], scale=phi_prior[1])

    # if lp == np.inf:
    #     lp = 1e20

    return -lp


@njit
def f_frequentist_PMD_posterior(q, A, c, phi, z, k, n):
    log_likelihood = f_frequentist_PMD(q, A, c, phi, z, k, n)
    log_p = log_prior(q, A, c, phi)
    return log_likelihood + log_p


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


class FrequentistPMD:
    def __init__(self, data, method="posterior"):
        self.z = data["z"]
        self.k = data["y"]
        self.n = data["N"]
        self.method = method
        self._setup_minuit()

        self.param_grid = {
            "q": sp_beta(*q_prior),  # mean = 0.2, shape = 4
            "A": sp_beta(*A_prior),  # mean = 0.2, shape = 4
            "c": sp_beta(*c_prior),  # mean = 0.1, shape = 10
            "phi": sp_exponential(*phi_prior),
        }

    def __call__(self, q, A, c, phi):

        if self.method == "likelihood":
            return self.f_log_likelihood(q, A, c, phi)

        elif self.method == "posterior":
            return self.f_log_posterior(q, A, c, phi)

    def f_log_likelihood(self, q, A, c, phi):
        return f_frequentist_PMD(q, A, c, phi, self.z, self.k, self.n)

    def f_log_posterior(self, q, A, c, phi):
        return f_frequentist_PMD_posterior(q, A, c, phi, self.z, self.k, self.n)

    def _setup_minuit(self, m=None):

        if self.method == "likelihood":
            f = self.f_log_likelihood

        elif self.method == "posterior":
            f = self.f_log_posterior

        if m is None:
            self.m = Minuit(f, q=0.1, A=0.1, c=0.01, phi=1000)
        else:
            self.m = m

        if self.method == "likelihood":
            eps = 0
        elif self.method == "posterior":
            eps = 1e-10

        self.m.limits["q"] = (0 + eps, 1 - eps)
        self.m.limits["A"] = (0 + eps, 1 - eps)
        self.m.limits["c"] = (0 + eps, 1 - eps)
        self.m.limits["phi"] = (2, None)
        self.m.errordef = Minuit.LIKELIHOOD

    def fit(self):
        self.m.migrad()

        # first time try to reinitialize with previous fit result
        if not self.m.valid:
            self.m.migrad()

        # if fit was accepted, stop
        if self.m.valid:
            self.valid = self.m.valid
            return self

        # if not working, continue with new guesses
        if not self.m.valid:

            self.i = 0
            while True:
                p0 = sample_from_param_grid(self.param_grid)
                for key, val in p0.items():
                    self.m.values[key] = val
                self.m.migrad()
                if self.m.valid or self.i >= 100:
                    break

                self.m.migrad()
                if self.m.valid or self.i >= 100:
                    break

                self.i += 1

        self.valid = self.m.valid
        return self

    @property
    def log_likelihood(self):
        return self.f_log_likelihood(*self.m.values)

    def migrad(self):
        return self.fit()

    def minos(self):
        self.m.minos()
        return self


# f = FrequentistPMD(data, method="posterior").fit()
# f.m
# %timeit FrequentistPMD(data, method="posterior").fit()


# f = FrequentistPMD(data, method="likelihood").fit()
# f.m
# %timeit FrequentistPMD(data, method="likelihood").fit()

# m = Minuit(f, q=0.1, A=0.1, c=0.01, phi=1000)
# m.limits["q"] = (0, 1)
# m.limits["A"] = (0, 1)
# m.limits["c"] = (0, 1)
# m.limits["phi"] = (2, None)
# m.errordef = Minuit.LIKELIHOOD
# m.migrad()
# m.migrad()

#%%


@njit
def log_betabinom_null(k, n, a, b):
    return (
        gammaln_vec(n + 1)
        + gammaln_vec(k + a)
        + gammaln_vec(n - k + b)
        + gammaln_scalar(a + b)
        - (
            gammaln_vec(k + 1)
            + gammaln_vec(n - k + 1)
            + gammaln_scalar(a)
            + gammaln_scalar(b)
            + gammaln_vec(n + a + b)
        )
    )


@njit
def f_frequentist_null(q, phi, k, n):
    a = q * phi
    b = (1 - q) * phi
    return -log_betabinom_null(k=k, n=n, a=a, b=b).sum()


class FrequentistNull:
    def __init__(self, data):
        self.z = data["z"]
        self.k = data["y"]
        self.n = data["N"]
        self._setup_minuit()

    def __call__(self, q, phi):
        return f_frequentist_null(q, phi, self.k, self.n)

    def _setup_minuit(self):
        self.m = Minuit(self.__call__, q=0.1, phi=100)
        self.m.limits["q"] = (0, 1)
        self.m.limits["phi"] = (2, None)
        self.m.errordef = Minuit.LIKELIHOOD

    def fit(self):
        self.m.migrad()
        return self

    def migrad(self):
        self.m.migrad()
        return self

    def minos(self):
        self.m.minos()
        return self

    @property
    def log_likelihood(self):
        return f_frequentist_null(*self.m.values, self.k, self.n)


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


class Frequentist:
    def __init__(self, data, method="posterior"):
        self.PMD = FrequentistPMD(data, method=method).fit()
        self.null = FrequentistNull(data).fit()
        p = compute_likelihood_ratio(self.PMD, self.null)
        self.LR, self.LR_P, self.LR_n_sigma = p

        self.valid = self.PMD.valid

        self.data = data
        self.z = data["z"]
        self.k = data["y"]
        self.n = data["N"]
        self.method = method

    def __repr__(self):
        return f"Frequentist(data, method={self.method})"

    def __str__(self):
        s = f"A = {self.A:.3f}, q = {self.q:.3f}, c = {self.c:.5f}, phi = {self.phi:.1f}, D_max = {self.D_max:.3f}, valid = {self.valid} \n"
        s += f"LR = {self.LR:.3f}, LR as prob = {self.LR_P:.4%}, LR as n_sigma = {self.LR_n_sigma:.3f}"
        return s

    @property
    def D_max_with_sigma(self):
        mu = self.PMD.m.values["A"] + self.PMD.m.values["c"]
        sigma2 = self.PMD.m.errors["A"] ** 2 + self.PMD.m.errors["c"] ** 2
        sigma2 += 2 * self.PMD.m.covariance["A", "c"]
        sigma = np.sqrt(sigma2)
        return mu, sigma

    @property
    def D_max(self):
        D_max = self.PMD.m.values["A"] + self.PMD.m.values["c"]
        return D_max

    @property
    def A(self):
        return self.PMD.m.values["A"]

    @property
    def q(self):
        return self.PMD.m.values["q"]

    @property
    def c(self):
        return self.PMD.m.values["c"]

    @property
    def phi(self):
        return self.PMD.m.values["phi"]

    def plot(self, N_points=1000):

        A = self.A
        q = self.q
        c = self.c
        phi = self.phi

        zz = np.linspace(1, 15, N_points)
        Dz = A * (1 - q) ** (np.abs(zz) - 1) + c

        a = Dz * phi
        b = (1 - Dz) * phi

        # Confidence interval with equal areas around the median (?mean?)
        intervals1 = sp_beta.interval(alpha=n_sigma_to_prob(n_sigma=1), a=a, b=b)
        intervals2 = sp_beta.interval(alpha=n_sigma_to_prob(n_sigma=2), a=a, b=b)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.z[:15], self.k[:15] / self.n[:15], "ok", label="Data Forward")
        ax.plot(-self.z[15:], self.k[15:] / self.n[15:], "xk", label="Data Reverse")

        ax.plot(zz, Dz, color="C0", lw=2, label="mean")

        for i, intervals in enumerate([intervals1, intervals2]):

            ax.fill_between(
                zz,
                intervals[0],
                intervals[1],
                color="C2",
                alpha=0.1,
                label=f"CI, {i+1}σ, equal",
            )

            ax.plot(zz, intervals1[0], color="C2", ls="--")
            ax.plot(zz, intervals1[1], color="C2", ls="--")

        # if False:

        #     median = sp_beta.median(a=a, b=b)
        #     ax.plot(zz, median, color="C3", lw=2, label="median")

        #     minmax = HDR_beta_vec(a, b)
        #     ax.fill_between(
        #         zz[::10],
        #         minmax[:, 0],
        #         minmax[:, 1],
        #         color="C1",
        #         alpha=0.1,
        #         label="CI, 68%, HDR",
        #     )

        #     ax.plot(zz[::10], minmax[:, 0], color="C1", ls="--")
        #     ax.plot(zz[::10], minmax[:, 1], color="C1", ls="--")

        ax.legend()
        ax.set(xlabel="Position", ylabel="Damage", ylim=(0, None))

        return fig, ax
