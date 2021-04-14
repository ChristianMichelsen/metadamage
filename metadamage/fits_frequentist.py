import numpy as np
from scipy.stats import chi2 as sp_chi2
from scipy.stats import norm as sp_norm
from iminuit import describe
import matplotlib.pyplot as plt
from scipy.special import erfinv, erf

from scipy.stats import betabinom
from iminuit import Minuit


#%%

from numba import njit
import math


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


class FrequentistPMD:
    def __init__(self, data):
        self.z = data["z"]
        self.k = data["y"]
        self.n = data["N"]
        self._setup_minuit()

    def __call__(self, q, A, c, phi):
        return f_frequentist_PMD(q, A, c, phi, self.z, self.k, self.n)

    def _setup_minuit(self, m=None):
        if m is None:
            self.m = Minuit(self.__call__, q=0.1, A=0.1, c=0.01, phi=1000)
        else:
            self.m = m

        self.m.limits["q"] = (0, 1)
        self.m.limits["A"] = (0, 1)
        self.m.limits["c"] = (0, 1)
        self.m.limits["phi"] = (2, None)
        self.m.errordef = Minuit.LIKELIHOOD

    def fit(self):
        self.m.migrad()

        # first time try to reinitialize with previous fit result
        if not self.m.valid:
            m = Minuit(self.__call__, *self.m.values)
            self._setup_minuit(m)
            self.m.migrad()

        # if not working, continue with second guess
        if not self.m.valid:
            m = Minuit(self.__call__, q=0.1, A=0.5, c=0.01, phi=1000)
            self._setup_minuit(m)
            self.m.migrad()

        self.valid = self.m.valid
        return self

    def migrad(self):
        return self.fit()

    def minos(self):
        self.m.minos()
        return self


#%%


@njit
def gammaln_scalar(x):
    return math.lgamma(x)


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


#%%


def prob_to_n_sigma(p):
    return np.sqrt(2) * erfinv(p)


def n_sigma_to_prob(n_sigma):
    return erf(n_sigma / np.sqrt(2))


def compute_likelihood_ratio(frequentist_PMD, frequentist_null):
    LR = -2 * (frequentist_PMD.m.fval - frequentist_null.m.fval)

    df = len(describe(frequentist_PMD)) - len(describe(frequentist_null))
    LR_P = sp_chi2.sf(x=LR, df=df)
    LR_n_sigma = prob_to_n_sigma(1 - LR_P)

    return LR, LR_P, LR_n_sigma


from scipy.stats import beta as sp_beta
from scipy.optimize import fmin


def HDR_beta(a, b, alpha=0.68):
    # freeze distribution with given arguments
    # initial guess for HDIlowTailPr

    dist = sp_beta(a=a, b=b)

    def intervalWidth(lowTailPr):
        return dist.ppf(alpha + lowTailPr) - dist.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, 1.0 - alpha, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return dist.ppf([HDIlowTailPr, alpha + HDIlowTailPr])


def HDR_beta_vec(a, b, alpha=0.68):
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return HDR_beta(a, b, alpha=alpha)
    else:
        it = zip(a[::10], b[::10])
        out = np.array([HDR_beta(ai, bi, alpha=alpha) for ai, bi in it])
        return out


class Frequentist:
    def __init__(self, data):
        self.PMD = FrequentistPMD(data).fit()
        self.null = FrequentistNull(data).fit()
        p = compute_likelihood_ratio(self.PMD, self.null)
        self.LR, self.LR_P, self.LR_n_sigma = p

        self.valid = self.PMD.valid

        self.data = data
        self.z = data["z"]
        self.k = data["y"]
        self.n = data["N"]

    def __repr__(self):
        return "Frequentist(data)"

    def __str__(self):
        s = f"A = {self.A:.3f}, q = {self.q:.3f}, c = {self.c:.5f}, phi = {self.phi:.1f}, valid = {self.valid} \n"
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

    def make_plot(self, N_points=1000):

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

        ax.plot(self.z[:15], self.k[:15] / self.n[:15], "ok", label="Data")

        ax.plot(zz, Dz, color="C0", lw=2, label="mean")

        median = sp_beta.median(a=a, b=b)
        ax.plot(zz, median, color="C3", lw=2, label="median")

        ax.fill_between(
            zz,
            intervals1[0],
            intervals1[1],
            color="C1",
            alpha=0.1,
            label="CI, 68%, equal (median)",
        )

        ax.plot(zz, intervals1[0], color="C1", ls="--")
        ax.plot(zz, intervals1[1], color="C1", ls="--")

        minmax = HDR_beta_vec(a, b)
        ax.fill_between(
            zz[::10],
            minmax[:, 0],
            minmax[:, 1],
            color="C2",
            alpha=0.1,
            label="CI, 68%, HDR",
        )

        ax.plot(zz[::10], minmax[:, 0], color="C2", ls="--")
        ax.plot(zz[::10], minmax[:, 1], color="C2", ls="--")

        ax.legend()
        ax.set(xlabel="Position", ylabel="Damage", ylim=(0, None))

        return fig, ax
