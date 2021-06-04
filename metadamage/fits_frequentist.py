from iminuit import Minuit
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.stats import (
    beta as sp_beta,
    betabinom as sp_betabinom,
    expon as sp_exponential,
)

from metadamage import fit_utils


#%%

priors = fit_utils.get_priors()
q_prior = priors["q"]  # mean = 0.2, concentration = 5
A_prior = priors["A"]  # mean = 0.2, concentration = 5
c_prior = priors["c"]  # mean = 0.1, concentration = 10
phi_prior = priors["phi"]


#%%


#%%


@njit
def log_likelihood_PMD(q, A, c, phi, z, k, N):
    Dz = A * (1 - q) ** (np.abs(z) - 1) + c
    alpha = Dz * phi
    beta = (1 - Dz) * phi
    return -fit_utils.log_betabinom_PMD(k=k, N=N, alpha=alpha, beta=beta).sum()
    # return -sp_betabinom.logpmf(k=k, n=n, a=alpha, b=beta).sum()


@njit
def log_prior_PMD(q, A, c, phi):
    lp = fit_utils.log_beta(q, *q_prior)
    lp += fit_utils.log_beta(A, *A_prior)
    lp += fit_utils.log_beta(c, *c_prior)
    lp += fit_utils.log_exponential(phi, *phi_prior)
    return -lp


@njit
def log_posterior_PMD(q, A, c, phi, z, k, N):
    log_likelihood = log_likelihood_PMD(q, A, c, phi, z, k, N)
    log_p = log_prior_PMD(q, A, c, phi)
    return log_likelihood + log_p


#%%


class FrequentistPMD:
    def __init__(self, data, method="posterior"):
        self.z = data["z"]
        self.k = data["k"]
        self.N = data["N"]
        self.method = method
        self._setup_p0()
        self._setup_minuit()
        self.is_fitted = False

    def __repr__(self):
        s = f"FrequentistPMD(data, method={self.method}). \n\n"
        if self.is_fitted:
            s += self.__str__()
        return s

    def __str__(self):
        if self.is_fitted:
            s = f"A = {self.A:.3f}, q = {self.q:.3f}, c = {self.c:.5f}, phi = {self.phi:.1f} \n"
            s += f"D_max = {self.D_max:.3f} +/- {self.D_max_std:.3f}, rho_Ac = {self.rho_Ac:.3f} \n"
            s += f"valid = {self.valid}"
            return s
        else:
            return f"FrequentistPMD(data, method={self.method}). \n\n"

    def __call__(self, q, A, c, phi):
        if self.method == "likelihood":
            return self.log_likelihood_PMD(q, A, c, phi)
        elif self.method == "posterior":
            return self.log_posterior_PMD(q, A, c, phi)

    def log_likelihood_PMD(self, q, A, c, phi):
        return log_likelihood_PMD(q, A, c, phi, self.z, self.k, self.N)

    def log_posterior_PMD(self, q, A, c, phi):
        return log_posterior_PMD(q, A, c, phi, self.z, self.k, self.N)

    def _setup_p0(self):
        # if self.force_null_fit:
        # self.p0 = dict(q=0.0, A=0.0, c=0.01, phi=1000)
        # else:
        self.p0 = dict(q=0.1, A=0.1, c=0.01, phi=1000)
        self.param_grid = {
            "q": sp_beta(*q_prior),  # mean = 0.2, shape = 4
            "A": sp_beta(*A_prior),  # mean = 0.2, shape = 4
            "c": sp_beta(*c_prior),  # mean = 0.1, shape = 10
            "phi": sp_exponential(*phi_prior),
        }

    def _setup_minuit(self, m=None):

        if self.method == "likelihood":
            f = self.log_likelihood_PMD

        elif self.method == "posterior":
            f = self.log_posterior_PMD

        if m is None:
            self.m = Minuit(f, **self.p0)
        else:
            self.m = m

        if self.method == "likelihood":
            self.m.limits["A"] = (0, 1)
            self.m.limits["q"] = (0, 1)
            self.m.limits["c"] = (0, 1)

        elif self.method == "posterior":
            eps = 1e-10
            self.m.limits["A"] = (0 + eps, 1 - eps)
            self.m.limits["q"] = (0 + eps, 1 - eps)
            self.m.limits["c"] = (0 + eps, 1 - eps)

        self.m.limits["phi"] = (2, None)
        self.m.errordef = Minuit.LIKELIHOOD

    def fit(self):
        self.m.migrad()
        self.is_fitted = True

        # first time try to reinitialize with previous fit result
        if not self.m.valid:
            self.m.migrad()

        # if not working, continue with new guesses
        if not self.m.valid:
            self.i = 0
            while True:
                p0 = fit_utils.sample_from_param_grid(self.param_grid)
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
        self._set_D_max()
        return self

    @property
    def log_likelihood(self):
        return self.log_likelihood_PMD(*self.values)

    def migrad(self):
        return self.fit()

    def minos(self):
        self.m.minos()
        return self

    @property
    def values(self):
        return self.m.values

    @property
    def A(self):
        return self.m.values["A"]

    @property
    def A_std(self):
        return self.m.errors["A"]

    @property
    def q(self):
        return self.m.values["q"]

    @property
    def q_std(self):
        return self.m.errors["q"]

    @property
    def c(self):
        return self.m.values["c"]

    @property
    def c_std(self):
        return self.m.errors["c"]

    @property
    def phi(self):
        return self.m.values["phi"]

    @property
    def phi_std(self):
        return self.m.errors["phi"]

    def _set_D_max(self):

        A = self.A
        # q = self.q
        c = self.c
        phi = self.phi

        Dz_z1 = A + c
        alpha = Dz_z1 * phi
        beta = (1 - Dz_z1) * phi

        dist = sp_betabinom(n=self.N[0], a=alpha, b=beta)

        # mu = dist.mean() / frequentist.N[0] - c
        mu = A
        std = np.sqrt(dist.var()) / self.N[0]

        self.D_max = mu
        self.D_max_std = std

    @property
    def correlation(self):
        return self.m.covariance.correlation()

    @property
    def rho_Ac(self):
        return self.correlation["A", "c"]


#%%


@njit
def f_frequentist_null(c, phi, k, N):
    alpha = c * phi
    beta = (1 - c) * phi
    return -fit_utils.log_betabinom_null(k=k, N=N, alpha=alpha, beta=beta).sum()


class FrequentistNull:
    def __init__(self, data):
        self.z = data["z"]
        self.k = data["k"]
        self.N = data["N"]
        self._setup_minuit()

    def __call__(self, c, phi):
        return f_frequentist_null(c, phi, self.k, self.N)

    def _setup_minuit(self):
        self.m = Minuit(self.__call__, c=0.1, phi=100)
        self.m.limits["c"] = (0, 1)
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
        return f_frequentist_null(*self.m.values, self.k, self.N)

    @property
    def c(self):
        return self.m.values["c"]

    @property
    def phi(self):
        return self.m.values["phi"]

    @property
    def values(self):
        return self.m.values


#%%


class Frequentist:
    def __init__(self, data, method="posterior"):
        self.PMD = FrequentistPMD(data, method=method).fit()
        self.null = FrequentistNull(data).fit()
        p = fit_utils.compute_likelihood_ratio(self.PMD, self.null)
        self.lambda_LR, self.lambda_LR_P, self.lambda_LR_n_sigma = p

        self.valid = self.PMD.valid

        self.data = data
        self.z = data["z"]
        self.k = data["k"]
        self.N = data["N"]
        self.method = method

    def __repr__(self):
        s = f"Frequentist(data, method={self.method}). \n\n"
        s += self.__str__()
        return s

    def __str__(self):
        s = f"A = {self.A:.3f}, q = {self.q:.3f}, c = {self.c:.5f}, phi = {self.phi:.1f} \n"
        s += f"D_max = {self.D_max:.3f} +/- {self.D_max_std:.3f}, rho_Ac = {self.rho_Ac:.3f} \n"
        s += f"lambda_LR = {self.lambda_LR:.3f}, lambda_LR as prob = {self.lambda_LR_P:.4%}, lambda_LR as n_sigma = {self.lambda_LR_n_sigma:.3f} \n"
        s += f"valid = {self.valid}"
        return s

    @property
    def D_max(self):
        return self.PMD.D_max

    @property
    def D_max_std(self):
        return self.PMD.D_max_std

    @property
    def A(self):
        return self.PMD.A

    @property
    def A_std(self):
        return self.PMD.A_std

    @property
    def q(self):
        return self.PMD.q

    @property
    def q_std(self):
        return self.PMD.q_std

    @property
    def c(self):
        return self.PMD.c

    @property
    def c_std(self):
        return self.PMD.c_std

    @property
    def phi(self):
        return self.PMD.phi

    @property
    def phi_std(self):
        return self.PMD.phi_std

    @property
    def rho_Ac(self):
        return self.PMD.rho_Ac

    def plot(self, N_points=1000):

        A = self.A
        q = self.q
        c = self.c
        phi = self.phi

        zz = self.z[:15]
        N = self.N[:15]

        Dz = A * (1 - q) ** (np.abs(zz) - 1) + c

        alpha = Dz * phi
        beta = (1 - Dz) * phi

        dist = sp_betabinom(n=N, a=alpha, b=beta)
        std = np.sqrt(dist.var()) / N

        # Confidence interval with equal areas around the median (?mean?)
        # intervals1 = dist.interval(alpha=n_sigma_to_prob(n_sigma=1), a=alpha, b=beta)
        # intervals2 = dist.interval(alpha=n_sigma_to_prob(n_sigma=2), a=alpha, b=beta)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.z[:15], self.k[:15] / self.N[:15], "sk", label="Data Forward")
        ax.plot(-self.z[15:], self.k[15:] / self.N[15:], "xk", label="Data Reverse")

        ax.errorbar(
            zz,
            Dz,
            std,
            fmt="o",
            color="C2",
            elinewidth=2,
            capsize=5,
            capthick=2,
            lw=2,
            label="Fit",
        )

        # for i, intervals in enumerate([intervals1, intervals2]):

        #     ax.fill_between(
        #         zz,
        #         intervals[0],
        #         intervals[1],
        #         color="C2",
        #         alpha=0.1,
        #         label=f"CI, {i+1}σ, equal",
        #     )

        #     ax.plot(zz, intervals1[0], color="C2", ls="--")
        #     ax.plot(zz, intervals1[1], color="C2", ls="--")

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


#%%


def make_fits(fit_result, data):
    # reload(fits_frequentist)
    # frequentist_likelihood = fits_frequentist.Frequentist(data, method='likelihood')
    # frequentist_posterior = fits_frequentist.Frequentist(data, method="posterior")

    np.random.seed(42)

    frequentist = Frequentist(data, method="posterior")
    # frequentist.PMD.m
    # print(frequentist)
    # frequentist.plot()

    data_forward = {key: val[data["z"] > 0] for key, val in data.items()}
    data_reverse = {key: val[data["z"] < 0] for key, val in data.items()}

    frequentist_forward = Frequentist(data_forward, method="posterior")
    # print(frequentist_forward)
    # frequentist_forward.plot()
    frequentist_reverse = Frequentist(data_reverse, method="posterior")
    # print(frequentist_reverse)
    # frequentist_reverse.plot()

    vars_to_keep = [
        "lambda_LR",
        "D_max",
        "D_max_std",
        "q",
        "q_std",
        "phi",
        "phi_std",
        "A",
        "A_std",
        "c",
        "c_std",
        "rho_Ac",
        "lambda_LR_P",
        "lambda_LR_n_sigma",
        "valid",
    ]

    for var in vars_to_keep:
        fit_result[f"{var}"] = getattr(frequentist, var)

    numerator = frequentist_forward.D_max - frequentist_reverse.D_max
    delimiter = np.sqrt(
        frequentist_forward.D_max_std ** 2 + frequentist_reverse.D_max_std ** 2
    )
    fit_result["asymmetry"] = np.abs(numerator) / delimiter

    for var in vars_to_keep:
        fit_result[f"forward_{var}"] = getattr(frequentist_forward, var)

    for var in vars_to_keep:
        fit_result[f"reverse_{var}"] = getattr(frequentist_reverse, var)

    return frequentist, frequentist_forward, frequentist_reverse
