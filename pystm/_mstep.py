"""M-step updates (port of STMmu.R, STMsigma.R, STMoptbeta.R)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky


class PrevalenceRegressionError(RuntimeError):
    pass


def vb_variational_reg(Y, X, Xcorr=None, b0=1.0, d0=1.0, max_iter=1000):
    """Variational linear regression with a half-Cauchy hyperprior.

    Port of vb.variational.reg(); the first column of ``X`` is assumed to
    be the (unpenalized) intercept.
    """
    if Xcorr is None:
        Xcorr = X.T @ X
    XYcorr = X.T @ Y

    N, D = X.shape
    an = (1 + N) / 2
    w = np.zeros(D)
    error_prec = 1.0
    cn = D
    dn = 1.0
    Ea = cn / dn
    ba = 1.0

    for _ in range(max_iter):
        w_old = w

        prior_diag = np.full(D, Ea)
        prior_diag[0] = 0.0
        invV = error_prec * Xcorr + np.diag(prior_diag)
        L = cholesky(invV, lower=True)
        V = cho_solve((L, True), np.eye(D))
        w = error_prec * (V @ XYcorr)

        sse = np.sum((X @ w - Y) ** 2)
        bn = 0.5 * (sse + np.trace(Xcorr @ V)) + ba
        error_prec = an / bn
        ba = 1.0 / (error_prec + b0)

        da = 2.0 / (Ea + d0)
        dn = 2.0 * da + (w[1:] @ w[1:] + np.diag(V)[1:].sum())
        Ea = cn / dn

        if np.abs(w - w_old).sum() < 1e-4:
            return w

    raise PrevalenceRegressionError(
        "Prevalence regression failed to converge within the iteration "
        "limit. You can raise it with gamma_max_iter."
    )


def opt_mu(lambda_, covar=None, max_iter=1000):
    """Update the prevalence model (opt.mu, modes CTM and Pooled).

    Returns ``(mu, gamma)``.  Without covariates (CTM mode) ``mu`` is the
    shared (K-1,) mean and ``gamma`` is None.  With covariates ``mu`` is
    (N, K-1), the per-document prior means, and ``gamma`` is (P, K-1).
    """
    if covar is None:
        return lambda_.mean(axis=0), None

    Xcorr = covar.T @ covar
    gamma = np.column_stack([
        vb_variational_reg(lambda_[:, k], covar, Xcorr=Xcorr, max_iter=max_iter)
        for k in range(lambda_.shape[1])
    ])
    mu = covar @ gamma
    return mu, gamma


def opt_sigma(nu, lambda_, mu, sigprior):
    """Update the global covariance matrix (opt.sigma).

    ``mu`` is (K-1,) for the shared mean or (N, K-1) for covariate models.
    """
    if mu.ndim == 1:
        diff = lambda_ - mu[None, :]
    else:
        diff = lambda_ - mu
    sigma = (diff.T @ diff + nu) / lambda_.shape[0]
    return sigprior * np.diag(np.diag(sigma)) + (1 - sigprior) * sigma


def opt_beta(beta_ss):
    """Update the topic-word distributions (opt.beta, LDA-beta mode)."""
    row_sums = beta_ss[0].sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return [beta_ss[0] / row_sums]
