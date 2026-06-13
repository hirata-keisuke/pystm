"""Covariate effect estimation (port of estimateEffect.R / thetaPosterior.R).

Regressions where topic proportions are the outcome, propagating the
measurement uncertainty of theta via the method of composition: draw
theta from the variational posterior, run the OLS, repeat, then pool.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state

from ._utils import row_softmax


def _global_sigma(model):
    """Global approximation to the per-document posterior covariance.

    Subtracts the contribution of deviations from the prior mean out of
    the topic covariance, leaving the (average) variational covariance
    (thetapost.global in R).
    """
    lambda_ = model.eta_
    mu = model.mu_
    diff = lambda_ - (mu[None, :] if mu.ndim == 1 else mu)
    covariance = (diff.T @ diff) / lambda_.shape[0]
    sigma = model.sigma_ - covariance
    # guard against indefiniteness from the subtraction
    evals, evecs = np.linalg.eigh(sigma)
    if evals[0] <= 0:
        evals = np.maximum(evals, 1e-10)
        sigma = (evecs * evals) @ evecs.T
    return sigma


def _draw_theta(model, rng):
    """One draw of theta for every document (Global approximation)."""
    sigma = _global_sigma(model)
    chol = np.linalg.cholesky(sigma)
    z = rng.standard_normal(model.eta_.shape)
    eta = model.eta_ + z @ chol.T
    return row_softmax(
        np.column_stack([eta, np.zeros(eta.shape[0])])
    )


class _QRRegression:
    """OLS with a cached QR decomposition (qr.lm / summary.qr.lm in R)."""

    def __init__(self, xmat, prior=None):
        self.n_obs = xmat.shape[0]
        p = xmat.shape[1]
        if prior is not None:
            if np.isscalar(prior):
                prior = np.diag(np.full(p, float(prior)))
            xmat = np.vstack([xmat, np.linalg.cholesky(prior).T])
        if np.linalg.matrix_rank(xmat) < p:
            warnings.warn(
                "Covariate matrix is singular; adding a small ridge prior "
                "(1e-5) for numerical stability.", stacklevel=3,
            )
            xmat = np.vstack([xmat, np.sqrt(1e-5) * np.eye(p)])
        self.xmat = xmat
        self.q, self.r = np.linalg.qr(xmat)
        self.rinv = np.linalg.inv(self.r)
        self.df_residual = xmat.shape[0] - p

    def fit(self, y):
        if y.shape[0] != self.xmat.shape[0]:
            y = np.concatenate(
                [y, np.zeros(self.xmat.shape[0] - y.shape[0])]
            )
        coef = self.rinv @ (self.q.T @ y)
        resid = y - self.xmat @ coef
        resvar = (resid @ resid) / self.df_residual
        vcov = resvar * (self.rinv @ self.rinv.T)
        return coef, vcov


class EstimatedEffects:
    """Result of :func:`estimate_effect`.

    Attributes
    ----------
    parameters : dict
        Maps topic index to a list of ``(coef, vcov)`` pairs, one per
        composition draw.
    topics : list of int
        Topics for which effects were estimated (0-based).
    n_obs : int
        Number of documents.
    n_params : int
        Number of regression coefficients (including the intercept).
    """

    def __init__(self, parameters, topics, n_obs, n_params):
        self.parameters = parameters
        self.topics = topics
        self.n_obs = n_obs
        self.n_params = n_params

    def summary(self, topics=None, nsim=500, random_state=None):
        """Pooled coefficient tables (summary.estimateEffect in R).

        Returns a dict mapping topic index to a record array with fields
        ``estimate``, ``std_error``, ``t_value`` and ``p_value``, one row
        per regression coefficient.
        """
        rng = check_random_state(random_state)
        topics = self.topics if topics is None else list(topics)
        dtype = [("estimate", float), ("std_error", float),
                 ("t_value", float), ("p_value", float)]
        tables = {}
        for k in topics:
            if k not in self.parameters:
                raise ValueError(f"Topic {k} was not estimated.")
            sims = np.vstack([
                rng.multivariate_normal(est, vcov, size=nsim)
                for est, vcov in self.parameters[k]
            ])
            est = sims.mean(axis=0)
            se = sims.std(axis=0, ddof=1)
            tval = est / se
            rdf = self.n_obs - self.n_params
            p = 2 * stats.t.sf(np.abs(tval), rdf)
            table = np.zeros(len(est), dtype=dtype)
            table["estimate"] = est
            table["std_error"] = se
            table["t_value"] = tval
            table["p_value"] = p
            tables[k] = table
        return tables


def estimate_effect(model, prevalence, topics=None,
                    uncertainty="Global", nsims=25, prior=None,
                    random_state=None):
    """Regress topic proportions on covariates (estimateEffect in R).

    Parameters
    ----------
    model : fitted StructuralTopicModel
    prevalence : array-like of shape (n_samples, n_covariates)
        Covariate design matrix for the regression; an intercept column
        is added automatically.  Should normally contain (at least) the
        covariates used when fitting the model.  Categorical variables
        must be encoded numerically beforehand.
    topics : iterable of int, optional
        0-based topic indices to estimate effects for (default: all).
    uncertainty : {"Global", "None"}, default="Global"
        "Global" draws theta from the variational posterior using a
        globally shared covariance approximation; "None" uses the MAP
        theta without measurement uncertainty.  (The R package's "Local"
        method is not implemented.)
    nsims : int, default=25
        Number of method-of-composition draws ("Global" only).
    prior : float or ndarray, optional
        Ridge penalty (scalar or full precision matrix) added to the
        regression for numerical stability.
    """
    if not hasattr(model, "theta_"):
        raise ValueError("model must be a fitted StructuralTopicModel.")
    if uncertainty not in ("Global", "None"):
        raise ValueError(
            "uncertainty must be 'Global' or 'None' ('Local' is not "
            "implemented; 'Global' is the recommended method)."
        )
    rng = check_random_state(random_state)
    K = model.theta_.shape[1]
    topics = list(range(K)) if topics is None else list(topics)
    if any(k < 0 or k >= K for k in topics):
        raise ValueError("topics must be 0-based indices below n_components.")

    xmat = np.asarray(prevalence, dtype=np.float64)
    if xmat.ndim == 1:
        xmat = xmat[:, None]
    if xmat.shape[0] != model.theta_.shape[0]:
        raise ValueError(
            "prevalence has a different number of rows than the fitted "
            "documents."
        )
    if not np.allclose(xmat[:, 0], 1.0):
        xmat = np.column_stack([np.ones(xmat.shape[0]), xmat])

    reg = _QRRegression(xmat, prior=prior)
    if uncertainty == "None":
        nsims = 1

    parameters = {k: [] for k in topics}
    for _ in range(nsims):
        if uncertainty == "None":
            theta = model.theta_
        else:
            theta = _draw_theta(model, rng)
        for k in topics:
            parameters[k].append(reg.fit(theta[:, k]))

    return EstimatedEffects(parameters, topics,
                            n_obs=model.theta_.shape[0],
                            n_params=xmat.shape[1])
