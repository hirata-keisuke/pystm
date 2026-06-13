"""Variational E-step (port of STMestep.R, STMlncpp.R and STMCfuns.cpp).

For each document the variational posterior over eta (the K-1 dimensional
logistic-normal document-topic parameter) is approximated by a Laplace
approximation: the mode ``lambda`` is found with BFGS and the covariance
``nu`` is the inverse Hessian at the mode.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve
from scipy.optimize import minimize

# Numerical guards absent from the R/C++ original: BFGS line searches can
# probe points where exp(eta) over/underflows or a document word has
# (numerically) zero probability under every topic, poisoning the search
# with inf/nan.  Clipping eta and flooring the logs/denominators keeps the
# objective finite without affecting values in the normal range.
_ETA_CLIP = 200.0
_TINY = 1e-300


def _expeta(eta):
    return np.append(np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP)), 1.0)


def neg_lhood(eta, beta_d, doc_ct, mu_d, siginv):
    """Negative collapsed objective for one document (lhoodcpp)."""
    expeta = _expeta(eta)
    ndoc = doc_ct.sum()
    word_probs = np.maximum(expeta @ beta_d, _TINY)
    part1 = np.log(word_probs) @ doc_ct - ndoc * np.log(expeta.sum())
    diff = eta - mu_d
    part2 = 0.5 * diff @ siginv @ diff
    return part2 - part1


def neg_grad(eta, beta_d, doc_ct, mu_d, siginv):
    """Gradient of :func:`neg_lhood` (gradcpp)."""
    expeta = _expeta(eta)
    EB = beta_d * expeta[:, None]
    denom = np.maximum(EB.sum(axis=0), _TINY)
    part1 = EB @ (doc_ct / denom) - (doc_ct.sum() / expeta.sum()) * expeta
    part2 = siginv @ (eta - mu_d)
    return part2 - part1[:-1]


def hessian_phi_bound(eta, beta_d, doc_ct, mu_d, siginv, sigmaentropy):
    """Compute the Laplace covariance, token assignments and bound (hpbcpp).

    Returns ``(phis, nu, bound)`` where ``phis`` is K x V_d expected token
    counts per topic, ``nu`` is the (K-1) x (K-1) posterior covariance of
    eta, and ``bound`` is the document's contribution to the global ELBO.
    """
    expeta = _expeta(eta)
    theta = expeta / expeta.sum()
    ndoc = doc_ct.sum()
    sqrtct = np.sqrt(doc_ct)

    EB = beta_d * expeta[:, None]
    EB *= (sqrtct / np.maximum(EB.sum(axis=0), _TINY))[None, :]

    hess = EB @ EB.T - ndoc * np.outer(theta, theta)
    # turn EB into phi (expected token counts per topic and word)
    EB *= sqrtct[None, :]
    np.fill_diagonal(hess, np.diag(hess) - (EB.sum(axis=1) - ndoc * theta))
    hess = hess[:-1, :-1] + siginv

    try:
        L = cholesky(hess, lower=True)
    except np.linalg.LinAlgError:
        # not positive definite: enforce diagonal dominance as in hpbcpp
        dvec = np.diag(hess).copy()
        magnitudes = np.abs(hess).sum(axis=1) - np.abs(dvec)
        dvec = np.maximum(dvec, magnitudes)
        np.fill_diagonal(hess, dvec)
        L = cholesky(hess, lower=True)

    det_term = -np.log(np.diag(L)).sum()
    nu = cho_solve((L, True), np.eye(hess.shape[0]))

    diff = eta - mu_d
    bound = (
        np.log(np.maximum(theta @ beta_d, _TINY)) @ doc_ct
        + det_term
        - 0.5 * diff @ siginv @ diff
        - sigmaentropy
    )
    return EB, nu, bound


def optimize_document(eta, beta_d, doc_ct, mu_d, siginv, sigmaentropy,
                      max_optim_iter=500):
    """Infer one document's variational parameters (logisticnormalcpp)."""
    res = minimize(
        neg_lhood,
        eta,
        args=(beta_d, doc_ct, mu_d, siginv),
        jac=neg_grad,
        method="BFGS",
        options={"maxiter": max_optim_iter},
    )
    eta_hat = res.x if np.isfinite(res.x).all() else eta
    return (hessian_phi_bound(eta_hat, beta_d, doc_ct, mu_d, siginv,
                              sigmaentropy), eta_hat)


def decompose_sigma(sigma):
    """Precompute the inverse and entropy term shared by all documents."""
    try:
        chol_u = cholesky(sigma, lower=False)
        sigmaentropy = np.log(np.diag(chol_u)).sum()
        siginv = cho_solve((chol_u, False), np.eye(sigma.shape[0]))
    except np.linalg.LinAlgError:
        sigmaentropy = 0.5 * np.linalg.slogdet(sigma)[1]
        siginv = solve(sigma, np.eye(sigma.shape[0]), assume_a="sym")
    return siginv, sigmaentropy


def estep(docs, beta_index, update_mu, beta, lambda_old, mu, sigma,
          max_optim_iter=500):
    """Run the E-step over all documents and accumulate sufficient stats.

    Parameters mirror estep() in STMestep.R.  ``docs`` is a list of
    ``(word_indices, counts)`` pairs, ``beta`` a list with one K x V matrix
    per content level (always length 1 here), ``mu`` is (K-1,) when shared
    or (N, K-1) when document specific.

    Returns ``(sigma_ss, beta_ss, bound, lambda_)``.
    """
    K, V = beta[0].shape
    N = len(docs)
    A = len(beta)

    sigma_ss = np.zeros((K - 1, K - 1))
    beta_ss = [np.zeros((K, V)) for _ in range(A)]
    bound = np.empty(N)
    lambda_ = np.empty((N, K - 1))

    siginv, sigmaentropy = decompose_sigma(sigma)

    for i, (words, counts) in enumerate(docs):
        aspect = beta_index[i]
        init = lambda_old[i]
        mu_d = mu[i] if update_mu else mu
        beta_d = np.ascontiguousarray(beta[aspect][:, words])

        (phis, nu, bnd), eta_hat = optimize_document(
            init, beta_d, counts, mu_d, siginv, sigmaentropy,
            max_optim_iter=max_optim_iter,
        )

        sigma_ss += nu
        beta_ss[aspect][:, words] += phis
        bound[i] = bnd
        lambda_[i] = eta_hat

    return sigma_ss, beta_ss, bound, lambda_
