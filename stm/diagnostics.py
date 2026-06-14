"""Model diagnostics (ports of semanticCoherence.R, exclusivity.R,
residuals.R and topicCorr.R)."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.sparse import csc_matrix, csr_matrix, issparse
from scipy.stats import rankdata

from ._utils import safelog


def _as_csr(X):
    return csr_matrix(X) if not issparse(X) else X.tocsr()


def _semcoh_one_beta(X, logbeta, M):
    """Semantic coherence per topic for one beta (semCoh1beta in R)."""
    K = logbeta.shape[0]
    top_words = np.argsort(-logbeta, axis=1)[:, :M]
    wordlist, positions = np.unique(top_words, return_inverse=True)
    labels = positions.reshape(K, M)

    sub = csc_matrix(X[:, wordlist])
    sub.data = np.minimum(sub.data, 1.0)  # binarize
    cross = (sub.T @ sub).toarray()  # document co-occurrence counts

    result = np.zeros(K)
    for k in range(K):
        idx = labels[k]
        for a in range(M):
            for b in range(M):
                m_, l_ = idx[a], idx[b]
                if m_ > l_:
                    result[k] += (np.log(0.01 + cross[m_, l_])
                                  - np.log(cross[l_, l_] + 0.01))
    return result


def semantic_coherence(model, X, content=None, M=10):
    """Semantic coherence (Mimno et al. 2011) per topic.

    Higher is better; the metric checks that a topic's top ``M`` words
    co-occur within documents.  For content covariate models pass the
    content labels; the score is then the document-weighted average over
    the aspect-specific betas, as in the R package.
    """
    X = _as_csr(X)
    if model.aspect_components_ is None:
        return _semcoh_one_beta(X, safelog(model.components_), M)
    if content is None:
        raise ValueError(
            "The model was fitted with a content covariate; pass the "
            "matching content labels."
        )
    levels = model.content_levels_
    index = np.searchsorted(levels, np.asarray(content).ravel())
    result = np.zeros(model.components_.shape[0])
    for a in range(len(levels)):
        subset = index == a
        if not subset.any():
            continue
        logbeta = safelog(model.aspect_components_[a])
        result += _semcoh_one_beta(X[subset], logbeta, M) * subset.sum()
    return result / X.shape[0]


def exclusivity(model, M=10, frexw=0.7):
    """FREX-based exclusivity per topic (exclusivity in R).

    Not defined for content covariate models (matching the R package).
    """
    if model.aspect_components_ is not None:
        raise ValueError(
            "Exclusivity calculation is only designed for models without "
            "content covariates."
        )
    w = frexw
    tbeta = model.components_.T  # (V, K)
    mat = tbeta / tbeta.sum(axis=1, keepdims=True)

    ex = np.apply_along_axis(rankdata, 0, mat) / mat.shape[0]
    fr = np.apply_along_axis(rankdata, 0, tbeta) / mat.shape[0]
    frex = 1.0 / (w / ex + (1 - w) / fr)
    index = np.argsort(-tbeta, axis=0)[:M]
    return np.array([
        frex[index[:, k], k].sum() for k in range(tbeta.shape[1])
    ])


def check_residuals(model, X, content=None, tol=0.01):
    """Multinomial dispersion of the residuals (Taddy 2012).

    Under a correctly specified model the dispersion is 1; values above 1
    suggest the number of topics is too small.  Returns a dict with
    ``dispersion``, ``pvalue`` and ``df``.
    """
    X = _as_csr(X)
    n, V = X.shape
    K = model.components_.shape[0]
    theta = model.theta_
    if theta.shape[0] != n:
        raise ValueError("X must be the corpus the model was fitted on.")

    beta = model._beta_list()
    if model.content_levels_ is not None:
        if content is None:
            raise ValueError(
                "The model was fitted with a content covariate; pass the "
                "matching content labels."
            )
        index = np.searchsorted(model.content_levels_,
                                np.asarray(content).ravel())
    else:
        index = np.zeros(n, dtype=np.int64)

    d = n * (K - 1) + K * (V - 1)
    D = 0.0
    Nhat = 0
    for i in range(n):
        row = X.getrow(i)
        q = theta[i] @ beta[index[i]]  # (V,)
        m = row.sum()
        Nhat += int((q * m > tol).sum())
        x = np.zeros(V)
        x[row.indices] = row.data
        denom = m * q * (1 - q)
        D += ((x**2 - 2 * x * q * m) / denom).sum() + (m * q / (1 - q)).sum()

    df = Nhat - V - d
    with np.errstate(invalid="ignore"):
        dispersion = D / df
        pvalue = stats.chi2.sf(D, df) if df > 0 else np.nan
    return {"dispersion": dispersion, "pvalue": pvalue, "df": df}


class TopicCorrelations:
    """Result of :func:`topic_corr` (class topicCorr in R).

    Attributes
    ----------
    cor : ndarray (K, K)
        Correlation matrix with entries below ``cutoff`` (in absolute
        value) set to zero.
    posadj : ndarray (K, K)
        Adjacency matrix of positive correlations above the cutoff.
    poscor : ndarray (K, K)
        Correlations masked to the positive adjacency structure.
    """

    def __init__(self, cor, posadj, poscor):
        self.cor = cor
        self.posadj = posadj
        self.poscor = poscor


def topic_corr(model, cutoff=0.01):
    """Topic correlation graph from theta (topicCorr method="simple").

    The R package's "huge" method (semiparametric graphical model
    selection) depends on the huge package and is not implemented.
    """
    cormat = np.corrcoef(model.theta_, rowvar=False)
    posadj = (cormat > cutoff).astype(float)
    poscor = cormat * posadj
    cor = np.where(np.abs(cormat) > cutoff, cormat, 0.0)
    return TopicCorrelations(cor=cor, posadj=posadj, poscor=poscor)
