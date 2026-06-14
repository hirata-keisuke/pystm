"""Spectral initialization via anchor words (port of spectral.R).

Implements the method of Arora et al. (2013): build the word co-occurrence
gram matrix, greedily select anchor words, then recover the topic-word
matrix with RecoverL2.  The R package solves the simplex-constrained
regression exactly with quadprog by default; here we use a penalized NNLS
formulation which matches it closely.  The exponentiated gradient
algorithm (R's ``recoverEG=TRUE`` option) is available as an alternative.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.spatial import ConvexHull


def gram(mat: csr_matrix) -> np.ndarray:
    """Word co-occurrence gram matrix from a sparse doc-term matrix."""
    nd = np.asarray(mat.sum(axis=1)).ravel()
    keep = nd >= 2  # undefined for docs with fewer than 2 tokens
    mat = mat[keep]
    nd = nd[keep]
    divisor = nd * (nd - 1)

    htilde = mat.multiply(1.0 / np.sqrt(divisor)[:, None]).tocsr()
    hhat = np.asarray(mat.multiply(1.0 / divisor[:, None]).sum(axis=0)).ravel()
    Q = (htilde.T @ htilde).toarray()
    Q[np.diag_indices_from(Q)] -= hhat
    return Q


def fast_anchor(Qbar: np.ndarray, K: int) -> np.ndarray:
    """Greedy anchor word selection by stabilized Gram-Schmidt."""
    Qbar = Qbar.copy()
    basis = np.zeros(K, dtype=np.int64)
    row_squared_sums = (Qbar**2).sum(axis=1)

    for i in range(K):
        basis[i] = int(np.argmax(row_squared_sums))
        max_val = row_squared_sums[basis[i]]
        Qbar[basis[i]] *= 1.0 / np.sqrt(max_val)

        inner_products = Qbar @ Qbar[basis[i]]
        project = np.outer(inner_products, Qbar[basis[i]])
        project[basis[: i + 1]] = 0.0
        Qbar -= project
        row_squared_sums = (Qbar**2).sum(axis=1)
        row_squared_sums[basis[: i + 1]] = 0.0
    return basis


def tsne_anchor(Qbar: np.ndarray, random_state=None) -> np.ndarray:
    """Anchor word selection by the Lee & Mimno (2014) algorithm (K=0).

    Project the row-normalized gram matrix to three dimensions with
    t-SNE and take the vertices of the convex hull of the projection as
    the anchor words.  The number of anchors found determines the number
    of topics, so this is the data-driven way to choose K (R's
    ``K=0`` / ``tsneAnchor``).

    Duplicate rows of ``Qbar`` make t-SNE fail; following the R package
    we add a touch of noise to the non-zero entries of duplicate rows and
    renormalize before retrying.
    """
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(
        random_state if isinstance(random_state, (int, np.integer)) else None
    )

    def _project(mat):
        # perplexity must stay below the number of samples
        perplexity = min(30.0, max(5.0, (mat.shape[0] - 1) / 3.0))
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            init="pca",
            random_state=random_state
            if isinstance(random_state, (int, np.integer))
            else 0,
        )
        return tsne.fit_transform(mat)

    try:
        proj = _project(Qbar)
    except ValueError:
        # most likely caused by duplicate rows; jitter them and retry
        Qbar = Qbar.copy()
        _, inverse, counts = np.unique(
            Qbar, axis=0, return_inverse=True, return_counts=True
        )
        dup = np.where(counts[inverse] > 1)[0]
        for r in dup:
            row = Qbar[r]
            nz = row > 0
            row[nz] = rng.uniform(0.0, 1e-5, size=int(nz.sum()))
            s = row.sum()
            if s > 0:
                row /= s
            Qbar[r] = row
        proj = _project(Qbar)

    hull = ConvexHull(proj)
    anchors = np.sort(np.unique(hull.vertices))
    return anchors


def expgrad(X, y, XtX=None, alpha=None, tol=1e-7, max_iter=500):
    """Exponentiated gradient for simplex-constrained least squares."""
    if alpha is None:
        alpha = np.full(X.shape[0], 1.0 / X.shape[0])
    if XtX is None:
        XtX = X @ X.T
    ytX = y @ X.T

    eta = 50.0
    sse_old = np.inf
    for _ in range(max_iter):
        grad = ytX - alpha @ XtX
        sse = grad @ grad
        grad = 2.0 * eta * grad
        alpha = alpha * np.exp(grad - grad.max())
        alpha = alpha / alpha.sum()
        if abs(np.sqrt(sse_old) - np.sqrt(sse)) < tol:
            break
        sse_old = sse
    return alpha


def recover_l2(Qbar, anchors, wprob, solver="nnls"):
    """Recover the K x V topic-word matrix from the anchor rows.

    Each word's row of ``Qbar`` is expressed as a convex combination of
    the anchor rows.  ``solver="nnls"`` enforces the sum-to-one constraint
    through a heavily weighted penalty row in a non-negative least-squares
    problem (the analogue of the exact quadprog solve used by the R
    package); ``solver="expgrad"`` uses exponentiated gradient descent
    (R's ``recoverEG=TRUE``).
    """
    X = Qbar[anchors]
    XtX = X @ X.T
    K = len(anchors)
    anchor_pos = {a: idx for idx, a in enumerate(anchors)}

    if solver == "nnls":
        penalty = 1000.0
        X_aug = np.vstack([X.T, np.full(K, penalty)])

    weights = np.empty((Qbar.shape[0], K))
    for i in range(Qbar.shape[0]):
        if i in anchor_pos:
            vec = np.zeros(K)
            vec[anchor_pos[i]] = 1.0
            weights[i] = vec
        elif solver == "nnls":
            solution, _ = nnls(X_aug, np.append(Qbar[i], penalty))
            solution = np.maximum(solution, np.finfo(np.float64).eps)
            weights[i] = solution / solution.sum()
        else:
            solution = expgrad(X, Qbar[i], XtX)
            solution = np.maximum(solution, np.finfo(np.float64).eps)
            weights[i] = solution

    A = weights * wprob[:, None]
    A = A.T / A.sum(axis=0)[None, :].T
    return A


def spectral_init(X: csr_matrix, K: int, max_vocab: int | None = 10000,
                  solver: str = "nnls", random_state=None):
    """Spectral initialization of beta (the Spectral branch of stm.init).

    ``K=0`` triggers the Lee & Mimno (2014) data-driven topic count: the
    anchors (and hence K) are chosen by :func:`tsne_anchor`.  The number
    of topics actually used is always returned alongside ``beta`` so the
    caller can size the remaining parameters.

    Returns
    -------
    beta : ndarray of shape (K, V)
    K : int
        The number of topics, equal to the input ``K`` unless ``K=0`` was
        requested, in which case it is the number of anchors found.
    """
    V = X.shape[1]
    if K >= V:
        raise ValueError(
            "Spectral initialization cannot be used when K >= vocabulary size."
        )

    wprob = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)
    wprob /= wprob.sum()

    keep = None
    if max_vocab is not None and V > max_vocab:
        keep = np.argsort(wprob)[::-1][:max_vocab]
        X = X[:, keep]
        wprob = wprob[keep]

    Q = gram(X)
    Qsums = Q.sum(axis=1)
    if np.any(Qsums == 0):
        nonzero = Qsums != 0
        keep = np.where(nonzero)[0] if keep is None else keep[nonzero]
        Q = Q[np.ix_(nonzero, nonzero)]
        Qsums = Qsums[nonzero]
        wprob = wprob[nonzero]
    Qbar = Q / Qsums[:, None]

    if K == 0:
        anchors = tsne_anchor(Qbar, random_state=random_state)
        K = len(anchors)
    else:
        anchors = fast_anchor(Qbar, K)
    beta = recover_l2(Qbar, anchors, wprob, solver=solver)

    if keep is not None:
        # reintroduce dropped words with a small amount of mass
        beta_new = np.zeros((K, V))
        beta_new[:, keep] = beta
        beta_new += 0.001 / V
        beta = beta_new / beta_new.sum(axis=1, keepdims=True)
    return beta, K
