"""Choosing the number of topics (ports of searchK.R and heldout.R)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse, lil_matrix
from scipy.special import gammaln
from sklearn.utils import check_random_state

from ._utils import safelog
from .diagnostics import check_residuals, exclusivity, semantic_coherence
from .stm import StructuralTopicModel


def make_heldout(X, N=None, proportion=0.5, random_state=None):
    """Hold out a fraction of tokens for document completion (make.heldout).

    Selects ``N`` documents (default 10%) and removes ``proportion`` of
    each one's tokens.  Held-out tokens whose word disappears entirely
    from the training corpus are dropped (mirroring the R package's vocab
    remapping).

    Returns a dict with ``X_train`` (csr matrix), ``index`` (held-out
    document ids) and ``docs`` (list of ``(word_indices, counts)``).
    """
    X = csr_matrix(X) if not issparse(X) else X.tocsr()
    rng = check_random_state(random_state)
    n_docs, V = X.shape
    if N is None:
        N = int(np.floor(0.1 * n_docs))
    if not 0 < N <= n_docs:
        raise ValueError("N must be between 1 and the number of documents.")
    if not 0 < proportion < 1:
        raise ValueError("proportion must be in (0, 1).")

    index = np.sort(rng.choice(n_docs, size=N, replace=False))
    X_train = lil_matrix(X.copy())
    missing_docs = []
    kept_index = []
    for i in index:
        row = X.getrow(i)
        if row.indices.shape[0] < 2:
            continue  # too few distinct words to split (as in R)
        tokens = np.repeat(row.indices, row.data.astype(np.int64))
        nsamp = max(1, int(np.floor(proportion * tokens.shape[0])))
        nsamp = min(nsamp, tokens.shape[0] - 1)  # keep the doc non-empty
        held = rng.choice(tokens.shape[0], size=nsamp, replace=False)
        held_counts = np.bincount(tokens[held], minlength=V)
        words = np.flatnonzero(held_counts)
        for w in words:
            X_train[i, w] -= held_counts[w]
        missing_docs.append((words, held_counts[words].astype(np.float64)))
        kept_index.append(i)
    X_train = csr_matrix(X_train)
    X_train.eliminate_zeros()

    # drop held-out tokens of words that vanished from the training corpus
    train_wcounts = np.asarray(X_train.sum(axis=0)).ravel()
    final_docs, final_index = [], []
    for i, (words, counts) in zip(kept_index, missing_docs):
        keep = train_wcounts[words] > 0
        if keep.any():
            final_docs.append((words[keep], counts[keep]))
            final_index.append(i)
    return {
        "X_train": X_train,
        "index": np.asarray(final_index, dtype=np.int64),
        "docs": final_docs,
    }


def eval_heldout(model, heldout, content=None):
    """Heldout log-likelihood by document completion (eval.heldout).

    Uses the fitted theta of each partially observed document to score
    its held-out tokens.  Returns a dict with ``expected_heldout`` (the
    mean over documents of the mean per-token log-probability) and the
    per-document values ``doc_heldout``.
    """
    beta = model._beta_list()
    if model.content_levels_ is not None:
        if content is None:
            raise ValueError(
                "The model was fitted with a content covariate; pass the "
                "matching content labels."
            )
        index_all = np.searchsorted(model.content_levels_,
                                    np.asarray(content).ravel())
    else:
        index_all = np.zeros(model.theta_.shape[0], dtype=np.int64)

    doc_scores = np.empty(len(heldout["docs"]))
    for j, (doc_id, (words, counts)) in enumerate(
            zip(heldout["index"], heldout["docs"])):
        logprobs = safelog(
            model.theta_[doc_id] @ beta[index_all[doc_id]][:, words]
        )
        doc_scores[j] = np.repeat(logprobs, counts.astype(np.int64)).mean()
    return {
        "expected_heldout": float(np.nanmean(doc_scores)),
        "doc_heldout": doc_scores,
    }


def search_k(X, K_values, *, prevalence=None, content=None, N=None,
             proportion=0.5, M=10, heldout_random_state=None,
             model_params=None, verbose=False):
    """Fit models over a grid of K and compute diagnostics (searchK).

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
    K_values : iterable of int
        Topic numbers to evaluate.
    prevalence, content : optional
        Covariates forwarded to :meth:`StructuralTopicModel.fit`.
    N, proportion : heldout construction parameters (see make_heldout).
    M : int, default=10
        Number of top words for exclusivity / semantic coherence.
    model_params : dict, optional
        Extra keyword arguments for the StructuralTopicModel constructor.

    Returns
    -------
    dict of arrays keyed by "K", "heldout", "residual", "bound",
    "lbound", "exclus", "semcoh", "em_its" (exclusivity and semantic
    coherence are omitted for content covariate models, as in R).
    """
    K_values = list(K_values)
    model_params = dict(model_params or {})
    heldout = make_heldout(X, N=N, proportion=proportion,
                           random_state=heldout_random_state)

    results = {key: [] for key in
               ("K", "heldout", "residual", "bound", "lbound",
                "exclus", "semcoh", "em_its")}
    for K in K_values:
        if verbose:
            print(f"searchK: fitting K={K} ...")
        model = StructuralTopicModel(n_components=K, **model_params)
        model.fit(heldout["X_train"], prevalence=prevalence, content=content)

        results["K"].append(K)
        results["heldout"].append(
            eval_heldout(model, heldout, content=content)["expected_heldout"]
        )
        results["residual"].append(
            check_residuals(model, heldout["X_train"],
                            content=content)["dispersion"]
        )
        bound = max(model.bound_)
        results["bound"].append(bound)
        results["lbound"].append(bound + gammaln(K + 1))
        if content is None:
            results["exclus"].append(
                float(np.mean(exclusivity(model, M=M, frexw=0.7)))
            )
            results["semcoh"].append(
                float(np.mean(semantic_coherence(model, heldout["X_train"],
                                                 M=M)))
            )
        results["em_its"].append(model.n_iter_)

    if content is not None:
        del results["exclus"], results["semcoh"]
    return {key: np.asarray(val) for key, val in results.items()}
