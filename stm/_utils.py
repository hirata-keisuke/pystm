"""Shared numerical utilities (port of STMfunctions.R)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.special import logsumexp


def safelog(x: np.ndarray, min_value: float = -1000.0) -> np.ndarray:
    """log(x) with -inf (and anything below ``min_value``) clamped."""
    with np.errstate(divide="ignore"):
        out = np.log(x)
    return np.maximum(out, min_value)


def row_softmax(mat: np.ndarray) -> np.ndarray:
    """Row-wise softmax of a 2-d array."""
    return np.exp(mat - logsumexp(mat, axis=1, keepdims=True))


def to_doc_list(X) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a document-term count matrix to the stm document format.

    Returns one ``(word_indices, word_counts)`` pair per document, holding
    the column indices of the document's distinct terms and their counts
    (the two rows of the R package's document matrices, zero-indexed).
    """
    X = csr_matrix(X) if not issparse(X) else X.tocsr()
    if (X.data < 0).any():
        raise ValueError("X must contain non-negative counts.")
    if np.any(X.data != np.round(X.data)):
        raise ValueError("X must contain integer counts.")
    docs = []
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i + 1]
        words = X.indices[start:end].astype(np.int64)
        counts = X.data[start:end].astype(np.float64)
        keep = counts > 0
        docs.append((words[keep], counts[keep]))
    return docs
