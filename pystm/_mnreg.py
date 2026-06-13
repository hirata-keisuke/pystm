"""Content covariate M-step (port of STMmnreg.R).

The SAGE-style topic-word update with content covariates is estimated via
Distributed Multinomial Regression (Taddy 2013): the multinomial is
factorized into independent Poisson regressions, one per vocabulary word,
each with an L1 penalty.  The R package solves these with glmnet; here we
implement an equivalent lasso-penalized Poisson solver (IRLS + coordinate
descent over a regularization path with information-criterion selection).

The solver exploits the structure of the problem heavily.  The design
matrix consists of three groups of indicator columns (topic main effects,
aspect main effects, topic-by-aspect interactions).  Columns within a
group touch disjoint rows, so a coordinate-descent pass over a whole
group can be performed as one vectorized update; and since every word
shares the same design, all V regressions are advanced simultaneously.
"""

from __future__ import annotations

import numpy as np
from scipy.special import xlogy


def _poisson_deviance(Y, mu):
    """Per-word Poisson deviance, with the y=0 terms handled."""
    return 2.0 * (xlogy(Y, Y) - xlogy(Y, mu) - (Y - mu)).sum(axis=(0, 1))


def _soft_threshold(rho, lam):
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)


class _StructuredPoissonLasso:
    """Distributed Poisson lasso with the STM content design.

    Data is held as (A, K, V) arrays.  ``lam`` arguments are per-word
    penalty levels of shape (V,).
    """

    def __init__(self, Y3, offsets3, use_aspect, use_inter):
        self.Y3 = Y3
        self.offsets3 = offsets3
        self.A, self.K, self.V = Y3.shape
        self.n = self.A * self.K
        self.use_aspect = use_aspect
        self.use_inter = use_inter
        self.b_topic = np.zeros((self.K, self.V))
        self.b_aspect = np.zeros((self.A, self.V)) if use_aspect else None
        self.b_inter = (
            np.zeros((self.A, self.K, self.V)) if use_inter else None
        )

    def linpred(self):
        eta = self.offsets3 + self.b_topic[None, :, :]
        if self.use_aspect:
            eta = eta + self.b_aspect[:, None, :]
        if self.use_inter:
            eta = eta + self.b_inter
        return np.clip(eta, -50.0, 30.0)

    def _block_update(self, R, W, Wsum, b, lam, axis):
        """One exact CD pass over a group of disjoint indicator columns.

        ``axis`` is the (A, K, V) axis summed over to aggregate a column's
        rows (None for the interaction block where each row is its own
        column).  Updates ``R`` in place and returns (new_b, max_delta).
        """
        if axis is None:
            num = R + W * b
            denom = Wsum
        else:
            num = R.sum(axis=axis) + Wsum * b
            denom = Wsum
        rho = num / self.n
        b_new = _soft_threshold(rho, lam)
        with np.errstate(invalid="ignore", divide="ignore"):
            b_new = np.where(denom / self.n > 1e-12,
                             b_new / (denom / self.n), 0.0)
        delta = b_new - b
        max_delta = np.abs(delta).max() if delta.size else 0.0
        if max_delta > 0:
            if axis is None:
                R -= W * delta
            else:
                R -= W * np.expand_dims(delta, axis)
        return b_new, max_delta

    def fit_one_lambda(self, lam, tol, max_irls, max_sweeps):
        """Solve at one penalty level, warm-starting from current state."""
        for _ in range(max_irls):
            eta = self.linpred()
            mu = np.exp(eta)
            W = mu
            R = self.Y3 - mu
            Wsum_topic = W.sum(axis=0)
            Wsum_aspect = W.sum(axis=1) if self.use_aspect else None
            outer_delta = 0.0
            for _ in range(max_sweeps):
                d = 0.0
                self.b_topic, d1 = self._block_update(
                    R, W, Wsum_topic, self.b_topic, lam, axis=0)
                d = max(d, d1)
                if self.use_aspect:
                    self.b_aspect, d1 = self._block_update(
                        R, W, Wsum_aspect, self.b_aspect, lam, axis=1)
                    d = max(d, d1)
                if self.use_inter:
                    self.b_inter, d1 = self._block_update(
                        R, W, W, self.b_inter, lam, axis=None)
                    d = max(d, d1)
                outer_delta = max(outer_delta, d)
                if d < tol:
                    break
            if outer_delta < tol:
                break

    def df(self):
        out = (self.b_topic != 0).sum(axis=0)
        if self.use_aspect:
            out = out + (self.b_aspect != 0).sum(axis=0)
        if self.use_inter:
            out = out + (self.b_inter != 0).sum(axis=(0, 1))
        return out

    def coef_rows(self):
        """Stack coefficients as the R package's kappa params (p, V)."""
        rows = [self.b_topic]
        if self.use_aspect:
            rows.append(self.b_aspect)
        if self.use_inter:
            rows.append(self.b_inter.reshape(self.n, self.V))
        return np.vstack(rows)

    def state(self):
        return (self.b_topic.copy(),
                None if self.b_aspect is None else self.b_aspect.copy(),
                None if self.b_inter is None else self.b_inter.copy())

    def set_state(self, state):
        self.b_topic, self.b_aspect, self.b_inter = (
            state[0].copy(),
            None if state[1] is None else state[1].copy(),
            None if state[2] is None else state[2].copy(),
        )


def mnreg(beta_ss, wcounts, *, interactions=True, nlambda=250,
          lambda_min_ratio=0.001, ic_k=2.0, tol=1e-4,
          max_irls=4, max_sweeps=8):
    """Update beta and kappa from the E-step expected counts (mnreg in R).

    Only the (default) fixed-intercept variant is implemented: the
    intercept of each word's Poisson regression is fixed to the background
    log-probability ``m``.

    Returns ``(beta, kappa)`` where ``beta`` is a list of A matrices of
    shape (K, V) and ``kappa`` is a dict with the baseline ``m`` and the
    selected deviation coefficients ``params`` of shape (p, V).
    """
    A = len(beta_ss)
    K, V = beta_ss[0].shape
    use_aspect = A > 1
    use_inter = interactions and A > 1

    Y3 = np.stack(beta_ss)  # (A, K, V)
    m = np.log(wcounts) - np.log(wcounts.sum())
    row_totals = np.maximum(Y3.sum(axis=2), 1e-10)  # (A, K)
    offsets3 = m[None, None, :] + np.log(row_totals)[:, :, None]

    solver = _StructuredPoissonLasso(Y3, offsets3, use_aspect, use_inter)
    n = A * K

    mu0 = np.exp(np.clip(offsets3, -50.0, 30.0))
    nulldev = _poisson_deviance(Y3, mu0)
    # per-word lambda_max: max over columns of |score| at the null model
    R0 = Y3 - mu0
    scores = [np.abs(R0.sum(axis=0))]            # topic columns (K, V)
    if use_aspect:
        scores.append(np.abs(R0.sum(axis=1)))    # aspect columns (A, V)
    if use_inter:
        scores.append(np.abs(R0).reshape(n, V))  # interaction columns
    lambda_max = np.vstack(scores).max(axis=0) / n
    lambda_max = np.maximum(lambda_max, 1e-10)

    rel_path = np.exp(np.linspace(0.0, np.log(lambda_min_ratio), nlambda))
    best_ic = nulldev.copy()  # path point 0: all coefficients zero
    best_state = solver.state()
    any_improved = False

    for step in rel_path[1:]:
        lam = lambda_max * step
        solver.fit_one_lambda(lam, tol=tol, max_irls=max_irls,
                              max_sweeps=max_sweeps)
        mu = np.exp(solver.linpred())
        dev = 2.0 * (xlogy(Y3, Y3) - xlogy(Y3, mu) - (Y3 - mu)).sum(axis=(0, 1))
        ic = dev + ic_k * solver.df()
        improved = ic < best_ic
        if improved.any():
            any_improved = True
            cur = solver.state()
            best_state[0][:, improved] = cur[0][:, improved]
            if cur[1] is not None:
                best_state[1][:, improved] = cur[1][:, improved]
            if cur[2] is not None:
                best_state[2][:, :, improved] = cur[2][:, :, improved]
            best_ic[improved] = ic[improved]

    final = _StructuredPoissonLasso(Y3, offsets3, use_aspect, use_inter)
    if any_improved:
        final.set_state(best_state)

    linpred = final.linpred().reshape(n, V) - (
        np.log(row_totals).reshape(n)[:, None]
    )
    linpred -= linpred.max(axis=1, keepdims=True)
    explinpred = np.exp(linpred)
    beta_full = explinpred / explinpred.sum(axis=1, keepdims=True)

    beta = [beta_full[a * K:(a + 1) * K] for a in range(A)]
    kappa = {"m": m, "params": final.coef_rows()}
    return beta, kappa
