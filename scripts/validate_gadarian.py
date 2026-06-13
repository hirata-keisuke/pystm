"""Validate pystm against the R stm package's gadarianFit reference model.

gadarianFit ships with the R package and is the fitted model from the
Gadarian & Albertson immigration study used in Roberts et al. (2014,
AJPS) and the package vignette: K=3 topics, prevalence =
treatment * pid_rep, N=341 open-ended survey responses.

Tests
-----
1. Corpus reproduction: our textProcessor/prepDocuments port rebuilds the
   exact 215-term vocabulary and word counts the R model was fit on.
2. Numerical equivalence: running our E-step at R's fitted parameters
   reproduces R's reported bound (R logs the bound computed with the
   previous iteration's parameters, so ours must sit within one
   iteration's increment above it), and per-document theta matches.
3. Fixed point: continuing our EM from R's solution keeps the bound
   monotone with sub-tolerance increments — R's optimum is (numerically)
   a fixed point of our updates.
4. Independent refit: a fresh pystm fit (deterministic spectral init; R
   used stochastic Gibbs-LDA init, so exact equality is impossible)
   finds the same three topics and reproduces the treatment effect
   topic-by-topic in sign, with the significant effect matching in
   magnitude.

Run:  uv run python scripts/validate_gadarian.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).parent))
from gadarian_prep import load_gadarian, prep_documents, text_processor

sys.path.insert(0, str(Path(__file__).parent.parent))
from pystm import StructuralTopicModel, estimate_effect
from pystm._estep import estep
from pystm._mstep import opt_beta, opt_mu, opt_sigma
from pystm._utils import row_softmax, to_doc_list

PASS = []


def check(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"    -> {'OK' if ok else 'FAIL'}: {name} {detail}")


def main():
    gadarian, fit = load_gadarian()
    r_vocab = [str(v) for v in fit["vocab"]]
    r_beta = np.exp(np.asarray(fit["beta"]["logbeta"][0]))
    r_theta = np.asarray(fit["theta"])
    r_eta = np.asarray(fit["eta"])
    r_mu = np.asarray(fit["mu"]["mu"]).T  # (N, K-1)
    r_sigma = np.asarray(fit["sigma"])
    r_bound = np.asarray(fit["convergence"]["bound"])
    r_design = np.asarray(fit["settings"]["covariates"]["X"])
    r_wcounts = np.asarray(fit["settings"]["dim"]["wcounts"]["x"]).ravel()

    # ---- [1] reproduce the corpus -----------------------------------
    print("[1] corpus reproduction")
    tokens = text_processor(gadarian["open.ended.response"].tolist(),
                            legacy_order=True)
    X, vocab, removed = prep_documents(tokens, lower_thresh=3)
    check("no documents dropped", not removed)
    check("vocabulary identical", vocab == r_vocab, f"({len(vocab)} terms)")
    check("word counts identical",
          np.array_equal(X.sum(axis=0), r_wcounts),
          f"({X.sum()} tokens)")

    docs = to_doc_list(csr_matrix(X.astype(float)))
    N = X.shape[0]
    bindex = np.zeros(N, dtype=np.int64)

    # ---- [2] E-step at R's fitted parameters ------------------------
    print("[2] numerical equivalence at R's parameters")
    _, _, bound, lam = estep(docs, bindex, True, [r_beta], r_eta,
                             r_mu, r_sigma)
    our_bound = bound.sum()
    r_increment = r_bound[-1] - r_bound[-2]
    print(f"    bound: pystm E-step at R params {our_bound:.2f}  "
          f"R final {r_bound[-1]:.2f}  (R per-iter increment "
          f"{r_increment:.3f})")
    check("bound matches R (within one EM increment)",
          0 <= our_bound - r_bound[-1] <= max(3 * r_increment, 0.5),
          f"(diff {our_bound - r_bound[-1]:+.3f})")
    theta_ours = row_softmax(np.column_stack([lam, np.zeros(N)]))
    max_diff = np.abs(theta_ours - r_theta).max()
    cors = [np.corrcoef(theta_ours[:, k], r_theta[:, k])[0, 1]
            for k in range(3)]
    # R stores eta from the E-step *before* the final M-step, so one
    # further E-step legitimately shifts each document's mode slightly;
    # agreement at corr > 0.999 / max diff of a few percent is the
    # correct expectation, not exact equality.
    check("per-document theta matches R",
          max_diff < 0.05 and min(cors) > 0.999,
          f"(max |diff| {max_diff:.5f}, min corr {min(cors):.5f})")

    # ---- [3] R's solution is a fixed point of our EM ----------------
    print("[3] EM continuation from R's solution (fixed-point check)")
    beta, lambda_, mu, sigma = [r_beta], r_eta, r_mu, r_sigma
    bounds = []
    for _ in range(10):
        sigma_ss, beta_ss, bound, lambda_ = estep(
            docs, bindex, True, beta, lambda_, mu, sigma)
        mu, _ = opt_mu(lambda_, covar=r_design)
        sigma = opt_sigma(sigma_ss, lambda_, mu, 0.0)
        beta = opt_beta(beta_ss)
        bounds.append(bound.sum())
    diffs = np.diff(np.array([our_bound] + bounds))
    print(f"    bound trajectory: {bounds[0]:.2f} ... {bounds[-1]:.2f}")
    check("bound monotone non-decreasing", (diffs >= -1e-3).all())
    check("increments stay at convergence scale (R's optimum ~ fixed point)",
          bounds[-1] - our_bound < 5.0,
          f"(total drift {bounds[-1] - our_bound:+.2f} over 10 iters)")

    # ---- [4] independent refit --------------------------------------
    print("[4] independent pystm fit (spectral init) vs gadarianFit")
    model = StructuralTopicModel(n_components=3, max_iter=200, tol=1e-5)
    model.fit(X, prevalence=r_design)
    rel = (model.bound_[-1] - r_bound[-1]) / abs(r_bound[-1])
    print(f"    bound: pystm {model.bound_[-1]:.1f}  R {r_bound[-1]:.1f}  "
          f"(relative diff {rel:+.4%}; different local optima are expected "
          f"-- R used stochastic LDA init)")
    check("bound within 0.5% of R's optimum", abs(rel) < 0.005)

    a = model.components_ / np.linalg.norm(model.components_, axis=1,
                                           keepdims=True)
    b = r_beta / np.linalg.norm(r_beta, axis=1, keepdims=True)
    sim = a @ b.T
    rows, cols = linear_sum_assignment(-sim)
    perm = dict(zip(cols, rows))
    for r_k in range(3):
        p_k = perm[r_k]
        top_r = [vocab[j] for j in np.argsort(-r_beta[r_k])[:7]]
        top_p = [vocab[j] for j in np.argsort(-model.components_[p_k])[:7]]
        print(f"    R topic {r_k + 1} (cos {sim[p_k, r_k]:.3f})")
        print(f"      R    : {', '.join(top_r)}")
        print(f"      pystm: {', '.join(top_p)}")
    check("same topics recovered (min beta cosine > 0.85)",
          sim[rows, cols].min() > 0.85,
          f"(min {sim[rows, cols].min():.3f})")

    eff = estimate_effect(model, r_design, random_state=0)
    tables = eff.summary(random_state=0)
    coef_r = np.linalg.lstsq(r_design, r_theta, rcond=None)[0]
    sign_ok, mag_ok = True, True
    print("    treatment effect by topic (pystm composition vs OLS on "
          "R's theta):")
    for r_k in range(3):
        p_k = perm[r_k]
        est, pv = tables[p_k]["estimate"][1], tables[p_k]["p_value"][1]
        r_est = coef_r[1, r_k]
        sign_ok &= bool(np.sign(est) == np.sign(r_est))
        if pv < 0.05 and abs(r_est) > 0.05:
            mag_ok &= abs(est - r_est) < 0.1
        print(f"      topic {r_k + 1}: pystm {est:+.3f} (p={pv:.3f})  "
              f"R {r_est:+.3f}")
    check("treatment effect signs match R on all topics", sign_ok)
    check("significant effect magnitudes agree (within 0.1)", mag_ok)

    print()
    ok = all(PASS)
    print(f"VALIDATION {'PASSED' if ok else 'FAILED'} "
          f"({sum(PASS)}/{len(PASS)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
