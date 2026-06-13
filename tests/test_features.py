"""Tests for content covariates, estimate_effect, search_k and diagnostics."""

import numpy as np
import pytest

from stm import (
    StructuralTopicModel,
    check_residuals,
    estimate_effect,
    eval_heldout,
    exclusivity,
    make_heldout,
    search_k,
    semantic_coherence,
    topic_corr,
)
from tests.test_stm import make_corpus, match_topics


@pytest.fixture(scope="module")
def corpus():
    return make_corpus(n_docs=200, n_topics=3, vocab_size=60, doc_len=80,
                       prevalence_effect=2.0, seed=0)


@pytest.fixture(scope="module")
def fitted(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=50)
    model.fit(X, prevalence=covar)
    return model


def make_content_corpus(n_docs=150, n_topics=3, vocab_size=60, doc_len=60,
                        seed=11):
    """Corpus where word use within each topic differs by aspect."""
    rng = np.random.default_rng(seed)
    K, V = n_topics, vocab_size
    block = V // K
    base = np.full((K, V), 0.1 / V)
    for k in range(K):
        base[k, k * block:(k + 1) * block] += 1.0
    beta_a = [base.copy(), base.copy()]
    for k in range(K):
        beta_a[0][k, k * block:k * block + block // 2] *= 3.0
        beta_a[1][k, k * block + block // 2:(k + 1) * block] *= 3.0
    beta_a = [b / b.sum(axis=1, keepdims=True) for b in beta_a]

    aspect = rng.integers(0, 2, size=n_docs)
    theta = rng.dirichlet(np.full(K, 0.3), size=n_docs)
    X = np.zeros((n_docs, V), dtype=np.int64)
    for i in range(n_docs):
        z = rng.choice(K, size=doc_len, p=theta[i])
        for k in range(K):
            n_k = (z == k).sum()
            if n_k:
                X[i] += rng.multinomial(n_k, beta_a[aspect[i]][k])
    return X, aspect, base, block


@pytest.fixture(scope="module")
def content_fitted():
    X, aspect, base, block = make_content_corpus()
    model = StructuralTopicModel(n_components=3, max_iter=25)
    model.fit(X, content=aspect)
    return model, X, aspect, base, block


# ---------------------------------------------------------------------------
# content covariates

def test_content_shapes(content_fitted):
    model, X, aspect, base, _ = content_fitted
    assert model.aspect_components_.shape == (2, 3, X.shape[1])
    assert model.content_levels_.tolist() == [0, 1]
    assert model.kappa_ is not None
    assert model.kappa_["m"].shape == (X.shape[1],)
    np.testing.assert_allclose(
        model.aspect_components_.sum(axis=2), 1.0, rtol=1e-8
    )


def test_content_recovers_topics_and_aspects(content_fitted):
    model, _, _, base, block = content_fitted
    perm, sims = match_topics(model.components_, base)
    assert sims.min() > 0.9
    # aspect 0 should put more mass on the front half of each topic block
    ac = model.aspect_components_
    for est_k, true_k in enumerate(perm):
        front = slice(true_k * block, true_k * block + block // 2)
        back = slice(true_k * block + block // 2, (true_k + 1) * block)
        assert ac[0, est_k, front].sum() > ac[0, est_k, back].sum()
        assert ac[1, est_k, back].sum() > ac[1, est_k, front].sum()


def test_content_kappa_sparsity(content_fitted):
    """The lasso should zero out a meaningful share of the deviations."""
    model = content_fitted[0]
    params = model.kappa_["params"]
    assert (params == 0).mean() > 0.1


def test_content_transform_and_score(content_fitted):
    model, X, aspect, _, _ = content_fitted
    theta = model.transform(X[:20], content=aspect[:20])
    assert theta.shape == (20, 3)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0, rtol=1e-8)
    assert np.isfinite(model.score(X[:20], content=aspect[:20]))


def test_content_required_in_transform(content_fitted):
    model, X, _, _, _ = content_fitted
    with pytest.raises(ValueError, match="content"):
        model.transform(X[:5])


def test_content_unseen_level(content_fitted):
    model, X, _, _, _ = content_fitted
    with pytest.raises(ValueError, match="Unseen"):
        model.transform(X[:5], content=np.array([0, 1, 2, 0, 1]))


def test_content_not_fitted_error(fitted, corpus):
    X = corpus[0]
    with pytest.raises(ValueError, match="without a content"):
        fitted.transform(X[:5], content=np.zeros(5))


# ---------------------------------------------------------------------------
# estimate_effect

def test_estimate_effect_recovers_prevalence(corpus, fitted):
    X, covar, beta_true, _ = corpus
    perm, _ = match_topics(fitted.components_, beta_true)
    boosted = int(np.where(perm == 0)[0][0])

    eff = estimate_effect(fitted, covar, random_state=0)
    tables = eff.summary(random_state=0)
    assert set(tables) == {0, 1, 2}
    table = tables[boosted]
    assert table.shape == (2,)  # intercept + covariate
    # x=1 boosts the topic: positive, significant slope on theta
    x = covar.ravel().astype(bool)
    true_diff = fitted.theta_[x, boosted].mean() - fitted.theta_[~x, boosted].mean()
    assert table["estimate"][1] == pytest.approx(true_diff, abs=0.05)
    assert table["p_value"][1] < 0.01


def test_estimate_effect_uncertainty_widens_se(corpus, fitted):
    _, covar, _, _ = corpus
    eff_g = estimate_effect(fitted, covar, nsims=25, random_state=0)
    eff_n = estimate_effect(fitted, covar, uncertainty="None")
    se_g = eff_g.summary(random_state=0)[0]["std_error"][1]
    se_n = eff_n.summary(random_state=0)[0]["std_error"][1]
    assert se_g >= se_n * 0.9  # Global adds measurement uncertainty


def test_estimate_effect_topics_subset(corpus, fitted):
    _, covar, _, _ = corpus
    eff = estimate_effect(fitted, covar, topics=[1], random_state=0)
    tables = eff.summary()
    assert list(tables) == [1]
    with pytest.raises(ValueError, match="not estimated"):
        eff.summary(topics=[0])


def test_estimate_effect_validation(corpus, fitted):
    _, covar, _, _ = corpus
    with pytest.raises(ValueError, match="uncertainty"):
        estimate_effect(fitted, covar, uncertainty="Local")
    with pytest.raises(ValueError, match="rows"):
        estimate_effect(fitted, covar[:-5])
    with pytest.raises(ValueError, match="topics"):
        estimate_effect(fitted, covar, topics=[99])


# ---------------------------------------------------------------------------
# diagnostics

def test_topic_corr(fitted):
    tc = topic_corr(fitted)
    K = 3
    assert tc.cor.shape == (K, K)
    np.testing.assert_allclose(np.diag(tc.cor), 1.0)
    assert (tc.posadj == tc.posadj.T).all()
    assert ((tc.poscor >= 0) | np.isclose(tc.poscor, 0)).all()


def test_semantic_coherence(corpus, fitted):
    X = corpus[0]
    scores = semantic_coherence(fitted, X, M=5)
    assert scores.shape == (3,)
    # planted block topics co-occur heavily: scores should beat a model
    # with shuffled (mismatched) topic-word distributions
    shuffled = fitted.components_[:, np.random.default_rng(0).permutation(60)]
    bad = StructuralTopicModel(n_components=3)
    bad.components_ = shuffled
    bad.aspect_components_ = None
    bad_scores = semantic_coherence(bad, X, M=5)
    assert scores.mean() > bad_scores.mean()


def test_exclusivity(fitted, content_fitted):
    scores = exclusivity(fitted, M=5)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="content"):
        exclusivity(content_fitted[0])


def test_check_residuals(corpus, fitted):
    X = corpus[0]
    out = check_residuals(fitted, X)
    assert np.isfinite(out["dispersion"])
    assert out["dispersion"] > 0


# ---------------------------------------------------------------------------
# warm_start / perplexity

def test_warm_start_continues_em(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=3,
                                 warm_start=True)
    model.fit(X, prevalence=covar)
    first_bounds = list(model.bound_)
    assert len(first_bounds) == 3 and not model.converged_

    model.fit(X, prevalence=covar)
    # history accumulates and EM resumes monotonically at the junction
    assert model.bound_[:3] == first_bounds
    assert len(model.bound_) > 3
    assert model.bound_[3] >= first_bounds[-1] - 1e-6 * abs(first_bounds[-1])
    diffs = np.diff(model.bound_)
    assert (diffs >= -1e-6 * np.abs(np.array(model.bound_)[:-1])).all()
    assert model.n_iter_ == len(model.bound_)


def test_warm_start_matches_cold_fit(corpus):
    """3+47 warm iterations should land where a single 50-iter fit does."""
    X, covar, beta_true, _ = corpus
    warm = StructuralTopicModel(n_components=3, max_iter=3, warm_start=True)
    warm.fit(X, prevalence=covar)
    warm.set_params(max_iter=47)
    warm.fit(X, prevalence=covar)
    cold = StructuralTopicModel(n_components=3, max_iter=50)
    cold.fit(X, prevalence=covar)
    assert warm.bound_[-1] == pytest.approx(cold.bound_[-1], rel=1e-4)
    _, sims = match_topics(warm.components_, cold.components_)
    assert sims.min() > 0.999


def test_warm_start_false_refits(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=3)
    model.fit(X, prevalence=covar)
    model.fit(X, prevalence=covar)
    assert len(model.bound_) <= 3  # history was reset


def test_warm_start_rejects_changed_k(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=2,
                                 warm_start=True)
    model.fit(X, prevalence=covar)
    model.set_params(n_components=4)
    with pytest.raises(ValueError, match="n_components"):
        model.fit(X, prevalence=covar)


def test_perplexity(corpus, fitted):
    X, covar, _, _ = corpus
    perp = fitted.perplexity(X, prevalence=covar)
    assert np.isfinite(perp) and perp > 0
    # definition: exp(-bound / n_tokens)
    score = fitted.score(X, prevalence=covar)
    assert perp == pytest.approx(np.exp(-score / X.sum()), rel=1e-6)
    # a barely-trained model should be more perplexed
    rough = StructuralTopicModel(n_components=3, init="random",
                                 max_iter=1, random_state=0)
    rough.fit(X, prevalence=covar)
    assert rough.perplexity(X, prevalence=covar) > perp


# ---------------------------------------------------------------------------
# heldout / search_k

def test_make_heldout_conserves_tokens(corpus):
    X = corpus[0]
    heldout = make_heldout(X, N=30, random_state=0)
    X_train = heldout["X_train"].toarray()
    total_missing = np.zeros(X.shape[1])
    for doc_id, (words, counts) in zip(heldout["index"], heldout["docs"]):
        row_missing = np.zeros(X.shape[1])
        row_missing[words] = counts
        np.testing.assert_array_equal(X_train[doc_id] + row_missing, X[doc_id])
        total_missing += row_missing
    assert total_missing.sum() > 0
    assert (X_train.sum(axis=1) > 0).all()  # no empty training docs


def test_eval_heldout(corpus):
    X, covar, _, _ = corpus
    heldout = make_heldout(X, N=30, random_state=0)
    model = StructuralTopicModel(n_components=3, max_iter=15)
    model.fit(heldout["X_train"], prevalence=covar)
    out = eval_heldout(model, heldout)
    assert np.isfinite(out["expected_heldout"])
    assert out["expected_heldout"] < 0
    assert out["doc_heldout"].shape == (len(heldout["docs"]),)


def test_search_k(corpus):
    X, covar, _, _ = corpus
    res = search_k(
        X, [3, 4], prevalence=covar, N=25, heldout_random_state=0,
        model_params={"max_iter": 8},
    )
    expected_keys = {"K", "heldout", "residual", "bound", "lbound",
                     "exclus", "semcoh", "em_its"}
    assert set(res) == expected_keys
    assert res["K"].tolist() == [3, 4]
    for key in expected_keys - {"K"}:
        assert res[key].shape == (2,)
        assert np.isfinite(res[key]).all()
    # the data has 3 planted topics; K=3 should win on heldout likelihood
    # (not asserted strictly since max_iter is small, just sanity-check sign)
    assert (res["heldout"] < 0).all()
