"""Tests for stm.StructuralTopicModel.

Validation strategy (no R available for direct comparison):
- ELBO increases over EM iterations and converges,
- the topic-word matrix recovers planted topics on synthetic data,
- prevalence covariate effects are recovered with the right sign,
- the sklearn API contract (params, transform, errors) holds.
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from stm import StructuralTopicModel


def make_corpus(n_docs=200, n_topics=3, vocab_size=60, doc_len=80,
                prevalence_effect=2.0, eta_sd=0.5, seed=0):
    """Generate documents from the STM generative process.

    A binary covariate shifts the prior mean of topic 0 (eta dimension 0)
    by ``prevalence_effect``.  Larger ``eta_sd`` makes per-document topic
    proportions more concentrated (more realistic corpora).
    """
    rng = np.random.default_rng(seed)
    K, V = n_topics, vocab_size

    # planted topics: disjoint blocks of high-probability words + noise
    beta = np.full((K, V), 0.1 / V)
    block = V // K
    for k in range(K):
        beta[k, k * block:(k + 1) * block] += 1.0
    beta /= beta.sum(axis=1, keepdims=True)

    x = rng.integers(0, 2, size=n_docs)
    gamma = np.zeros((2, K - 1))
    gamma[1, 0] = prevalence_effect  # covariate boosts topic 0

    design = np.column_stack([np.ones(n_docs), x])
    eta = design @ gamma + rng.normal(0, eta_sd, size=(n_docs, K - 1))
    full_eta = np.column_stack([eta, np.zeros(n_docs)])
    theta = np.exp(full_eta)
    theta /= theta.sum(axis=1, keepdims=True)

    X = np.zeros((n_docs, V), dtype=np.int64)
    for i in range(n_docs):
        z = rng.choice(K, size=doc_len, p=theta[i])
        for k in range(K):
            n_k = (z == k).sum()
            if n_k:
                X[i] += rng.multinomial(n_k, beta[k])
    return X, x[:, None].astype(float), beta, theta


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def fitted(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=50, tol=1e-5)
    model.fit(X, prevalence=covar)
    return model


def match_topics(estimated, truth):
    """Cosine-similarity-optimal matching between topic sets."""
    est = estimated / np.linalg.norm(estimated, axis=1, keepdims=True)
    tru = truth / np.linalg.norm(truth, axis=1, keepdims=True)
    sim = est @ tru.T
    rows, cols = linear_sum_assignment(-sim)
    return cols, sim[rows, cols]


def test_bound_increases(fitted):
    bound = np.array(fitted.bound_)
    assert bound[-1] > bound[0]
    # after the first few iterations the bound should be near-monotone
    diffs = np.diff(bound[2:])
    assert (diffs >= -1e-6 * np.abs(bound[2:-1])).all()


def test_converged(fitted):
    assert fitted.converged_
    assert fitted.n_iter_ < 50


def test_topic_recovery(corpus, fitted):
    _, _, beta_true, _ = corpus
    _, sims = match_topics(fitted.components_, beta_true)
    assert sims.min() > 0.9


def test_components_are_distributions(fitted):
    assert fitted.components_.shape == (3, 60)
    np.testing.assert_allclose(fitted.components_.sum(axis=1), 1.0, rtol=1e-8)
    assert (fitted.components_ >= 0).all()


def test_theta_shape_and_simplex(corpus, fitted):
    X, _, _, _ = corpus
    assert fitted.theta_.shape == (X.shape[0], 3)
    np.testing.assert_allclose(fitted.theta_.sum(axis=1), 1.0, rtol=1e-8)


def test_prevalence_effect_recovered(corpus, fitted):
    """Documents with x=1 should put more mass on the boosted topic."""
    _, covar, beta_true, _ = corpus
    perm, _ = match_topics(fitted.components_, beta_true)
    boosted = int(np.where(perm == 0)[0][0])  # estimated topic matching true topic 0
    x = covar.ravel().astype(bool)
    diff = fitted.theta_[x, boosted].mean() - fitted.theta_[~x, boosted].mean()
    assert diff > 0.15


def test_gamma_shape(fitted):
    assert fitted.gamma_.shape == (2, 2)  # intercept + covariate, K-1 columns


def test_transform_matches_fit(corpus, fitted):
    X, covar, _, _ = corpus
    theta_new = fitted.transform(X, prevalence=covar)
    # same data, same covariates: should land very close to the fit thetas
    assert np.abs(theta_new - fitted.theta_).max() < 0.05


def test_transform_requires_covariates(corpus, fitted):
    X, _, _, _ = corpus
    with pytest.raises(ValueError, match="prevalence"):
        fitted.transform(X)


def test_ctm_mode(corpus):
    """No covariates: the model reduces to a CTM and still recovers topics."""
    X, _, beta_true, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=30)
    theta = model.fit_transform(X)
    assert model.gamma_ is None
    assert model.mu_.shape == (2,)
    assert theta.shape == (X.shape[0], 3)
    _, sims = match_topics(model.components_, beta_true)
    assert sims.min() > 0.9


def test_random_init(corpus):
    """Random init is prone to local optima (as documented for the R
    package), so only check that estimation runs and the bound improves."""
    X, _, _, _ = corpus
    model = StructuralTopicModel(
        n_components=3, init="random", max_iter=40, random_state=1
    )
    model.fit(X)
    assert model.bound_[-1] > model.bound_[0]
    np.testing.assert_allclose(model.components_.sum(axis=1), 1.0, rtol=1e-8)


def test_score(corpus, fitted):
    X, covar, _, _ = corpus
    s = fitted.score(X, prevalence=covar)
    assert np.isfinite(s)
    assert s == pytest.approx(fitted.bound_[-1], rel=0.05)


def test_top_words(fitted):
    top = fitted.top_words(n_words=5)
    assert top.shape == (3, 5)
    frex = fitted.top_words(n_words=5, kind="frex")
    assert frex.shape == (3, 5)
    # the planted block structure means top words should lie in one block
    for k in range(3):
        blocks = top[k] // 20
        assert len(set(blocks)) == 1


def test_gamma_coefficient_recovery():
    """On a well-separated corpus the prevalence coefficient itself
    should be recovered close to its true value."""
    X, covar, beta_true, _ = make_corpus(
        n_docs=400, n_topics=5, vocab_size=150, doc_len=80,
        prevalence_effect=1.0, eta_sd=2.0, seed=7,
    )
    model = StructuralTopicModel(n_components=5, max_iter=60)
    model.fit(X, prevalence=covar)
    perm, sims = match_topics(model.components_, beta_true)
    assert sims.min() > 0.9
    boosted = int(np.where(perm == 0)[0][0])
    # effect on eta in the full K-dim space (reference topic has eta = 0);
    # true effect: +1.0 on the boosted topic relative to every other topic
    delta = np.append(model.gamma_[1], 0.0)
    others = np.delete(np.arange(5), boosted)
    effect = delta[boosted] - delta[others].mean()
    assert effect == pytest.approx(1.0, abs=0.4)


def test_sparse_input(corpus):
    from scipy.sparse import csr_matrix

    X, covar, _, _ = corpus
    model = StructuralTopicModel(n_components=3, max_iter=5)
    model.fit(csr_matrix(X), prevalence=covar)
    assert model.components_.shape == (3, X.shape[1])


def test_sklearn_clone_and_params():
    from sklearn.base import clone

    model = StructuralTopicModel(n_components=4, tol=1e-4)
    cloned = clone(model)
    assert cloned.get_params()["n_components"] == 4
    cloned.set_params(n_components=5)
    assert cloned.n_components == 5


def test_input_validation():
    model = StructuralTopicModel(n_components=1)
    with pytest.raises(ValueError, match="n_components"):
        model.fit(np.ones((5, 10)))

    model = StructuralTopicModel(n_components=3)
    X = np.ones((5, 10))
    X[2] = 0  # empty document
    with pytest.raises(ValueError, match="empty"):
        model.fit(X)

    with pytest.raises(ValueError, match="rows"):
        StructuralTopicModel(n_components=3).fit(
            np.ones((5, 10)), prevalence=np.ones((4, 1))
        )


# ---------------------------------------------------------------------------
# K=0: automatic topic count (Lee & Mimno 2014)


@pytest.fixture(scope="module")
def fitted_k0(corpus):
    X, covar, _, _ = corpus
    model = StructuralTopicModel(
        n_components=0, init="spectral", max_iter=30, random_state=0
    )
    model.fit(X, prevalence=covar)
    return model


def test_k0_chooses_topic_count(fitted_k0):
    K = fitted_k0.n_components_
    # K is data-driven and must be a consistent, usable value
    assert isinstance(K, (int, np.integer))
    assert K >= 2
    assert fitted_k0.components_.shape[0] == K
    assert fitted_k0.theta_.shape[1] == K
    assert fitted_k0.sigma_.shape == (K - 1, K - 1)


def test_k0_components_are_distributions(fitted_k0):
    beta = fitted_k0.components_
    assert np.all(beta >= 0)
    assert np.allclose(beta.sum(axis=1), 1.0)


def test_k0_transform_and_score(corpus, fitted_k0):
    X, covar, _, _ = corpus
    theta = fitted_k0.transform(X[:5], prevalence=covar[:5])
    assert theta.shape == (5, fitted_k0.n_components_)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.isfinite(fitted_k0.score(X[:5], prevalence=covar[:5]))


def test_k0_requires_spectral():
    with pytest.raises(ValueError, match="spectral"):
        StructuralTopicModel(n_components=0, init="random").fit(
            np.ones((5, 10))
        )
