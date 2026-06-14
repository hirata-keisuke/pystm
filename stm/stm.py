"""Structural Topic Model with a scikit-learn style API.

Port of the R ``stm`` package's core estimation routine (variational EM
for the logistic-normal topic model with prevalence covariates).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._estep import decompose_sigma, estep, optimize_document
from ._mnreg import mnreg
from ._mstep import opt_beta, opt_mu, opt_sigma
from ._spectral import spectral_init
from ._utils import row_softmax, safelog, to_doc_list


class StructuralTopicModel(BaseEstimator, TransformerMixin):
    """Structural Topic Model (STM) with variational EM.

    STM is a logistic-normal topic model in which document metadata
    ("prevalence covariates") shifts the prior mean of each document's
    topic proportions.  Without covariates the model reduces to the
    Correlated Topic Model.  The API follows
    :class:`sklearn.decomposition.LatentDirichletAllocation`; covariates
    are passed to :meth:`fit` / :meth:`transform` via the ``prevalence``
    keyword.

    Parameters
    ----------
    n_components : int, default=10
        Number of topics K (must be >= 2).  Set to ``0`` to choose the
        number of topics automatically from the data with the Lee & Mimno
        (2014) algorithm (requires ``init="spectral"``); the value used is
        then exposed as :attr:`n_components_`.
    init : {"spectral", "random"}, default="spectral"
        Initialization method.  "spectral" is the deterministic anchor-word
        algorithm of Arora et al. (2013), recommended by the stm authors.
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-5
        EM stops when the relative change of the approximate evidence
        lower bound falls below ``tol``.
    sigma_prior : float, default=0.0
        Weight in [0, 1] of regularization of the topic covariance matrix
        towards its diagonal.
    gamma_max_iter : int, default=1000
        Iteration limit for the variational prevalence regression.
    e_step_max_iter : int, default=500
        BFGS iteration limit for each document's optimization.
    max_vocab : int or None, default=10000
        With spectral initialization, only this many most frequent terms
        are used to build the gram matrix (None disables the cap).
    content_interactions : bool, default=True
        For content covariate models, whether to include topic-by-level
        interaction deviations in addition to the main effects.
    warm_start : bool, default=False
        When True, repeated calls to :meth:`fit` continue EM from the
        previous solution instead of re-initializing (the analogue of the
        R package's ``model=`` restart argument).  Each call runs up to
        ``max_iter`` further iterations and ``bound_`` keeps the full
        history.  ``n_components``, the vocabulary and the covariate
        setup must stay the same; per-document state is reused only when
        the corpus has the same number of documents.
    random_state : int, RandomState or None, default=None
        Used for ``init="random"`` and for the t-SNE projection of the
        automatic topic count (``n_components=0``).  The ordinary spectral
        method with a fixed K is deterministic and ignores it.
    verbose : int, default=0
        If positive, print the bound after each EM iteration.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Topic-word distributions (each row sums to 1).  Note this differs
        from sklearn's LDA, whose ``components_`` holds unnormalized
        pseudo-counts.  For content covariate models this is the
        aspect-frequency-weighted average of ``aspect_components_``.
    aspect_components_ : ndarray of shape (n_levels, n_components, n_features) or None
        Per-content-level topic-word distributions (content models only).
    kappa_ : dict or None
        SAGE parameters of the content model: baseline log-probabilities
        ``m`` (n_features,) and sparse deviations ``params``.
    content_levels_ : ndarray or None
        Sorted unique content covariate levels seen at fit time.
    gamma_ : ndarray of shape (1 + n_covariates, n_components - 1) or None
        Prevalence regression coefficients (first row is the intercept).
        None when no covariates were used.
    mu_ : ndarray
        Prior means of eta: shape (n_components - 1,) without covariates,
        (n_samples, n_components - 1) with covariates.
    sigma_ : ndarray of shape (n_components - 1, n_components - 1)
        Topic covariance matrix.
    theta_ : ndarray of shape (n_samples, n_components)
        Topic proportions of the training documents.
    eta_ : ndarray of shape (n_samples, n_components - 1)
        Variational means of the logistic-normal parameters.
    bound_ : list of float
        Approximate evidence lower bound at each EM iteration.
    n_iter_ : int
        Number of EM iterations run.
    n_components_ : int
        Number of topics actually used.  Equal to ``n_components`` unless
        ``n_components=0`` was requested, in which case it is the topic
        count selected by the Lee & Mimno (2014) algorithm.
    converged_ : bool
        Whether the bound converged before ``max_iter``.

    References
    ----------
    Roberts, M., Stewart, B., & Tingley, D. (2019). "stm: An R Package for
    Structural Topic Models." Journal of Statistical Software 91(2).
    """

    def __init__(self, n_components=10, *, init="spectral", max_iter=500,
                 tol=1e-5, sigma_prior=0.0, gamma_max_iter=1000,
                 e_step_max_iter=500, max_vocab=10000,
                 content_interactions=True, warm_start=False,
                 random_state=None, verbose=0):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.sigma_prior = sigma_prior
        self.gamma_max_iter = gamma_max_iter
        self.e_step_max_iter = e_step_max_iter
        self.max_vocab = max_vocab
        self.content_interactions = content_interactions
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # validation helpers

    def _validate_inputs(self, X, prevalence, *, reset):
        X = csr_matrix(X) if not issparse(X) else X.tocsr()
        X = X.astype(np.float64)
        if not reset and X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted "
                f"with {self.components_.shape[1]}."
            )
        doc_lens = np.asarray(X.sum(axis=1)).ravel()
        if np.any(doc_lens == 0):
            raise ValueError(
                "X contains empty documents; remove them before fitting "
                "(cf. prepDocuments in the R package)."
            )
        design = None
        if prevalence is not None:
            design = np.asarray(prevalence, dtype=np.float64)
            if design.ndim == 1:
                design = design[:, None]
            if design.shape[0] != X.shape[0]:
                raise ValueError(
                    "prevalence has a different number of rows than X."
                )
            if np.isnan(design).any():
                raise ValueError("Missing values in prevalence covariates.")
            # prepend an intercept unless one is already there
            if not np.allclose(design[:, 0], 1.0):
                design = np.column_stack([np.ones(design.shape[0]), design])
        return X, design

    def _encode_content(self, content, n_docs, *, reset):
        """Map content covariate labels to aspect indices (betaindex)."""
        if content is None:
            if not reset and getattr(self, "content_levels_", None) is not None:
                raise ValueError(
                    "The model was fitted with a content covariate; pass "
                    "the matching content labels."
                )
            return None, np.zeros(n_docs, dtype=np.int64)
        content = np.asarray(content).ravel()
        if content.shape[0] != n_docs:
            raise ValueError("content has a different number of rows than X.")
        if reset:
            levels = np.unique(content)
        else:
            if getattr(self, "content_levels_", None) is None:
                raise ValueError(
                    "The model was fitted without a content covariate."
                )
            levels = self.content_levels_
            unseen = ~np.isin(content, levels)
            if unseen.any():
                raise ValueError(
                    f"Unseen content levels: {np.unique(content[unseen])!r}"
                )
        index = np.searchsorted(levels, content)
        return levels, index

    # ------------------------------------------------------------------
    # fitting

    def fit(self, X, y=None, *, prevalence=None, content=None):
        """Fit the model to a document-term count matrix.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Word counts per document.
        y : ignored
        prevalence : array-like of shape (n_samples, n_covariates), optional
            Prevalence covariate design matrix.  An intercept column is
            added automatically.  Categorical variables must be encoded
            numerically (e.g. one-hot) beforehand.
        content : array-like of shape (n_samples,), optional
            Content covariate: one categorical label per document.  Each
            level gets its own topic-word distributions, estimated as
            sparse deviations from a shared baseline (SAGE-style, via
            distributed Poisson regression as in the R package's L1 mode).
        """
        K = self.n_components
        if not (isinstance(K, (int, np.integer)) and (K >= 2 or K == 0)):
            raise ValueError(
                "n_components must be an integer >= 2, or 0 to select the "
                "number of topics automatically (Lee & Mimno 2014)."
            )
        if not 0.0 <= self.sigma_prior <= 1.0:
            raise ValueError("sigma_prior must be between 0 and 1.")
        if self.init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral' or 'random'.")
        if K == 0 and self.init != "spectral":
            raise ValueError(
                "n_components=0 (automatic topic count) requires "
                "init='spectral'."
            )

        warm = self.warm_start and hasattr(self, "components_")
        X, design = self._validate_inputs(X, prevalence, reset=not warm)
        docs = to_doc_list(X)
        N, V = X.shape

        if warm:
            # ---- continue from the previous solution (cf. the R
            #      package's model= restart argument) ----
            if K == 0:
                # the topic count was already fixed by the first fit
                K = self.components_.shape[0]
            if self.components_.shape[0] != K:
                raise ValueError(
                    f"warm_start requires the same n_components as the "
                    f"previous fit ({self.components_.shape[0]}), "
                    f"got {K}."
                )
            levels, beta_index = self._encode_content(content, N,
                                                      reset=False)
            A = 1 if levels is None else len(levels)
            beta = [b.copy() for b in self._beta_list()]
            sigma = self.sigma_.copy()
            if (design is not None and self.gamma_ is not None
                    and design.shape[1] == self.gamma_.shape[0]):
                gamma = self.gamma_.copy()
                mu = design @ gamma
            else:
                gamma = None
                mu = (self.mu_.copy() if self.mu_.ndim == 1
                      else np.zeros(K - 1))
            # per-document state is only reusable for the same corpus
            lambda_ = (self.eta_.copy() if self.eta_.shape[0] == N
                       else np.zeros((N, K - 1)))
            kappa = self.kappa_
            bound_history = list(self.bound_)
        else:
            levels, beta_index = self._encode_content(content, N,
                                                      reset=True)
            A = 1 if levels is None else len(levels)

            # ---- initialization (stm.init) ----
            if self.init == "spectral":
                init_beta, K = spectral_init(
                    X, K, max_vocab=self.max_vocab,
                    random_state=self.random_state,
                )
            else:
                rng = check_random_state(self.random_state)
                b = rng.gamma(0.1, 1.0, size=(K, V))
                init_beta = b / b.sum(axis=1, keepdims=True)
            beta = [init_beta.copy() for _ in range(A)]
            mu = np.zeros(K - 1)
            sigma = np.diag(np.full(K - 1, 20.0))
            lambda_ = np.zeros((N, K - 1))
            gamma = None
            kappa = None
            bound_history = []

        wcounts = np.asarray(X.sum(axis=0)).ravel()

        # ---- EM loop (stm.control) ----
        converged = False
        for _ in range(self.max_iter):
            # like the R code, document-specific means are only available
            # once gamma has been estimated (i.e. from the second iteration)
            update_mu = gamma is not None
            sigma_ss, beta_ss, bound, lambda_ = estep(
                docs, beta_index, update_mu, beta, lambda_, mu, sigma,
                max_optim_iter=self.e_step_max_iter,
            )
            mu, gamma = opt_mu(lambda_, covar=design,
                               max_iter=self.gamma_max_iter)
            sigma = opt_sigma(sigma_ss, lambda_, mu, self.sigma_prior)
            if levels is None:
                beta = opt_beta(beta_ss)
            else:
                beta, kappa = mnreg(
                    beta_ss, wcounts,
                    interactions=self.content_interactions,
                )

            bound_history.append(float(bound.sum()))
            if self.verbose:
                print(f"Iteration {len(bound_history)}: "
                      f"bound = {bound_history[-1]:.2f}")
            if len(bound_history) > 1:
                old, new = bound_history[-2], bound_history[-1]
                if (new - old) / abs(old) < self.tol:
                    converged = True
                    if self.verbose:
                        print("Model converged.")
                    break

        # ---- pack the results ----
        self.content_levels_ = levels
        if levels is None:
            self.components_ = beta[0]
            self.aspect_components_ = None
            self.kappa_ = None
        else:
            self.aspect_components_ = np.stack(beta)
            # corpus-level summary: aspect betas weighted by frequency
            weights = np.bincount(beta_index, minlength=A) / N
            self.components_ = np.tensordot(
                weights, self.aspect_components_, axes=1
            )
            self.kappa_ = kappa
        self.gamma_ = gamma
        self.mu_ = mu
        self.sigma_ = sigma
        self.eta_ = lambda_
        full_eta = np.column_stack([lambda_, np.zeros(N)])
        self.theta_ = row_softmax(full_eta)
        self.bound_ = bound_history
        self.n_iter_ = len(bound_history)
        self.converged_ = converged
        self.n_features_in_ = V
        # K may have been chosen automatically (n_components=0); expose the
        # value actually used following the sklearn fitted-attribute convention
        self.n_components_ = K
        return self

    def fit_transform(self, X, y=None, *, prevalence=None, content=None):
        """Fit the model and return the training documents' theta."""
        return self.fit(X, prevalence=prevalence, content=content).theta_

    # ------------------------------------------------------------------
    # inference on new documents

    def _new_doc_priors(self, design):
        """Per-document prior means for held-out inference."""
        if self.gamma_ is not None:
            if design is None:
                raise ValueError(
                    "The model was fitted with prevalence covariates; pass "
                    "the matching covariates."
                )
            if design.shape[1] != self.gamma_.shape[0]:
                raise ValueError(
                    "prevalence has a different number of columns than "
                    "at fit time."
                )
            return design @ self.gamma_, True
        mu = self.mu_ if self.mu_.ndim == 1 else self.mu_.mean(axis=0)
        return mu, False

    def _beta_list(self):
        """Topic-word distributions as a per-aspect list."""
        if self.aspect_components_ is None:
            return [self.components_]
        return list(self.aspect_components_)

    def transform(self, X, *, prevalence=None, content=None):
        """Infer topic proportions for (possibly new) documents.

        Runs one E-step with the fitted global parameters held fixed
        (cf. fitNewDocuments in the R package) and returns theta of shape
        (n_samples, n_components).
        """
        check_is_fitted(self, "components_")
        X, design = self._validate_inputs(X, prevalence, reset=False)
        docs = to_doc_list(X)
        N = X.shape[0]
        K = self.n_components_
        _, beta_index = self._encode_content(content, N, reset=False)
        beta = self._beta_list()
        mu, update_mu = self._new_doc_priors(design)

        siginv, sigmaentropy = decompose_sigma(self.sigma_)
        eta = np.zeros((N, K - 1))
        for i, (words, counts) in enumerate(docs):
            beta_d = np.ascontiguousarray(beta[beta_index[i]][:, words])
            mu_d = mu[i] if update_mu else mu
            _, eta[i] = optimize_document(
                eta[i], beta_d, counts, mu_d, siginv, sigmaentropy,
                max_optim_iter=self.e_step_max_iter,
            )
        return row_softmax(np.column_stack([eta, np.zeros(N)]))

    def score(self, X, y=None, *, prevalence=None, content=None):
        """Approximate evidence lower bound of ``X`` under the fitted model."""
        check_is_fitted(self, "components_")
        X, design = self._validate_inputs(X, prevalence, reset=False)
        docs = to_doc_list(X)
        N = X.shape[0]
        K = self.n_components_
        _, beta_index = self._encode_content(content, N, reset=False)
        mu, update_mu = self._new_doc_priors(design)
        _, _, bound, _ = estep(
            docs, beta_index, update_mu,
            self._beta_list(), np.zeros((N, K - 1)), mu, self.sigma_,
            max_optim_iter=self.e_step_max_iter,
        )
        return float(bound.sum())

    def perplexity(self, X, *, prevalence=None, content=None):
        """Per-token perplexity of ``X``, ``exp(-bound / n_tokens)``.

        Like :meth:`sklearn.decomposition.LatentDirichletAllocation.perplexity`
        this is based on the variational bound (here the logistic-normal
        ELBO from :meth:`score`), so values are comparable between fits
        of this class on the same data; lower is better.
        """
        check_is_fitted(self, "components_")
        bound = self.score(X, prevalence=prevalence, content=content)
        X_csr, _ = self._validate_inputs(X, None, reset=False)
        n_tokens = X_csr.sum()
        return float(np.exp(-bound / n_tokens))

    # ------------------------------------------------------------------
    # interpretation helpers

    def top_words(self, n_words=10, *, kind="prob", frex_weight=0.5):
        """Indices of the top words per topic (cf. labelTopics).

        ``kind="prob"`` ranks by within-topic probability; ``kind="frex"``
        balances frequency and exclusivity with weight ``frex_weight``.
        Returns an array of shape (n_components, n_words).
        """
        check_is_fitted(self, "components_")
        logbeta = safelog(self.components_)
        if kind == "prob":
            scores = logbeta
        elif kind == "frex":
            from scipy.special import logsumexp

            excl = logbeta - logsumexp(logbeta, axis=0, keepdims=True)
            freq_rank = np.apply_along_axis(rankdata, 1, logbeta) / logbeta.shape[1]
            excl_rank = np.apply_along_axis(rankdata, 1, excl) / logbeta.shape[1]
            scores = 1.0 / (frex_weight / freq_rank + (1 - frex_weight) / excl_rank)
        else:
            raise ValueError("kind must be 'prob' or 'frex'.")
        return np.argsort(-scores, axis=1)[:, :n_words]
