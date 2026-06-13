# pystm — Structural Topic Model in Python

[![PyPI](https://img.shields.io/pypi/v/structural-topic-model)](https://pypi.org/project/structural-topic-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[日本語 README](README_ja.md)

A Python port of the R [stm](https://github.com/bstewart/stm) package (Roberts, Stewart & Tingley) with a scikit-learn style API.

## What is STM?

The Structural Topic Model (STM) is a logistic-normal topic model where document metadata ("prevalence covariates") shifts the prior mean of each document's topic proportions. Without covariates it reduces to the Correlated Topic Model (CTM). Estimation uses semi-collapsed variational EM, the same algorithm as the R `stm()` function.

## Installation

```bash
pip install structural-topic-model
```

## Quick Start

```python
import numpy as np
from pystm import StructuralTopicModel

# X: (n_docs, n_vocab) word count matrix (dense or scipy.sparse)
# covar: (n_docs, n_covariates) prevalence covariate matrix (intercept added automatically)

model = StructuralTopicModel(n_components=10, init="spectral")
model.fit(X, prevalence=covar)

model.theta_        # topic proportions of training docs (n_docs, K)
model.components_   # topic-word distributions (K, V), each row sums to 1
model.gamma_        # prevalence regression coefficients (1+P, K-1)
model.sigma_        # topic covariance matrix (K-1, K-1)

# Inference on new documents (cf. fitNewDocuments)
theta_new = model.transform(X_new, prevalence=covar_new)

# Top words per topic (cf. labelTopics)
model.top_words(n_words=10)               # by probability
model.top_words(n_words=10, kind="frex")  # FREX: frequency-exclusivity balance
```

Without covariates, the model is estimated as a CTM:

```python
model = StructuralTopicModel(n_components=10).fit(X)
```

## Content Covariates

Content covariates allow the vocabulary used within topics to vary by document category. Pass one categorical label per document:

```python
model = StructuralTopicModel(n_components=10)
model.fit(X, prevalence=covar, content=party_labels)

model.aspect_components_   # per-level topic-word distributions (n_levels, K, V)
model.kappa_["params"]     # sparse deviations from baseline (lasso-estimated)
model.content_levels_      # sorted unique levels

model.transform(X_new, prevalence=covar_new, content=labels_new)
```

Estimated via Distributed Poisson regression (equivalent to the R package's default `kappa.prior="L1"`).

## Covariate Effects (estimateEffect)

Regress topic proportions on covariates using method of composition, returning coefficients with measurement uncertainty:

```python
from pystm import estimate_effect

eff = estimate_effect(model, covar, uncertainty="Global", nsims=25)
tables = eff.summary()      # {topic: structured array with estimate/std_error/t_value/p_value}
tables[0]["estimate"]       # coefficients for topic 0 (first entry is intercept)
```

## Topic Selection (searchK)

```python
from pystm import search_k

res = search_k(X, K_values=[5, 10, 15], prevalence=covar,
               model_params={"max_iter": 100})
res["heldout"]   # heldout log-likelihood (document completion)
res["residual"]  # residual dispersion — Taddy (2012), closer to 1 is better
res["semcoh"]    # semantic coherence
res["exclus"]    # exclusivity
```

## Diagnostics

```python
from pystm import topic_corr, semantic_coherence, exclusivity, check_residuals

tc = topic_corr(model, cutoff=0.01)    # topic correlation graph (simple method)
tc.posadj                               # positive correlation adjacency matrix
semantic_coherence(model, X, M=10)     # semantic coherence per topic
exclusivity(model, M=10)               # exclusivity per topic
check_residuals(model, X)              # residual dispersion test {dispersion, pvalue, df}
```

## R–Python API Reference

| R | Python |
|---|---|
| `stm(docs, vocab, K, prevalence=~x, data=meta)` | `StructuralTopicModel(n_components=K).fit(X, prevalence=design)` |
| `init.type="Spectral"` (recommended, default) | `init="spectral"` (default) |
| `init.type="Random"` | `init="random"` |
| `gamma.prior="Pooled"` (default) | implemented (automatic when covariates are present) |
| `sigma.prior` | `sigma_prior` |
| `emtol` / `max.em.its` | `tol` / `max_iter` |
| `model=` (warm restart) | `warm_start=True` |
| `content=~group` (`kappa.prior="L1"`, default) | `fit(X, content=labels)` |
| `interactions` | `content_interactions` |
| `fitNewDocuments()` | `transform()` |
| `labelTopics()` | `top_words()` |
| `estimateEffect()` / `summary()` | `estimate_effect()` / `.summary()` |
| `searchK()` | `search_k()` |
| `make.heldout()` / `eval.heldout()` | `make_heldout()` / `eval_heldout()` |
| `topicCorr(method="simple")` | `topic_corr()` |
| `semanticCoherence()` / `exclusivity()` / `checkResiduals()` | `semantic_coherence()` / `exclusivity()` / `check_residuals()` |
| `$theta` / `$beta` / `$sigma` / `$mu$gamma` | `theta_` / `components_` / `sigma_` / `gamma_` |
| `$beta$logbeta` (content model) | `aspect_components_` (probability scale) |
| `$beta$kappa` | `kappa_` |

### Differences from scikit-learn LDA

- `components_` contains normalized probability distributions (sklearn LDA stores unnormalized pseudo-counts).
- Covariates are passed via `fit(X, prevalence=...)` / `transform(X, prevalence=...)`. R formulas are not supported — encode categorical variables numerically beforehand (e.g. `pandas.get_dummies` or `patsy`).
- `perplexity(X)` is provided, computed from the variational lower bound (`exp(-bound / n_tokens)`); lower is better.
- `warm_start=True` enables continued learning across repeated `fit()` calls (cf. R's `model=` argument).

### Not Implemented

- `gamma.prior="L1"` (prevalence-side glmnet mode)
- `kappa.prior="Jeffreys"` (legacy content estimation)
- `fixedintercept=FALSE`
- LDA (collapsed Gibbs) initialization, `ngroups` memoized inference, `K=0` (Lee & Mimno)
- `estimateEffect()` with `uncertainty="Local"`, formula interface (pass pre-expanded basis matrices instead)
- `topicCorr(method="huge")`, `selectModel()`, `permutationTest()`, plot functions

## Development

```bash
uv sync
uv run pytest tests/
```

## References

- Roberts, M., Stewart, B., & Tingley, D. (2019). stm: An R Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
- Arora, S. et al. (2013). A Practical Algorithm for Topic Modeling with Provable Guarantees. *ICML*.
