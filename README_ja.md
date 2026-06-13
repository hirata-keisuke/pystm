# pystm — Python による構造的トピックモデル

[![PyPI](https://img.shields.io/pypi/v/structural-topic-model)](https://pypi.org/project/structural-topic-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

R の [stm](https://github.com/bstewart/stm) パッケージ (Roberts, Stewart & Tingley) のコア推定アルゴリズムを Python に移植したものです。API は scikit-learn の `LatentDirichletAllocation` に倣っています。

[English README](https://github.com/hirata-keisuke/pystm/blob/main/README.md)

## STM とは

STM (Structural Topic Model) はロジスティック正規トピックモデルで、文書のメタデータ（prevalence 共変量）が各文書のトピック比率の事前平均をシフトさせます。共変量なしの場合は Correlated Topic Model (CTM) に帰着します。推定は semi-collapsed 変分 EM で行います（R 版 `stm()` と同一のアルゴリズム）。

## インストール

```bash
pip install structural-topic-model
```

## クイックスタート

```python
import numpy as np
from pystm import StructuralTopicModel

# X: (n_docs, n_vocab) の単語カウント行列（dense / scipy.sparse どちらも可）
# covar: (n_docs, n_covariates) の prevalence 共変量（切片は自動付与）

model = StructuralTopicModel(n_components=10, init="spectral")
model.fit(X, prevalence=covar)

model.theta_        # 学習文書のトピック比率 (n_docs, K)
model.components_   # トピック-単語分布 (K, V)。各行の和は 1
model.gamma_        # prevalence 回帰係数 (1+P, K-1)。先頭行が切片
model.sigma_        # トピック共分散行列 (K-1, K-1)

# 新規文書の推論（fitNewDocuments 相当）
theta_new = model.transform(X_new, prevalence=covar_new)

# トピックの代表語（labelTopics 相当）
model.top_words(n_words=10)               # 確率順
model.top_words(n_words=10, kind="frex")  # FREX（頻度と排他性のバランス）
```

共変量を渡さなければ CTM として推定されます：

```python
model = StructuralTopicModel(n_components=10).fit(X)
```

## content 共変量

文書のカテゴリによってトピック内の語彙の使い方が変わるモデルです。各文書に 1 つのカテゴリラベルを渡します：

```python
model = StructuralTopicModel(n_components=10)
model.fit(X, prevalence=covar, content=party_labels)  # 例: 政党ラベル

model.aspect_components_   # カテゴリ別トピック-語彙分布 (n_levels, K, V)
model.kappa_["params"]     # ベースラインからのスパースな偏差（lasso 推定）
model.content_levels_      # カテゴリ水準

model.transform(X_new, prevalence=covar_new, content=labels_new)
```

R 版の `kappa.prior="L1"`（既定）に相当する Distributed Poisson 回帰で推定します。

## 共変量効果の推定（estimateEffect 相当）

トピック比率を目的変数とする回帰を method of composition（変分事後分布からの θ サンプリング）で行い、測定不確実性込みの係数を返します：

```python
from pystm import estimate_effect

eff = estimate_effect(model, covar, uncertainty="Global", nsims=25)
tables = eff.summary()      # {topic: 構造化配列（estimate/std_error/t_value/p_value）}
tables[0]["estimate"]       # トピック 0 の回帰係数（先頭が切片）
```

## トピック数の選択（searchK 相当）

```python
from pystm import search_k

res = search_k(X, K_values=[5, 10, 15], prevalence=covar,
               model_params={"max_iter": 100})
res["heldout"]   # document completion による heldout 対数尤度
res["residual"]  # Taddy (2012) の残差分散（1 に近いほど良い）
res["semcoh"]    # 意味的一貫性
res["exclus"]    # 排他性
```

## 診断

```python
from pystm import topic_corr, semantic_coherence, exclusivity, check_residuals

tc = topic_corr(model, cutoff=0.01)    # トピック相関グラフ（simple 法）
tc.posadj                               # 正相関の隣接行列
semantic_coherence(model, X, M=10)     # トピックごとの意味的一貫性
exclusivity(model, M=10)               # トピックごとの排他性
check_residuals(model, X)              # 残差分散検定 {dispersion, pvalue, df}
```

## R 版との対応表

| R | Python |
|---|---|
| `stm(docs, vocab, K, prevalence=~x, data=meta)` | `StructuralTopicModel(n_components=K).fit(X, prevalence=design)` |
| `init.type="Spectral"`（推奨・既定） | `init="spectral"`（既定） |
| `init.type="Random"` | `init="random"` |
| `gamma.prior="Pooled"`（既定） | 実装済み（共変量ありのとき自動） |
| `sigma.prior` | `sigma_prior` |
| `emtol` / `max.em.its` | `tol` / `max_iter` |
| `model=`（ウォームスタート） | `warm_start=True` |
| `content=~group`（`kappa.prior="L1"`, 既定） | `fit(X, content=labels)` |
| `interactions` | `content_interactions` |
| `fitNewDocuments()` | `transform()` |
| `labelTopics()` | `top_words()` |
| `estimateEffect()` / `summary()` | `estimate_effect()` / `.summary()` |
| `searchK()` | `search_k()` |
| `make.heldout()` / `eval.heldout()` | `make_heldout()` / `eval_heldout()` |
| `topicCorr(method="simple")` | `topic_corr()` |
| `semanticCoherence()` / `exclusivity()` / `checkResiduals()` | `semantic_coherence()` / `exclusivity()` / `check_residuals()` |
| `$theta` / `$beta` / `$sigma` / `$mu$gamma` | `theta_` / `components_` / `sigma_` / `gamma_` |
| `$beta$logbeta`（content モデル） | `aspect_components_`（確率スケール） |
| `$beta$kappa` | `kappa_` |

### scikit-learn LDA との API 差分

- `components_` は正規化済みの確率分布（sklearn LDA は擬似カウント）。
- 共変量は `fit(X, prevalence=...)` / `transform(X, prevalence=...)` のキーワードで渡します。R の formula は使えないので、カテゴリ変数は事前に数値化してください（`pandas.get_dummies` や `patsy` が便利です）。
- `perplexity(X)` を提供（変分下界ベースの `exp(-bound/総トークン数)`。低いほど良い）。
- `warm_start=True` で継続学習が可能（R 版 `model=` 相当）。

### 未実装

- `gamma.prior="L1"`（prevalence 側の glmnet 依存モード）
- `kappa.prior="Jeffreys"`（content の旧推定法）
- `fixedintercept=FALSE`
- LDA（collapsed Gibbs）初期化、`ngroups` メモ化推論、`K=0`（Lee & Mimno）
- `estimateEffect()` の `uncertainty="Local"`、formula インターフェース（スプライン `s()` 等は事前に基底展開した行列を渡せば等価）
- `topicCorr(method="huge")`、`selectModel()`、`permutationTest()`、プロット関数群

## 開発

```bash
uv sync
uv run pytest tests/
```

## 参考文献

- Roberts, M., Stewart, B., & Tingley, D. (2019). stm: An R Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
- Arora, S. et al. (2013). A Practical Algorithm for Topic Modeling with Provable Guarantees. *ICML*.
