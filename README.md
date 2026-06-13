# pystm — Structural Topic Model in Python

R の [stm](https://github.com/bstewart/stm) パッケージ(Roberts, Stewart & Tingley)のコア推定アルゴリズムを Python に移植したものです。API は scikit-learn の `LatentDirichletAllocation` に倣っています。

## STM とは

STM はロジスティック正規トピックモデルで、文書のメタデータ(prevalence 共変量)が各文書のトピック比率の事前平均をシフトさせます。共変量なしの場合は Correlated Topic Model (CTM) に帰着します。推定は semi-collapsed 変分 EM で行います(R 版 `stm()` と同一のアルゴリズム)。

## 使い方

```python
import numpy as np
from pystm import StructuralTopicModel

# X: (n_docs, n_vocab) の単語カウント行列(dense / scipy.sparse どちらも可)
# covar: (n_docs, n_covariates) の prevalence 共変量(切片は自動付与)

model = StructuralTopicModel(n_components=10, init="spectral")
model.fit(X, prevalence=covar)

model.theta_        # 学習文書のトピック比率 (n_docs, K)
model.components_   # トピック-単語分布 (K, V)。各行の和は1
model.gamma_        # prevalence 回帰係数 (1+P, K-1)。先頭行が切片
model.sigma_        # トピック共分散行列 (K-1, K-1)

# 新規文書の推論(fitNewDocuments 相当)
theta_new = model.transform(X_new, prevalence=covar_new)

# トピックの代表語(labelTopics 相当)
model.top_words(n_words=10)               # 確率順
model.top_words(n_words=10, kind="frex")  # FREX(頻度と排他性のバランス)
```

共変量を渡さなければ CTM として推定されます:

```python
model = StructuralTopicModel(n_components=10).fit(X)
```

### content 共変量(SAGE / Distributed Multinomial Regression)

文書のカテゴリによってトピック内の語彙の使い方が変わるモデルです。各文書に1つのカテゴリラベルを渡します:

```python
model = StructuralTopicModel(n_components=10)
model.fit(X, prevalence=covar, content=party_labels)  # 例: 政党ラベル

model.aspect_components_   # カテゴリ別トピック-語彙分布 (n_levels, K, V)
model.kappa_["params"]     # ベースラインからのスパースな偏差(lasso 推定)
model.content_levels_      # カテゴリ水準
# transform / score にも同じ content を渡す
model.transform(X_new, prevalence=c_new, content=labels_new)
```

R 版の `kappa.prior="L1"`(既定)に相当する Distributed Poisson 回帰で推定します。glmnet の代わりに、設計行列のインジケータ構造を利用して語彙方向に完全ベクトル化した IRLS+座標降下の Poisson lasso を実装しています(正則化パスと情報量規準による選択も R と同じ)。

### estimateEffect 相当: 共変量効果の推定

トピック比率を目的変数とする回帰を method of composition(変分事後分布からの θ サンプリング)で行い、測定不確実性込みの係数を返します:

```python
from pystm import estimate_effect

eff = estimate_effect(model, covar, uncertainty="Global", nsims=25)
tables = eff.summary()        # {topic: 構造化配列(estimate/std_error/t_value/p_value)}
tables[0]["estimate"]         # トピック0の回帰係数(先頭が切片)
```

`uncertainty="Global"`(推奨・既定)と `"None"` をサポートします(R の `"Local"` は未実装)。

### searchK 相当: トピック数の選択

```python
from pystm import search_k

res = search_k(X, K_values=[5, 10, 15], prevalence=covar,
               model_params={"max_iter": 100})
res["heldout"]   # document completion による heldout 対数尤度
res["residual"]  # Taddy (2012) の残差分散(1 に近いほど良い)
res["semcoh"]    # 意味的一貫性 / res["exclus"]: 排他性
res["bound"], res["lbound"], res["em_its"]
```

### その他の診断

```python
from pystm import topic_corr, semantic_coherence, exclusivity, check_residuals

tc = topic_corr(model, cutoff=0.01)   # トピック相関グラフ(simple 法)
tc.posadj                              # 正相関の隣接行列
semantic_coherence(model, X, M=10)     # トピックごとの意味的一貫性
exclusivity(model, M=10)               # トピックごとの排他性(content モデル不可)
check_residuals(model, X)              # 残差分散検定 {dispersion, pvalue, df}
```

## R 版との対応

| R | Python |
|---|---|
| `stm(docs, vocab, K, prevalence=~x, data=meta)` | `StructuralTopicModel(n_components=K).fit(X, prevalence=design)` |
| `init.type="Spectral"` (推奨・既定) | `init="spectral"` (既定) |
| `init.type="Random"` | `init="random"` |
| `gamma.prior="Pooled"` (既定) | 実装済み(共変量ありのとき自動) |
| `sigma.prior` | `sigma_prior` |
| `emtol` / `max.em.its` | `tol` / `max_iter` |
| `model=`(フィット済みモデルから再開) | `warm_start=True`(fit を繰り返し呼ぶ。`bound_` に履歴が蓄積) |
| `content=~group`(`kappa.prior="L1"`, 既定) | `fit(X, content=labels)` |
| `interactions` | `content_interactions` |
| `fitNewDocuments()` | `transform()` |
| `labelTopics()` | `top_words()` |
| `estimateEffect()` / `summary()` | `estimate_effect()` / `.summary()` |
| `searchK()` | `search_k()` |
| `make.heldout()` / `eval.heldout()` | `make_heldout()` / `eval_heldout()` |
| `topicCorr(method="simple")` | `topic_corr()` |
| `semanticCoherence()` / `exclusivity()` / `checkResiduals()` | `semantic_coherence()` / `exclusivity()` / `check_residuals()` |
| `$theta` / `$beta` / `$sigma` / `$mu$gamma` | `theta_` / `components_` / `sigma_` / `gamma_` |
| `$beta$logbeta`(content モデル) | `aspect_components_`(確率スケール) |
| `$beta$kappa` | `kappa_` |

### scikit-learn LDA との API 差分

- `perplexity(X)` を sklearn LDA と同様に提供(変分下界ベースの `exp(-bound/総トークン数)`。低いほど良い)。
- `warm_start=True` で sklearn 流の継続学習(R 版 `model=` 相当)。
- `components_` は正規化済みの確率分布(sklearn LDA は擬似カウント)。
- 共変量は `fit(X, prevalence=...)` / `transform(X, prevalence=...)` のキーワードで渡す。R の formula は使えないので、カテゴリ変数は事前に one-hot 等で数値化してください(`patsy` や `pandas.get_dummies` が便利)。

### 未実装

- `gamma.prior="L1"`(prevalence 側の glmnet 依存モード)
- `kappa.prior="Jeffreys"`(content の旧推定法。R 版でも後方互換のためだけに残されている)
- `fixedintercept=FALSE`(content モデルの切片推定)
- LDA(collapsed Gibbs)初期化、`ngroups` メモ化推論、`K=0`(Lee & Mimno)
- `estimateEffect()` の `uncertainty="Local"`、formula インターフェース(スプライン `s()` 等は事前に基底展開した行列を渡せば等価)
- `topicCorr(method="huge")`(huge パッケージ依存)、`selectModel()`、`permutationTest()`、プロット関数群

また、spectral 初期化の RecoverL2 は R 版既定の quadprog の代わりにペナルティ付き NNLS による厳密に近い解法を使います(指数勾配法 `recoverEG=TRUE` 相当も `pystm._spectral.recover_l2(solver="expgrad")` として利用可能)。

## 実装メモ(R 版からの移植で見つかった重要な点)

1. **`update.mu` の切り替えタイミング**: R 版(`stm.control.R`)では E-step に渡す事前平均の選択を `update.mu = !is.null(mu$gamma)` で判定している。つまり**初回 E-step は共有平均(ゼロベクトル)を使い、γ が推定された 2 回目以降に文書別の事前平均 Xγ に切り替わる**。「prevalence 共変量があるか」で判定すると、初回 E-step で形状不一致または誤った事前を使うバグになる(本実装も最初これを踏んだ)。

2. **RecoverL2 のソルバー選択**: R 版の既定は quadprog による厳密な単体制約付き QP(`recoverEG=FALSE`)。論文由来の指数勾配法(`recoverEG=TRUE`)は反復上限 500 では、**1つのトピックが支配的なコーパス(文書内でトピックが強く混ざる場合)に収束不足**となり、初期化品質が大きく劣化した(K=10 の合成データで cos 類似度 0.45 前後 vs NNLS で 0.97)。反復を 20,000 まで増やしても改善しなかったため、最適化の遅さではなく平坦な目的関数で実質停止していた。本実装はペナルティ付き NNLS(和=1 制約を重み付き行で課す)を既定とした。

3. **Random 初期化は局所解に落ちやすい**(R 版ドキュメントの記述どおり、seed によりトピック復元が大きく変わる)。動作確認・検証には決定的な Spectral 初期化を使うこと。

4. **gram 行列の検証方法**: スペクトル初期化の正しさは、合成データで経験 gram 行列が理論期待値 `β' E[θθ'] β`(行正規化後)と一致するかで切り分けられる(本実装では最大誤差 ~1e-3 で一致)。初期化品質が悪いときは実装バグではなく、コーパス側の共起信号の弱さ(θ の混合度)が原因のことがある。

5. **mnreg(content 共変量)の高速化**: Distributed Poisson 回帰の設計行列は「トピック主効果 / アスペクト主効果 / 交互作用」の3グループのインジケータ列で、**各グループ内の列は互いに素な行しか触らない**。そのためグループ単位の座標降下が1回の行列演算になり、さらに全語彙が同一の設計行列を共有するので V 方向にも完全ベクトル化できる。汎用の座標降下実装と比べ同一解で大幅に高速(さらに IRLS 上限 4 / スイープ上限 8 / tol 1e-4 に絞っても β の最大差は ~2e-5)。

6. **同梱 gadarianFit の前処理は現行 textProcessor と異なる**: パッケージ同梱の `gadarianFit`(2017年)の語彙は、現行 `textProcessor.R` の処理順(句読点除去→ストップワード除去、ダッシュ保存)では再現できない。旧版の処理順(**ストップワード除去が句読点除去より先**=アポストロフィ付きの "can't" 等がストップワードとして除去される、かつ**ダッシュ非保存**= "tax-payers"→"taxpayers")+ `lower.thresh=3` で215語が完全一致する([scripts/gadarian_prep.py](scripts/gadarian_prep.py) の `legacy_order=True`)。R 版の再現実験をする際はパッケージバージョンごとの前処理差に注意。

7. **E-step の数値ガードが実データでは必須**: R/C++ 原実装どおりの素朴な `exp(eta)` 計算は、実コーパス(短文・偏った β・大きめ K)で BFGS の直線探索が極端な点を踏んだときに inf/NaN を発生させ、Hessian の cholesky が落ちる。η のクリップ(±200)、log と除算の下限(1e-300)、非有限解のフォールバックを `_estep.py` に追加した(通常領域の値は不変、合成データ・gadarian 検証とも退行なし)。

8. **heldout 構築時の語彙消失**: トークンを訓練側から取り除くと、コーパス全体から消える語が生じうる。R 版 `make.heldout` は語彙を再番号付けして missing 側からも削除している。これを怠ると、その語の β が 0 になり heldout 対数尤度が -inf になる。本実装も missing 側から該当トークンを除外している。

## R 版との検証(gadarianFit)

R パッケージ同梱の `gadarianFit`(Roberts et al. 2014 AJPS の Gadarian & Albertson 移民調査データ、K=3、prevalence = treatment*pid_rep、N=341)を参照解として、本実装を数値レベルで検証済み。再現方法:

```bash
uv run python scripts/validate_gadarian.py   # 11/11 チェック合格
```

| 検証項目 | 結果 |
|---|---|
| コーパス再現(textProcessor + prepDocuments の移植) | 語彙215語・単語カウントともR版と**完全一致** |
| R版パラメータでの E-step bound | pystm -13575.82 vs R -13575.91(R の1反復増分 0.103 の範囲内で一致) |
| R版パラメータでの文書別 θ | 相関 > 0.9999、最大差 0.02 |
| R版の解からの EM 継続(不動点チェック) | bound 単調増加・増分は収束閾値レベル(10反復で +1.86) |
| 独立フィット(Spectral 初期化)の bound | R比 -0.18%(R は確率的 LDA 初期化なので局所解の違いは想定内) |
| トピックの対応 | 3トピックとも cos 類似度 0.88 前後、上位語ほぼ一致(worri/immigr/border、job/tax/pay、peopl/countri/come) |
| treatment 効果 | 全トピックで符号一致。有意な正の効果は +0.215 vs R +0.219 とほぼ同値 |

注: R 版のフィット自体は LDA Gibbs 初期化(R の乱数)に依存するため完全一致は原理的に不可能。代わりに「R の解が本実装の EM の不動点になっているか」「bound 計算が R の報告値と一致するか」で実装の同一性を確認している。

## 開発

```bash
uv sync
uv run pytest tests/
```

## 他プロジェクトからの利用(配布)

配布名・import 名ともに `pystm`。実行時依存は numpy / scipy / scikit-learn のみ
(janome / dash 等は application 用の dev 依存で、配布物には含まれない)。

```bash
# PyPI からインストール
pip install structural-topic-model
# または
uv add structural-topic-model

# ローカルパス参照(開発中)
uv add --editable /path/to/202606_StructuralTopicModel

# Git 経由
uv add git+<リポジトリURL>

# wheel をビルド
uv build   # dist/structural_topic_model-x.y.z-py3-none-any.whl
```

配布名は `structural-topic-model`、import 名は `pystm` のまま維持しています
(PyPI の `pystm` は別の実装に取られているため)。

## 参考文献

- Roberts, M., Stewart, B., & Tingley, D. (2019). stm: An R Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
- Arora, S. et al. (2013). A Practical Algorithm for Topic Modeling with Provable Guarantees. *ICML*.
