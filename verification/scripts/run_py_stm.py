#!/usr/bin/env python3
# =============================================================================
# pystm (structural-topic-model) を、R がエクスポートした「同一 DTM」で実行する。
#   - 入力 DTM/vocab/meta は /work/shared から読み込む (R prepDocuments の結果)。
#   - prevalence は treatment 列のみ (intercept は pystm が自動付与)。
#   - K=3, init="spectral" で学習し、beta/theta/effect/収束をエクスポート。
# 出力先: /work/output
# =============================================================================
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from stm import StructuralTopicModel, estimate_effect

SHARED = "/work/shared"
OUT = "/work/output"
os.makedirs(OUT, exist_ok=True)

# ---- 1. 共有 DTM / vocab / meta を読み込み --------------------------------
dtm = mmread(os.path.join(SHARED, "dtm.mtx")).tocsr()  # (n_docs, n_vocab)
with open(os.path.join(SHARED, "vocab.txt")) as f:
    vocab = [line.rstrip("\n") for line in f]
meta = pd.read_csv(os.path.join(SHARED, "meta.csv"))
print(f"[PY] DTM shape={dtm.shape}, vocab={len(vocab)}, meta rows={len(meta)}")

# prevalence: treatment 列のみ (R も ~treatment で intercept 自動付与)
covar = meta[["treatment"]].to_numpy(dtype=float)  # (n_docs, 1)

# ---- 2. STM 実行 -----------------------------------------------------------
K = 3
model = StructuralTopicModel(n_components=K, init="spectral",
                             max_iter=200, random_state=12345)
model.fit(dtm, prevalence=covar)
print("[PY] stm fit done")

# ---- 3. 結果エクスポート ---------------------------------------------------
# beta: components_ は (K, V) 正規化済み確率
beta = np.asarray(model.components_)
pd.DataFrame(beta).to_csv(os.path.join(OUT, "py_beta.csv"), index=False)

# theta: (n_docs, K)
theta = np.asarray(model.theta_)
pd.DataFrame(theta).to_csv(os.path.join(OUT, "py_theta.csv"), index=False)

# 収束 bound (利用可能なら)
bound = None
for attr in ("bound_", "elbo_", "convergence_", "loss_history_"):
    if hasattr(model, attr):
        bound = np.asarray(getattr(model, attr)).ravel()
        break
if bound is not None and bound.size > 0:
    pd.DataFrame({"iter": np.arange(1, bound.size + 1), "bound": bound}) \
        .to_csv(os.path.join(OUT, "py_bound.csv"), index=False)
else:
    print("[PY] WARNING: convergence bound attribute not found")

# estimateEffect: treatment 効果
eff = estimate_effect(model, covar, uncertainty="Global", nsims=25)
tables = eff.summary()
rows = []
for k in range(K):
    tbl = tables[k]
    est = np.asarray(tbl["estimate"]).ravel()
    se = (np.asarray(tbl["std_error"]).ravel()
          if "std_error" in tbl.dtype.names else np.full(est.shape, np.nan))
    # term 0 = intercept, term 1 = treatment (順序は R に合わせて出力)
    for t in range(len(est)):
        term = "(Intercept)" if t == 0 else f"covar{t}"
        rows.append({"topic": k + 1, "term": term,
                     "estimate": est[t], "std_error": se[t]})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "py_effect.csv"), index=False)

print("[PY] all exports done")
