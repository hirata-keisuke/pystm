#!/usr/bin/env python3
# =============================================================================
# R stm と pystm の結果を比較してプロットする。
#   - トピック番号は実装間で順序が異なりうるため、beta のコサイン類似度 +
#     ハンガリアン法で 1:1 対応付けしてから比較する。
#   - beta(topic-word) / theta(doc-topic) / effect(treatment) / 収束 を比較。
# 出力: /work/output/*.png, /work/output/summary.txt
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

OUT = "/work/output"
log_lines = []
def log(s):
    print(s)
    log_lines.append(s)

# ---- load -----------------------------------------------------------------
r_beta = pd.read_csv(os.path.join(OUT, "r_beta.csv")).to_numpy()    # (K, V)
py_beta = pd.read_csv(os.path.join(OUT, "py_beta.csv")).to_numpy()  # (K, V)
r_theta = pd.read_csv(os.path.join(OUT, "r_theta.csv")).to_numpy()
py_theta = pd.read_csv(os.path.join(OUT, "py_theta.csv")).to_numpy()
with open(os.path.join(OUT, "vocab.txt")) as f:
    vocab = [l.rstrip("\n") for l in f]

K = r_beta.shape[0]
log(f"K={K}, V={r_beta.shape[1]}")

# ---- 1. トピック対応付け (beta コサイン類似度 -> ハンガリアン法) ------------
def cosine_matrix(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T

sim = cosine_matrix(r_beta, py_beta)          # (K_r, K_py)
row_ind, col_ind = linear_sum_assignment(-sim)  # 類似度最大化
mapping = dict(zip(row_ind, col_ind))         # R topic -> PY topic
log("topic mapping (R -> PY): " +
    ", ".join(f"R{r+1}->PY{c+1}(cos={sim[r,c]:.4f})"
              for r, c in zip(row_ind, col_ind)))

py_beta_m = py_beta[col_ind]      # R の順序に並べ替えた PY beta
py_theta_m = py_theta[:, col_ind]

# ---- 2. beta 比較 ----------------------------------------------------------
# (a) 各トピックの上位単語の重なり
log("\n--- top words overlap (top10) ---")
for k in range(K):
    r_top = set(np.argsort(r_beta[k])[::-1][:10])
    p_top = set(np.argsort(py_beta_m[k])[::-1][:10])
    jacc = len(r_top & p_top) / len(r_top | p_top)
    rw = [vocab[i] for i in np.argsort(r_beta[k])[::-1][:10]]
    pw = [vocab[i] for i in np.argsort(py_beta_m[k])[::-1][:10]]
    log(f"topic {k+1}: Jaccard={jacc:.2f}")
    log(f"   R : {rw}")
    log(f"   PY: {pw}")

# (b) beta 散布図 (全単語確率)
fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5))
if K == 1:
    axes = [axes]
for k in range(K):
    ax = axes[k]
    x, y = r_beta[k], py_beta_m[k]
    ax.scatter(x, y, s=6, alpha=0.4)
    lim = max(x.max(), y.max())
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    corr = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"beta topic{k+1} (corr={corr:.4f})")
    ax.set_xlabel("R stm"); ax.set_ylabel("pystm")
fig.suptitle("Topic-word probability (beta) comparison")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_beta.png"), dpi=130)
plt.close(fig)

# ---- 3. theta 比較 ---------------------------------------------------------
fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5))
if K == 1:
    axes = [axes]
for k in range(K):
    ax = axes[k]
    x, y = r_theta[:, k], py_theta_m[:, k]
    ax.scatter(x, y, s=10, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--", lw=1)
    corr = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"theta topic{k+1} (corr={corr:.4f})")
    ax.set_xlabel("R stm"); ax.set_ylabel("pystm")
fig.suptitle("Document-topic proportion (theta) comparison")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_theta.png"), dpi=130)
plt.close(fig)

log("\n--- theta correlation per topic ---")
for k in range(K):
    corr = np.corrcoef(r_theta[:, k], py_theta_m[:, k])[0, 1]
    log(f"topic {k+1}: corr={corr:.4f}")

# ---- 4. estimateEffect 比較 ------------------------------------------------
try:
    r_eff = pd.read_csv(os.path.join(OUT, "r_effect.csv"))
    py_eff = pd.read_csv(os.path.join(OUT, "py_effect.csv"))
    # treatment (intercept以外) の係数をトピックごとに比較
    def treat_estimate(df, topic_r):
        sub = df[df["topic"] == topic_r]
        nonint = sub[~sub["term"].str.contains("Intercept", case=False, na=False)]
        return float(nonint["estimate"].iloc[0]) if len(nonint) else np.nan
    rows = []
    for r in range(K):
        py_topic = mapping[r] + 1
        re = treat_estimate(r_eff, r + 1)
        pe = treat_estimate(py_eff, py_topic)
        rows.append((r + 1, py_topic, re, pe))
    log("\n--- treatment effect (estimate) ---")
    rt = [x[2] for x in rows]; pt = [x[3] for x in rows]
    for r, p, re_, pe_ in rows:
        log(f"R topic{r} (=PY{p}): R={re_:+.4f}  PY={pe_:+.4f}  diff={re_-pe_:+.4f}")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rt, pt, s=60)
    for r, p, re_, pe_ in rows:
        ax.annotate(f"T{r}", (re_, pe_))
    lim_lo = min(min(rt), min(pt)); lim_hi = max(max(rt), max(pt))
    pad = (lim_hi - lim_lo) * 0.1 + 1e-6
    ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad], "r--", lw=1)
    ax.set_xlabel("R stm treatment effect")
    ax.set_ylabel("pystm treatment effect")
    ax.set_title("estimateEffect (treatment) comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare_effect.png"), dpi=130)
    plt.close(fig)
except Exception as e:
    log(f"\n[effect comparison skipped] {e}")

# ---- 5. 収束 bound 比較 ----------------------------------------------------
try:
    r_b = pd.read_csv(os.path.join(OUT, "r_bound.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r_b["iter"], r_b["bound"], "-o", ms=3, label="R stm")
    py_path = os.path.join(OUT, "py_bound.csv")
    if os.path.exists(py_path):
        p_b = pd.read_csv(py_path)
        ax.plot(p_b["iter"], p_b["bound"], "-s", ms=3, label="pystm")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("bound / ELBO")
    ax.set_title("Convergence comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare_bound.png"), dpi=130)
    plt.close(fig)
    log(f"\n--- convergence ---")
    log(f"R final bound: {r_b['bound'].iloc[-1]:.4f} ({len(r_b)} iters)")
    if os.path.exists(py_path):
        log(f"PY final bound: {pd.read_csv(py_path)['bound'].iloc[-1]:.4f}")
except Exception as e:
    log(f"\n[bound comparison skipped] {e}")

# ---- summary 出力 ----------------------------------------------------------
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    f.write("\n".join(log_lines) + "\n")
print("\n[compare] plots + summary.txt written to /work/output")
