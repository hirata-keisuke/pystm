#!/usr/bin/env python3
# =============================================================================
# pystm の spectral_init を直接呼んで「初期 beta (K x V)」を取得し、共有する。
# この初期 beta を R(custom.beta) と pystm(warm_start) の両方に注入することで、
# 初期化の差を排除し、推定部 (E/M ステップ) の一致だけを検証できるようにする。
# =============================================================================
import os
import numpy as np
import pandas as pd
from scipy.io import mmread
from stm._spectral import spectral_init

SHARED = "/work/shared"
dtm = mmread(os.path.join(SHARED, "dtm.mtx")).tocsr()
K = 3

# pystm 内部の Spectral 初期化 (R spectral.R の移植) をそのまま使う。
# StructuralTopicModel のデフォルト (max_vocab=10000) と同一引数にして、
# run_py_stm.py の fit 内部が呼ぶ spectral_init と完全に一致させる。
init_beta, Kout = spectral_init(dtm, K, max_vocab=10000,
                                solver="nnls", random_state=12345)
assert Kout == K, f"K mismatch: {Kout}"
# 念のため各行が確率分布(和=1)であることを確認
init_beta = init_beta / init_beta.sum(axis=1, keepdims=True)
pd.DataFrame(init_beta).to_csv(os.path.join(SHARED, "init_beta.csv"), index=False)
print(f"[init] exported init_beta shape={init_beta.shape}, rowsums~{init_beta.sum(axis=1)}")
