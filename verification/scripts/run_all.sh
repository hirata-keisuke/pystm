#!/usr/bin/env bash
# コンテナ内で 前処理 -> R/Python STM -> 比較 を順に実行する。
# R 側は recoverEG=FALSE (quadprog 厳密解) を使い、pystm のデフォルト NNLS
# ソルバと Spectral 初期 beta を一致させた上で結果を比較する。
set -euo pipefail

echo "================ [1/4] R preprocess (DTM/vocab/meta/rds) ================"
Rscript /work/scripts/run_r_stm.R

echo "================ [2/4] R stm fit (Spectral, recoverEG=FALSE) ============"
Rscript /work/scripts/run_r_fit.R

echo "================ [3/4] pystm fit (Spectral, NNLS) ======================="
python /work/scripts/run_py_stm.py

echo "================ [4/4] compare =========================================="
python /work/scripts/compare.py

echo "================ DONE ================"
echo "outputs in /work/output:"
ls -la /work/output
