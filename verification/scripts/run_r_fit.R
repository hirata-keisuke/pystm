#!/usr/bin/env Rscript
# =============================================================================
# R stm の STM 実行部。run_r_stm.R が出力した前処理結果(r_prep.rds)を読み込み、
# pystm の spectral_init が出した初期 beta(init_beta.csv)を Custom 注入して学習する。
# これにより初期化の差を排除し、推定部(E/M)の一致を検証する。
# init_beta.csv が無ければ通常の Spectral 初期化にフォールバック。
# =============================================================================
suppressMessages({
  library(stm)
})

set.seed(12345)
shared_dir <- "/work/shared"
out_dir    <- "/work/output"

prep  <- readRDS(file.path(shared_dir, "r_prep.rds"))
docs  <- prep$docs
vocab <- prep$vocab
meta  <- prep$meta
K <- 3

# Spectral 初期化を使う。ただし recoverEG=FALSE を指定して beta 復元を
# quadprog の厳密解にする。
#   - R のデフォルト recoverEG=TRUE (指数勾配法 expgrad) はこのデータで
#     収束せず振動し、BLAS の丸め差だけで解が変わる。pystm も同じ式だが
#     非収束ゆえ別解になる (両者とも実装は正しいが数値的に不安定)。
#   - quadprog 厳密解は決定的で、pystm の NNLS ソルバ (デフォルト) と
#     初期 beta が完全一致する。これで初期化を揃え推定部の一致を検証する。
cat("[R] using Spectral init with recoverEG=FALSE (quadprog exact solve)\n")
fit <- stm(documents = docs, vocab = vocab, K = K,
           prevalence = ~ treatment, data = meta,
           init.type = "Spectral",
           control = list(recoverEG = FALSE),
           seed = 12345, max.em.its = 200, verbose = FALSE)
cat("[R] stm fit done\n")

# ---- 結果エクスポート ------------------------------------------------------
beta <- exp(fit$beta$logbeta[[1]])           # K x V
write.csv(beta, file.path(out_dir, "r_beta.csv"), row.names = FALSE)

theta <- fit$theta                           # n_docs x K
write.csv(theta, file.path(out_dir, "r_theta.csv"), row.names = FALSE)

bound <- fit$convergence$bound
write.csv(data.frame(iter = seq_along(bound), bound = bound),
          file.path(out_dir, "r_bound.csv"), row.names = FALSE)

prep_eff <- estimateEffect(1:K ~ treatment, fit, metadata = meta,
                           uncertainty = "Global")
sm <- summary(prep_eff)
eff_rows <- lapply(seq_len(K), function(k) {
  tbl <- sm$tables[[k]]
  data.frame(topic = k,
             term  = rownames(tbl),
             estimate = tbl[, "Estimate"],
             std_error = tbl[, "Std. Error"])
})
write.csv(do.call(rbind, eff_rows), file.path(out_dir, "r_effect.csv"),
          row.names = FALSE)

writeLines(vocab, file.path(out_dir, "vocab.txt"))
cat("[R] all exports done\n")
