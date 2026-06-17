#!/usr/bin/env Rscript
# R stm の Spectral 初期 beta を取り出し、pystm の spectral_init 出力と比較する。
# 初期化が一致しているかを切り分ける。
suppressMessages({ library(stm) })
shared_dir <- "/work/shared"
prep  <- readRDS(file.path(shared_dir, "r_prep.rds"))
docs  <- prep$docs; vocab <- prep$vocab; meta <- prep$meta
K <- 3

# stm.init を直接呼んで Spectral 初期化の beta を取得する。
ns <- getNamespace("stm")
# stm() 内部と同じ settings を最小構成で組む必要があるため、
# max.em.its=0 で stm を回して初期状態(model$beta)を取り出す。
fit0 <- stm(documents = docs, vocab = vocab, K = K,
            prevalence = ~ treatment, data = meta,
            init.type = "Spectral", seed = 12345,
            max.em.its = 0, verbose = FALSE)
# max.em.its=0 のとき beta は初期値のまま (logbeta)
init_beta_R <- exp(fit0$beta$logbeta[[1]])   # K x V
write.csv(init_beta_R, file.path(shared_dir, "r_init_beta.csv"), row.names = FALSE)
cat("[R] exported r_init_beta.csv  dim:", dim(init_beta_R), "\n")
cat("[R] rowsums:", rowSums(init_beta_R), "\n")
