#!/usr/bin/env Rscript
# =============================================================================
# R stm を gadarian データで実行し、結果と「共有用 DTM」をエクスポートする。
#   - gadarian は stm パッケージ同梱データ (gadarian.RData 相当)
#   - prepDocuments の結果 (documents/vocab/meta) をエクスポートし、
#     Python(pystm) が完全に同一の入力を使えるようにする。
#   - K=3, init.type="Spectral" (決定的) で学習。
# 出力先: /work/shared (Python と共有), /work/output (比較用)
# =============================================================================

suppressMessages({
  library(stm)
  library(Matrix)
})

set.seed(12345)

shared_dir <- "/work/shared"
out_dir    <- "/work/output"
dir.create(shared_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(out_dir,    showWarnings = FALSE, recursive = TRUE)

# ---- 1. データロード -------------------------------------------------------
# stm パッケージに gadarian が同梱されている (gadarian.RData)。
data(gadarian, package = "stm")
cat(sprintf("[R] gadarian: %d docs, columns: %s\n",
            nrow(gadarian), paste(colnames(gadarian), collapse = ", ")))

# ---- 2. 前処理 -------------------------------------------------------------
processed <- textProcessor(gadarian$open.ended.response,
                           metadata = gadarian,
                           verbose = FALSE)
out <- prepDocuments(processed$documents, processed$vocab,
                     processed$meta, verbose = FALSE)
docs  <- out$documents      # list: 各文書 2xN_j 行列 (行1=単語index(1始まり), 行2=count)
vocab <- out$vocab          # character vector
meta  <- out$meta           # data.frame (treatment, pid_rep, ...)

cat(sprintf("[R] after prepDocuments: %d docs, %d vocab terms\n",
            length(docs), length(vocab)))

# ---- 3. 共有用 DTM を疎行列(MatrixMarket)としてエクスポート -----------------
# documents(list) を (n_docs x n_vocab) の疎行列に変換。
n_docs  <- length(docs)
n_vocab <- length(vocab)
i_idx <- integer(0); j_idx <- integer(0); x_val <- numeric(0)
for (d in seq_len(n_docs)) {
  m <- docs[[d]]
  if (length(m) == 0) next
  i_idx <- c(i_idx, rep(d, ncol(m)))
  j_idx <- c(j_idx, m[1, ])      # 単語 index (1始まり)
  x_val <- c(x_val, m[2, ])      # count
}
dtm <- sparseMatrix(i = i_idx, j = j_idx, x = x_val,
                    dims = c(n_docs, n_vocab))
Matrix::writeMM(dtm, file = file.path(shared_dir, "dtm.mtx"))
writeLines(vocab, file.path(shared_dir, "vocab.txt"))
write.csv(meta, file.path(shared_dir, "meta.csv"), row.names = FALSE)
cat("[R] exported shared DTM/vocab/meta\n")

# 前処理結果を RDS で保存し、STM 実行スクリプト(run_r_fit.R)に引き継ぐ。
# (init_beta は本スクリプトの後・fit の前に Python で生成されるため分離する)
saveRDS(list(docs = docs, vocab = vocab, meta = meta),
        file = file.path(shared_dir, "r_prep.rds"))
cat("[R] saved r_prep.rds (preprocess done)\n")
