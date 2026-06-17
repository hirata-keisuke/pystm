# =============================================================================
# R stm と pystm(structural-topic-model) を 1 コンテナで実行・比較する環境。
#   - rocker/r-ver: R 本体
#   - uv: Python 3.12+ と pystm を管理 (pystm は numpy>=2.0, py>=3.12 が必要)
# =============================================================================
FROM rocker/r-ver:4.4.1

ENV DEBIAN_FRONTEND=noninteractive

# --- システム依存 (R パッケージのコンパイル & Python ビルド用) ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran \
        libxml2-dev libcurl4-openssl-dev libssl-dev \
        libglpk-dev libgsl-dev \
        ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# --- R パッケージ: stm + 依存 (Matrix は base 同梱) --------------------------
RUN Rscript -e 'install.packages(c("stm","tm","SnowballC"), repos="https://cloud.r-project.org")'
# tm が確実に入っていることを検証 (入っていなければビルドを失敗させる)
RUN Rscript -e 'stopifnot(requireNamespace("tm", quietly=TRUE), requireNamespace("SnowballC", quietly=TRUE), requireNamespace("stm", quietly=TRUE)); library(stm); library(tm); library(SnowballC); cat("OK stm:", as.character(packageVersion("stm")), "tm:", as.character(packageVersion("tm")), "\n")'

# --- uv (Python) ------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Python 3.12 + 必要パッケージ。pystm = structural-topic-model
RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN uv pip install --python /opt/venv/bin/python \
        "structural-topic-model" \
        "numpy>=2.0" "scipy>=1.14" "scikit-learn>=1.9.0" \
        pandas matplotlib \
    && python -c "import stm; print('pystm import OK')"

WORKDIR /work
CMD ["bash"]
