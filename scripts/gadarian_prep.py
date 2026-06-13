"""Replicate the R stm vignette preprocessing for the gadarian data.

Mirrors textProcessor() (tm pipeline: lowercase, punctuation/stopword/
number removal, Snowball stemming, wordLengths=c(3,Inf)) followed by
prepDocuments(lower.thresh=1).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import snowballstemmer

# tm::stopwords("en") — the Snowball English stopword list
TM_STOPWORDS_EN = """i me my myself we our ours ourselves you your yours
yourself yourselves he him his himself she her hers herself it its itself
they them their theirs themselves what which who whom this that these those
am is are was were be been being have has had having do does did doing would
should could ought i'm you're he's she's it's we're they're i've you've
we've they've i'd you'd he'd she'd we'd they'd i'll you'll he'll she'll
we'll they'll isn't aren't wasn't weren't hasn't haven't hadn't doesn't
don't didn't won't wouldn't shan't shouldn't can't cannot couldn't mustn't
let's that's who's what's here's there's when's where's why's how's a an
the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out
on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so
than too very""".split()


def text_processor(texts, legacy_order=False):
    """tm pipeline as run by textProcessor() with default arguments.

    ``legacy_order=True`` reproduces the ordering used by the older
    textProcessor that generated the packaged gadarianFit object:
    stopwords are removed while apostrophes are still present (so
    contractions like "can't" match the stopword list) and punctuation is
    stripped without preserving intra-word dashes ("tax-payers" becomes
    "taxpayers").  With ``prep_documents(lower_thresh=3)`` this exactly
    reproduces gadarianFit's 215-term vocabulary.
    """
    stemmer = snowballstemmer.stemmer("english")
    stop_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(w) for w in TM_STOPWORDS_EN) + r")\b"
    )
    docs = []
    for s in texts:
        s = re.sub(r"[^!-~]", " ", s)            # [^[:graph:]] -> space
        s = re.sub(r"\s+", " ", s).strip()       # stripWhitespace
        s = s.lower()
        if legacy_order:
            s = stop_pattern.sub("", s)          # stopwords first
            s = re.sub(r"[!-/:-@\[-`{-~]+", "", s)  # punct, dashes fused
        else:
            # removePunctuation(preserve_intra_word_dashes=TRUE)
            s = re.sub(r"(\w)-(\w)", "\\1\x01\\2", s)
            s = re.sub(r"[!-/:-@\[-`{-~]+", "", s)
            s = s.replace("\x01", "-")
            s = stop_pattern.sub("", s)          # removeWords(stopwords)
        s = re.sub(r"[0-9]+", "", s)             # removeNumbers
        tokens = s.split()
        tokens = stemmer.stemWords(tokens)       # stemDocument
        tokens = [t for t in tokens if len(t) >= 3]  # wordLengths=c(3,Inf)
        docs.append(tokens)
    return docs


def prep_documents(token_docs, lower_thresh=1):
    """prepDocuments(): drop words by document frequency, build counts."""
    vocab_all = sorted({t for doc in token_docs for t in doc})
    index = {w: j for j, w in enumerate(vocab_all)}
    docfreq = np.zeros(len(vocab_all), dtype=np.int64)
    for doc in token_docs:
        for w in set(doc):
            docfreq[index[w]] += 1
    keep = docfreq > lower_thresh
    vocab = [w for w, k in zip(vocab_all, keep) if k]
    index = {w: j for j, w in enumerate(vocab)}

    X = np.zeros((len(token_docs), len(vocab)), dtype=np.int64)
    removed_docs = []
    for i, doc in enumerate(token_docs):
        for w in doc:
            j = index.get(w)
            if j is not None:
                X[i, j] += 1
        if X[i].sum() == 0:
            removed_docs.append(i)
    if removed_docs:
        warnings.warn(f"{len(removed_docs)} documents lost all words.")
        keep_rows = np.setdiff1d(np.arange(len(token_docs)), removed_docs)
        X = X[keep_rows]
    return X, vocab, removed_docs


def load_gadarian(stm_data_dir="stm/data"):
    """Load the gadarian study data and the R-fitted reference model."""
    import rdata

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gadarian = rdata.read_rda(f"{stm_data_dir}/gadarian.RData")["gadarian"]
        fit = rdata.read_rda(f"{stm_data_dir}/gadarianFit.RData")["gadarianFit"]
    return gadarian, fit
