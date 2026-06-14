"""pystm: Python implementation of the Structural Topic Model.

A port of the R ``stm`` package (Roberts, Stewart & Tingley) with an API
modeled on scikit-learn's ``LatentDirichletAllocation``.
"""

from importlib.metadata import PackageNotFoundError, version

from .diagnostics import (
    TopicCorrelations,
    check_residuals,
    exclusivity,
    semantic_coherence,
    topic_corr,
)
from .effects import EstimatedEffects, estimate_effect
from .model_selection import eval_heldout, make_heldout, search_k
from .stm import StructuralTopicModel

__all__ = [
    "StructuralTopicModel",
    "estimate_effect",
    "EstimatedEffects",
    "search_k",
    "make_heldout",
    "eval_heldout",
    "topic_corr",
    "TopicCorrelations",
    "semantic_coherence",
    "exclusivity",
    "check_residuals",
]
try:
    # single source of truth: the version declared in pyproject.toml
    __version__ = version("structural-topic-model")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "unknown"
