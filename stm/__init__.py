"""pystm: Python implementation of the Structural Topic Model.

A port of the R ``stm`` package (Roberts, Stewart & Tingley) with an API
modeled on scikit-learn's ``LatentDirichletAllocation``.
"""

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
__version__ = "0.2.0"
