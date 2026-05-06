"""Embedding profile collector (requires [deep] extra)."""
from __future__ import annotations

import logging
from typing import List

from bedrock_attest.types import Signal

logger = logging.getLogger(__name__)


class EmbeddingProfileCollector:
    name = "embedding_profile"
    needs_extras: tuple = ("deep",)

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning("sentence-transformers/numpy not available; skipping %s", self.name)
            return Signal(name=self.name, value=0.0)

        if not outputs:
            return Signal(name=self.name, value=0.0)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(outputs, convert_to_numpy=True)
        centroid = embeddings.mean(axis=0)
        sims = [float(util.cos_sim(centroid, e)) for e in embeddings]
        mean_sim = sum(sims) / len(sims)
        variance = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
        return Signal(name=self.name, value=mean_sim, p50=variance)
