"""Anchor drift collector — cosine distance from anchor to recent outputs."""
from __future__ import annotations

import logging
from typing import List

from bedrock_attest.types import Signal

logger = logging.getLogger(__name__)


class AnchorDriftCollector:
    name = "anchor_drift"
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
        except ImportError:
            logger.warning("sentence-transformers not available; skipping %s", self.name)
            return Signal(name=self.name, value=0.0)

        if not outputs or not anchor_text:
            return Signal(name=self.name, value=0.0)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        anchor_emb = model.encode(anchor_text, convert_to_numpy=True)
        out_embs = model.encode(outputs, convert_to_numpy=True)
        distances = [1.0 - float(util.cos_sim(anchor_emb, e)) for e in out_embs]
        return Signal(name=self.name, value=sum(distances) / len(distances))
