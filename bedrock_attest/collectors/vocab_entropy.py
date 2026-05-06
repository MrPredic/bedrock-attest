"""Vocabulary Shannon-entropy collector."""
from __future__ import annotations

import math
from collections import Counter
from typing import List

from bedrock_attest.types import Signal


class VocabEntropyCollector:
    name = "vocab_entropy"
    needs_extras: tuple = ()

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        tokens = " ".join(outputs).lower().split()
        if not tokens:
            return Signal(name=self.name, value=0.0)
        counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        return Signal(name=self.name, value=entropy)
