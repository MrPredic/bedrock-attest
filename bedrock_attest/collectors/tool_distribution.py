"""Tool call distribution collector."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from bedrock_attest.types import Signal


class ToolDistributionCollector:
    name = "tool_distribution"
    needs_extras: tuple = ()

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        flat = [tool for call_list in tools_called for tool in call_list]
        if not flat:
            return Signal(name=self.name, value=0.0, distribution={})
        counts = Counter(flat)
        total = len(flat)
        distribution: Dict[str, float] = {t: c / total for t, c in counts.items()}
        return Signal(name=self.name, value=float(len(counts)), distribution=distribution)
