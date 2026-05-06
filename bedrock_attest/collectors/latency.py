"""Latency collector (P50, P95, mean)."""
from __future__ import annotations

from typing import List, Optional

from bedrock_attest.types import Signal


class LatencyCollector:
    name = "latency"
    needs_extras: tuple = ()

    def __init__(self) -> None:
        self._latencies: List[float] = []

    def set_latencies(self, latencies: List[float]) -> None:
        self._latencies = list(latencies)

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        if not self._latencies:
            return Signal(name=self.name, value=0.0)
        s = sorted(self._latencies)
        n = len(s)
        p50 = s[n // 2]
        p95 = s[int(0.95 * n)]
        mean = sum(s) / n
        return Signal(name=self.name, value=mean, p50=p50, p95=p95)
