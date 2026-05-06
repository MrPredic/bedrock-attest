"""Signal Collector protocol and registry."""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from bedrock_attest.types import Signal


@runtime_checkable
class Collector(Protocol):
    name: str
    needs_extras: tuple

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal: ...
