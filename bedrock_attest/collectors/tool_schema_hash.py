"""Tool schema hash collector."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

from bedrock_attest.types import Signal


class ToolSchemaHashCollector:
    name = "tool_schema_hash"
    needs_extras: tuple = ()

    def __init__(self, tools: List[Dict[str, Any]]) -> None:
        canonical = json.dumps(tools, sort_keys=True, ensure_ascii=False)
        self._hash = hashlib.sha256(canonical.encode()).hexdigest()
        self._tool_count = len(tools)

    @property
    def schema_hash_str(self) -> str:
        return self._hash

    def collect(
        self,
        outputs: List[str],
        inputs: List[str],
        anchor_text: str,
        tools_called: List[List[str]],
    ) -> Signal:
        numeric = float(int(self._hash[:8], 16)) % 1e9
        return Signal(
            name=self.name,
            value=numeric,
            distribution={"hash_prefix": float(int(self._hash[:8], 16)), "tool_count": float(self._tool_count)},
        )
