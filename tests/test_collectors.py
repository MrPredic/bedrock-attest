"""Tests for bedrock_attest.collectors.*"""
import sys
from unittest.mock import MagicMock

import pytest

from bedrock_attest.collectors import Collector
from bedrock_attest.collectors.anchor_drift import AnchorDriftCollector
from bedrock_attest.collectors.embedding_profile import EmbeddingProfileCollector
from bedrock_attest.collectors.latency import LatencyCollector
from bedrock_attest.collectors.refusal import RefusalCollector
from bedrock_attest.collectors.tool_distribution import ToolDistributionCollector
from bedrock_attest.collectors.tool_schema_hash import ToolSchemaHashCollector
from bedrock_attest.collectors.vocab_entropy import VocabEntropyCollector
from bedrock_attest.types import Signal

# helpers
_CALL = dict(inputs=["What is 2+2?"], anchor_text="You are a helpful assistant.", tools_called=[])


def _collect(collector, outputs, tools_called=None):
    tc = tools_called if tools_called is not None else [[] for _ in outputs]
    return collector.collect(
        outputs=outputs,
        inputs=_CALL["inputs"],
        anchor_text=_CALL["anchor_text"],
        tools_called=tc,
    )


# --- Protocol ---

def test_collector_protocol_satisfied():
    assert isinstance(RefusalCollector(), Collector)
    assert isinstance(LatencyCollector(), Collector)
    assert isinstance(VocabEntropyCollector(), Collector)
    assert isinstance(ToolDistributionCollector(), Collector)
    assert isinstance(ToolSchemaHashCollector([]), Collector)


# --- RefusalCollector ---

def test_refusal_no_refusal():
    s = _collect(RefusalCollector(), ["The answer is 42.", "Here you go."])
    assert isinstance(s, Signal)
    assert s.name == "refusal_rate"
    assert s.value == 0.0


def test_refusal_full_refusal():
    s = _collect(RefusalCollector(), ["I cannot help with that.", "I'm not able to do this."])
    assert s.value == 1.0


def test_refusal_partial():
    s = _collect(RefusalCollector(), ["Sure!", "I cannot do that."])
    assert s.value == pytest.approx(0.5)


def test_refusal_empty_outputs():
    s = _collect(RefusalCollector(), [])
    assert s.value == 0.0


# --- LatencyCollector ---

def test_latency_values():
    c = LatencyCollector()
    c.set_latencies([0.1, 0.2, 0.3, 0.4, 0.5])
    s = _collect(c, ["ok"])
    assert s.name == "latency"
    assert s.value == pytest.approx(0.3)
    assert s.p50 is not None
    assert s.p95 is not None


def test_latency_empty():
    s = _collect(LatencyCollector(), ["ok"])
    assert s.value == 0.0
    assert s.p50 is None


def test_latency_single():
    c = LatencyCollector()
    c.set_latencies([1.5])
    s = _collect(c, ["ok"])
    assert s.value == pytest.approx(1.5)


# --- VocabEntropyCollector ---

def test_entropy_single_token():
    s = _collect(VocabEntropyCollector(), ["word word word"])
    assert s.name == "vocab_entropy"
    assert s.value == pytest.approx(0.0)


def test_entropy_diverse():
    s = _collect(VocabEntropyCollector(), ["apple banana cherry date elderberry"])
    assert s.value > 0.0


def test_entropy_empty():
    s = _collect(VocabEntropyCollector(), [])
    assert s.value == 0.0


# --- ToolDistributionCollector ---

def test_tool_distribution_no_calls():
    s = _collect(ToolDistributionCollector(), ["ok"], tools_called=[[]])
    assert s.name == "tool_distribution"
    assert s.value == 0.0


def test_tool_distribution_single_tool():
    s = _collect(ToolDistributionCollector(), ["ok", "ok"], tools_called=[["search"], ["search"]])
    assert s.value == 1.0
    assert s.distribution == {"search": 1.0}


def test_tool_distribution_multiple():
    s = _collect(ToolDistributionCollector(), ["ok", "ok"],
                 tools_called=[["search", "read"], ["search"]])
    assert s.value == 2.0
    assert abs(s.distribution["search"] - 2 / 3) < 0.01


# --- ToolSchemaHashCollector ---

def test_schema_hash_deterministic():
    tools = [{"name": "read_file", "description": "reads"}]
    c1 = ToolSchemaHashCollector(tools)
    c2 = ToolSchemaHashCollector(tools)
    assert c1.schema_hash_str == c2.schema_hash_str


def test_schema_hash_changes_with_schema():
    c1 = ToolSchemaHashCollector([{"name": "a"}])
    c2 = ToolSchemaHashCollector([{"name": "b"}])
    assert c1.schema_hash_str != c2.schema_hash_str


def test_schema_hash_collect_returns_signal():
    c = ToolSchemaHashCollector([{"name": "read_file"}])
    s = _collect(c, ["ok"])
    assert s.name == "tool_schema_hash"
    assert s.value >= 0.0
    assert s.distribution is not None


def test_schema_hash_empty_tools():
    c = ToolSchemaHashCollector([])
    s = _collect(c, ["ok"])
    assert isinstance(s, Signal)


# --- EmbeddingProfileCollector + AnchorDriftCollector (graceful skip) ---

def test_embedding_profile_skip_without_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(EmbeddingProfileCollector(), ["hello world"])
    assert s.name == "embedding_profile"
    assert s.value == 0.0


def test_anchor_drift_skip_without_extras(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(AnchorDriftCollector(), ["hello"])
    assert s.name == "anchor_drift"
    assert s.value == 0.0


def test_embedding_profile_needs_extras():
    assert "deep" in EmbeddingProfileCollector.needs_extras


def test_anchor_drift_needs_extras():
    assert "deep" in AnchorDriftCollector.needs_extras


# --- embedding happy-path via mocked sentence_transformers ---

def _make_st_mock():
    import numpy as np
    mock_st = MagicMock()
    mock_model = MagicMock()
    n = 3
    embs = np.random.rand(n, 32).astype("float32")
    mock_model.encode.return_value = embs
    mock_model.cos_sim.side_effect = lambda a, b: MagicMock(__float__=lambda self: 0.9)
    mock_st.SentenceTransformer.return_value = mock_model
    # util.cos_sim returns float-like
    mock_st.util.cos_sim.return_value = 0.9
    return mock_st


def test_embedding_profile_happy_path(monkeypatch):
    import numpy as np
    mock_st = _make_st_mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    monkeypatch.setitem(sys.modules, "numpy", np)
    outputs = ["The cat sat.", "A dog ran.", "Birds flew."]
    s = _collect(EmbeddingProfileCollector(), outputs)
    assert s.name == "embedding_profile"
    assert s.value >= 0.0


def test_anchor_drift_happy_path(monkeypatch):
    mock_st = _make_st_mock()
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    outputs = ["The cat sat.", "A dog ran."]
    s = _collect(AnchorDriftCollector(), outputs)
    assert s.name == "anchor_drift"
    assert s.value >= 0.0


def test_anchor_drift_empty_outputs(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    s = _collect(AnchorDriftCollector(), [])
    assert s.value == 0.0
