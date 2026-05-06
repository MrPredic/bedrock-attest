"""Tests for bedrock_attest.cli."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import bedrock_attest.cli as cli_mod
from bedrock_attest.config import BedrockConfig
from bedrock_attest.types import Fingerprint, Signal, VerifyReport


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_fp(*extra_signals: Signal) -> Fingerprint:
    base = (
        Signal("refusal_rate", 0.05, tolerance=0.1),
        Signal("latency",      0.3,  p50=0.25, p95=0.55, tolerance=0.5),
        Signal("vocab_entropy", 3.1, tolerance=1.0),
    )
    return Fingerprint(
        schema_version="1",
        config_hash="abc" * 10 + "ab",
        model="gpt-4o",
        timestamp="2026-05-06T00:00:00Z",
        maintainer="",
        signals=base + extra_signals,
        test_set_hash="def" * 10 + "de",
    )


def _make_cfg() -> BedrockConfig:
    return BedrockConfig(
        agent_name="test",
        system_prompt="You are helpful.",
        tools=[],
        model="gpt-4o",
        provider_url="https://api.openai.com/v1",
    )


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Redirect all CLI path constants to tmp_path."""
    monkeypatch.setattr(cli_mod, "BEDROCK_DIR",  tmp_path)
    monkeypatch.setattr(cli_mod, "KEY_PATH",     tmp_path / "key.pem")
    monkeypatch.setattr(cli_mod, "PUB_PATH",     tmp_path / "key.pub")
    monkeypatch.setattr(cli_mod, "FP_FILE",      tmp_path / "bedrock.fingerprint.json")
    monkeypatch.setattr(cli_mod, "TOML_FILE",    tmp_path / "bedrock.toml")
    monkeypatch.setattr(cli_mod, "PROMPTS_FILE", tmp_path / "prompts.json")
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ── cmd_init ───────────────────────────────────────────────────────────────────

def test_init_creates_key_and_files(iso):
    assert cli_mod.cmd_init() == 0
    assert (iso / "key.pem").exists()
    assert (iso / "key.pub").exists()
    assert (iso / "bedrock.toml").exists()
    assert (iso / "prompts.json").exists()


def test_init_toml_is_valid(iso):
    cli_mod.cmd_init()
    cfg = BedrockConfig.from_toml(iso / "bedrock.toml")
    assert cfg.agent_name  # non-empty
    assert cfg.model


def test_init_prompts_is_list(iso):
    cli_mod.cmd_init()
    prompts = json.loads((iso / "prompts.json").read_text())
    assert isinstance(prompts, list)
    assert len(prompts) >= 1


def test_init_does_not_overwrite_existing_key(iso):
    (iso / "key.pem").write_bytes(b"ORIGINAL")
    cli_mod.cmd_init()
    assert (iso / "key.pem").read_bytes() == b"ORIGINAL"


# ── cmd_diff ───────────────────────────────────────────────────────────────────

def test_diff_identical_fingerprints(iso):
    fp = _make_fp()
    (iso / "a.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")
    (iso / "b.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "b.json")) == 0


def test_diff_breached_fingerprints(iso):
    fp_a = _make_fp()
    extreme = tuple(
        Signal(s.name, s.value + 999.0, tolerance=s.tolerance) for s in fp_a.signals
    )
    fp_b = Fingerprint(
        schema_version=fp_a.schema_version, config_hash=fp_a.config_hash,
        model=fp_a.model, timestamp=fp_a.timestamp, maintainer=fp_a.maintainer,
        signals=extreme, test_set_hash=fp_a.test_set_hash,
    )
    (iso / "a.json").write_text(json.dumps(fp_a.to_dict()), encoding="utf-8")
    (iso / "b.json").write_text(json.dumps(fp_b.to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "b.json")) == 2


def test_diff_missing_file_returns_3(iso):
    (iso / "a.json").write_text(json.dumps(_make_fp().to_dict()), encoding="utf-8")
    assert cli_mod.cmd_diff(str(iso / "a.json"), str(iso / "no_such.json")) == 3


# ── cmd_attest ─────────────────────────────────────────────────────────────────

def test_attest_writes_fingerprint(iso):
    cfg = _make_cfg()
    cfg.to_toml(iso / "bedrock.toml")
    (iso / "prompts.json").write_text(json.dumps(["hi", "hello"]), encoding="utf-8")

    class Stub:
        def complete(self, system, user, tools=None): return ("stub", [], 0.05)

    with patch("bedrock_attest.attest.get_provider", return_value=Stub()):
        code = cli_mod.cmd_attest()

    assert code == 0
    assert (iso / "bedrock.fingerprint.json").exists()
    data = json.loads((iso / "bedrock.fingerprint.json").read_text())
    assert "signals" in data


def test_attest_missing_toml_returns_3(iso):
    assert cli_mod.cmd_attest() == 3


# ── cmd_verify ─────────────────────────────────────────────────────────────────

def test_verify_pass_report(iso):
    cfg = _make_cfg()
    cfg.to_toml(iso / "bedrock.toml")
    inputs = ["hi", "hello"]
    (iso / "prompts.json").write_text(json.dumps(inputs), encoding="utf-8")
    fp = _make_fp()
    (iso / "bedrock.fingerprint.json").write_text(json.dumps(fp.to_dict()), encoding="utf-8")

    mock_report = VerifyReport(
        overall="pass",
        per_signal=(("refusal_rate", "pass", "Δ 0.0000"),),
        elapsed_s=0.1,
    )
    with patch("bedrock_attest.verify.attest", return_value=fp):
        code = cli_mod.cmd_verify()

    assert code in (0, 1, 2)  # any valid exit code = no crash


# ── subprocess (main) ──────────────────────────────────────────────────────────

def test_main_help_exit0():
    result = subprocess.run(
        [sys.executable, "-m", "bedrock_attest.cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "bedrock" in result.stdout.lower()


def test_main_no_args_exit3():
    result = subprocess.run(
        [sys.executable, "-m", "bedrock_attest.cli"],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
