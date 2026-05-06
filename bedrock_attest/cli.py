"""CLI entry point: bedrock init | attest | verify | diff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat,
)

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

BEDROCK_DIR  = Path.home() / ".bedrock-attest"
KEY_PATH     = BEDROCK_DIR / "key.pem"
PUB_PATH     = BEDROCK_DIR / "key.pub"
FP_FILE      = Path("bedrock.fingerprint.json")
TOML_FILE    = Path("bedrock.toml")
PROMPTS_FILE = Path("prompts.json")

_TOML_TEMPLATE = """\
[agent]
name = "my-agent"
system_prompt = "You are a helpful assistant."
model = "claude-haiku-4-5"
provider_url = "https://api.anthropic.com"
tolerance_default = 0.05
"""

_PROMPTS_TEMPLATE = [
    "Explain what you can help me with.",
    "Write a short Python function to reverse a string.",
    "What are best practices for error handling?",
]


def _icon(verdict: str) -> str:
    return {
        "pass":   f"{GREEN}✓{RESET}",
        "warn":   f"{YELLOW}⚠{RESET}",
        "breach": f"{RED}✗{RESET}",
    }.get(verdict, "?")


def _exit_code(overall: str) -> int:
    return {"pass": 0, "warn": 1, "breach": 2}.get(overall, 3)


# ── init ──────────────────────────────────────────────────────────────────────

def cmd_init() -> int:
    try:
        BEDROCK_DIR.mkdir(parents=True, exist_ok=True)

        if KEY_PATH.exists():
            print(f"{YELLOW}Key already exists at {KEY_PATH} — skipping key generation.{RESET}")
        else:
            priv = Ed25519PrivateKey.generate()
            KEY_PATH.write_bytes(priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
            PUB_PATH.write_bytes(priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
            print(f"{GREEN}✓{RESET} Key generated → {KEY_PATH}")

        if TOML_FILE.exists():
            print(f"{YELLOW}bedrock.toml already exists — skipping.{RESET}")
        else:
            TOML_FILE.write_text(_TOML_TEMPLATE, encoding="utf-8")
            print(f"{GREEN}✓{RESET} bedrock.toml scaffolded")

        if PROMPTS_FILE.exists():
            print(f"{YELLOW}prompts.json already exists — skipping.{RESET}")
        else:
            PROMPTS_FILE.write_text(json.dumps(_PROMPTS_TEMPLATE, indent=2), encoding="utf-8")
            print(f"{GREEN}✓{RESET} prompts.json scaffolded")

        print("\nNext: edit bedrock.toml, fill in prompts.json, then run: bedrock attest")
        return 0

    except Exception as exc:
        print(f"{RED}Error during init: {exc}{RESET}", file=sys.stderr)
        return 3


# ── attest ────────────────────────────────────────────────────────────────────

def cmd_attest() -> int:
    from bedrock_attest.attest import attest as _attest
    from bedrock_attest.config import BedrockConfig

    try:
        config = BedrockConfig.from_toml(TOML_FILE)
        inputs: list = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
        sign_key = str(KEY_PATH) if KEY_PATH.exists() else None

        print(f"Running {len(inputs)} test inputs against {config.model} …")
        fp = _attest(config, inputs, config.model, sign_key=sign_key)

        FP_FILE.write_text(json.dumps(fp.to_dict(), indent=2), encoding="utf-8")
        sig_note = f" + {KEY_PATH}.sig" if sign_key else ""
        print(f"{GREEN}✓{RESET} {len(fp.signals)} signals attested → {FP_FILE}{sig_note}")
        return 0

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── verify ────────────────────────────────────────────────────────────────────

def cmd_verify() -> int:
    from bedrock_attest.config import BedrockConfig
    from bedrock_attest.verify import verify as _verify

    try:
        config = BedrockConfig.from_toml(TOML_FILE)
        inputs: list = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))

        print(f"Re-attesting {len(inputs)} inputs against {config.model} …")
        report = _verify(str(FP_FILE), config, config.model, inputs)

        for name, verdict, detail in report.per_signal:
            print(f"  {_icon(verdict)} {name:<25} {detail}")

        overall_label = report.overall.upper()
        color = {0: GREEN, 1: YELLOW, 2: RED}.get(_exit_code(report.overall), RED)
        print(f"\n{color}Overall: {overall_label}{RESET}")
        return _exit_code(report.overall)

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── diff ──────────────────────────────────────────────────────────────────────

def cmd_diff(path_a: str, path_b: str) -> int:
    from bedrock_attest.types import Fingerprint

    try:
        fp_a = Fingerprint.from_dict(json.loads(Path(path_a).read_text(encoding="utf-8")))
        fp_b = Fingerprint.from_dict(json.loads(Path(path_b).read_text(encoding="utf-8")))

        sigs_b = {s.name: s for s in fp_b.signals}
        worst = "pass"

        print(f"{'Signal':<25} {'A':>10} {'B':>10} {'Δ':>10}  Verdict")
        print("-" * 65)

        for sig in fp_a.signals:
            if sig.name not in sigs_b:
                verdict = "warn"
                line = f"{sig.name:<25} {sig.value:>10.4f} {'—':>10} {'—':>10}  missing in B"
            else:
                b_val = sigs_b[sig.name].value
                delta = abs(sig.value - b_val)
                if delta <= sig.tolerance * 0.5:
                    verdict = "pass"
                elif delta <= sig.tolerance:
                    verdict = "warn"
                else:
                    verdict = "breach"
                line = f"{sig.name:<25} {sig.value:>10.4f} {b_val:>10.4f} {delta:>10.4f}"

            if _exit_code(verdict) > _exit_code(worst):
                worst = verdict
            print(f"  {_icon(verdict)} {line}")

        color = {0: GREEN, 1: YELLOW, 2: RED}.get(_exit_code(worst), RED)
        print(f"\n{color}Overall: {worst.upper()}{RESET}")
        return _exit_code(worst)

    except Exception as exc:
        print(f"{RED}Error: {exc}{RESET}", file=sys.stderr)
        return 3


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bedrock",
        description="Behavioral attestation for AI agents.",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.add_parser("init",   help="Generate key + scaffold bedrock.toml and prompts.json")
    sub.add_parser("attest", help="Run test suite, save bedrock.fingerprint.json")
    sub.add_parser("verify", help="Re-attest and compare against saved fingerprint")
    diff_p = sub.add_parser("diff", help="Compare two fingerprint files without re-attesting")
    diff_p.add_argument("a", metavar="FINGERPRINT_A")
    diff_p.add_argument("b", metavar="FINGERPRINT_B")

    args = parser.parse_args()

    if args.cmd == "init":
        sys.exit(cmd_init())
    elif args.cmd == "attest":
        sys.exit(cmd_attest())
    elif args.cmd == "verify":
        sys.exit(cmd_verify())
    elif args.cmd == "diff":
        sys.exit(cmd_diff(args.a, args.b))
    else:
        parser.print_help()
        sys.exit(3)


if __name__ == "__main__":
    main()
