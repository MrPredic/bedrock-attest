# bedrock-attest

> Sigstore for AI agents. Sign your agent's behavior today. Verify it was unchanged tomorrow.

**Status:** v0.1.0 — work in progress

## What it does

`bedrock-attest` creates a cryptographically signed behavioral fingerprint of your AI agent. When your model provider silently updates their model, you'll know.

```bash
pip install bedrock-attest
bedrock init      # scaffold config + generate signing key
bedrock attest    # capture behavioral fingerprint
bedrock verify    # detect drift since last attestation
```

Full documentation coming with v0.1.0 release.
