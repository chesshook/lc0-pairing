# LC0 Dual-T4 Kaggle Pairing

This repository contains a Kaggle-hosted LC0 service and a small GitHub endpoint-discovery pointer for the later ChessHook Android integration.

## Validated engine configuration

The supplied validation log reports 24/24 checks passed, two Tesla T4 GPUs participating, 12 legal requests through one persistent UCI process, and a 2.510x dual-T4/single-T4 throughput ratio in that test run. This is a measured smoke-test result, not a universal speed guarantee.

- LC0 v0.32.1
- BT4-it332 network, 382,645,315 bytes
- BT4 SHA256: `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`
- `--backend=demux`
- `--backend-opts=backend=cuda-fp16,(gpu=0),(gpu=1)`
- 4 CPU threads, minibatch 256, NN cache 1,000,000

## Repository contents

- `lc0_kaggle_pairing_setup.zip` — complete importable setup bundle (source, notebook, cache exporter, final bootstrap cell, requirements, and detailed instructions).
- `hooker.txt` — public discovery pointer only. Starts as `OFFLINE`; the Kaggle server replaces it with the active Quick Tunnel URL.

The GitHub repository stores no neural-network weights, engine executable, or API secret. The large immutable artifacts belong in Kaggle Notebook Output Files.

## One-time setup

1. Create a fine-grained GitHub token restricted to this repository with **Contents: Read and write** and Metadata: Read-only. Store it only in Kaggle Secrets as `GITHUB_TOKEN`.
2. In the existing validated Kaggle notebook, download this repository's setup ZIP and extract it. Run `lc0_kaggle_setup/cache_export.py` against the existing build. Do not rebuild LC0. The exporter checks the exact BT4 size and SHA256, copies the compiled binary, and downloads `cloudflared` once.
3. Save a Kaggle notebook version containing the `lc0-dual-t4-cache` output folder. Attach that notebook's Output Files as an Input to the new server notebook.
4. Configure Kaggle **GPU T4 x2**, Internet **On**, and the five Secrets below.
5. Import `lc0_dual_t4_server.ipynb` from the ZIP and execute its single code cell. Alternatively, paste `kaggle_final_cell.py` into a blank Kaggle notebook.

### Kaggle Secrets

| Secret | Value |
|---|---|
| `GITHUB_TOKEN` | Fine-grained token described above |
| `GITHUB_REPO_OWNER` | `chesshook` |
| `GITHUB_REPO_NAME` | `lc0-pairing` |
| `GITHUB_BRANCH` | `main` |
| `ENGINE_API_TOKEN` | Independently generated random bearer token; never commit it |

Generate `ENGINE_API_TOKEN` locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Keep it in Kaggle Secrets and your password manager. Do not paste it into chat, source code, or GitHub.

## Runtime behavior

At each fresh Kaggle runtime, the single cell downloads only this small source bundle, finds and verifies the mounted notebook-output cache under `/kaggle/input`, copies the small executable tools into session scratch (to restore execute permissions), and reuses the 383 MB weights file directly from the read-only mounted input. It does not download weights or compile LC0.

LC0 must still initialize CUDA and load BT4 into GPU memory after a full runtime restart. That cannot be persisted across Kaggle sessions.

The service starts one long-lived UCI process, warms it with a legal-move smoke test, exposes the local API on port 8765, starts a Cloudflare Quick Tunnel, and publishes the new base URL into `hooker.txt`. Interrupting the notebook cell requests graceful shutdown and attempts to publish `OFFLINE`.

## API contract

Discovery URL:

`https://raw.githubusercontent.com/chesshook/lc0-pairing/main/hooker.txt`

- `GET /health` — public readiness/version/GPU report.
- `POST /v1/analyze` — authenticated FEN analysis.
- Header: `Authorization: Bearer <ENGINE_API_TOKEN>`.
- JSON request: `{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","movetime_ms":1800}`.
- Response includes UCI and SAN best move, PV, score, depth, nodes, NPS, and elapsed time.

The endpoint accepts bounded FEN/time inputs only; it does not expose arbitrary UCI commands or shell execution. Analysis requests are serialized; overlapping calls receive HTTP 429 rather than queueing stale positions.

## Important limitations

Kaggle GPU sessions are finite and may be interrupted; this is a development/CTF backend, not an always-on server. Cloudflare Quick Tunnels are testing infrastructure with changing hostnames and no uptime guarantee. The phone must read `hooker.txt`, validate `/health`, and handle `OFFLINE`, stale URLs, timeouts, and 429 responses.
