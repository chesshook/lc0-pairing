# LC0 Dual-T4 Kaggle Pairing

This repository prepares a Kaggle-hosted LC0 service for the later ChessHook Android integration. GitHub stores only the public discovery pointer and small Python source files. It does **not** store the neural-network weights or compiled engine binary.

## Validated engine configuration

The supplied validation log reports 24/24 checks passed, both Tesla T4 GPUs participating, 12 consecutive legal-position requests through one persistent UCI process, and a 2.510x dual-T4/single-T4 throughput ratio in that test run. This is a benchmark observation, not a universal speed guarantee.

- LC0 v0.32.1
- BT4-it332 network, 382,645,315 bytes
- BT4 SHA256: `e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618`
- `--backend=demux`
- `--backend-opts=backend=cuda-fp16,(gpu=0),(gpu=1)`
- 4 CPU threads, minibatch 256, NN cache 1,000,000

## Files

- `cache_export.py` — one-time exporter for the already-compiled LC0 executable, BT4 network and `cloudflared` binary.
- `lc0_kaggle_server.py` — persistent UCI/HTTP server. It validates the attached cache, starts one LC0 process, serves a bounded authenticated JSON analysis API, runs a Cloudflare Quick Tunnel, and updates `hooker.txt`.
- `kaggle_final_cell.py` — the single Kaggle cell to copy into a new notebook.
- `hooker.txt` — public discovery pointer. Initially `OFFLINE`; the service replaces it with the active Quick Tunnel URL.

## One-time cache export (do this before the original validated Kaggle runtime disappears)

The earlier validation notes identify the existing artifacts under:

```
/kaggle/working/lc0_dual_t4_lab/lc0/build/release/lc0
/kaggle/working/lc0_dual_t4_lab/BT4-it332.pb.gz
```

In that same Kaggle session, execute:

```python
!wget -q https://raw.githubusercontent.com/chesshook/lc0-pairing/main/cache_export.py \
    -O /kaggle/working/cache_export.py
!python /kaggle/working/cache_export.py
```

The exporter verifies the BT4 size and SHA256, copies the existing LC0 executable and network into `/kaggle/working/lc0-dual-t4-cache`, and downloads the small Cloudflare tunnel client once. It does **not** compile LC0.

Confirm `CACHE EXPORT COMPLETE`, then save a new version of that notebook so `lc0-dual-t4-cache` appears in its Output Files. Attach those Output Files as an Input to the new server notebook. This is the key persistence step: the later runtimes reuse these output artifacts instead of redownloading weights or rebuilding.

## Kaggle server notebook

1. Create/import a Kaggle notebook.
2. Set Accelerator to **GPU T4 x2** and Internet to **On**.
3. Add Input -> Notebook Output Files -> choose the saved cache-export notebook.
4. Add and enable the Kaggle Secrets below.
5. Copy the complete source of `kaggle_final_cell.py` from this repository into one code cell and run it.

### Kaggle Secrets

| Secret | Value |
|---|---|
| `GITHUB_TOKEN` | Fine-grained GitHub token restricted to this repository, Contents: Read and write; Metadata: Read-only |
| `GITHUB_REPO_OWNER` | `chesshook` |
| `GITHUB_REPO_NAME` | `lc0-pairing` |
| `GITHUB_BRANCH` | `main` |
| `ENGINE_API_TOKEN` | A separate random bearer token; keep it private |

Generate `ENGINE_API_TOKEN` locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Store it in Kaggle Secrets and your password manager. Never commit it or paste it into the chat.

The GitHub token must be fine-grained and limited to this repository. The server uses it solely to create/update the root `hooker.txt` file. It retrieves the file's current SHA before each update.

## API contract

Discovery URL:

`https://raw.githubusercontent.com/chesshook/lc0-pairing/main/hooker.txt`

- `GET /health` — public readiness, LC0 version and GPU names.
- `POST /v1/analyze` — authenticated FEN analysis.
- Header: `Authorization: Bearer <ENGINE_API_TOKEN>`.
- JSON request: `{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","movetime_ms":1800}`.

The response includes UCI/SAN best move, PV, score, depth, nodes, NPS and elapsed time. The service accepts bounded FEN/time input and does not expose arbitrary UCI commands or shell execution. Overlapping analyses receive HTTP 429 rather than building a stale request queue.

## Restart behavior and limitations

| Component | Stored/handled how |
|---|---|
| LC0 executable, BT4, cloudflared | Saved Kaggle Notebook Output Files attached as Input |
| Python server source | Small file fetched from this public repository |
| GitHub PAT and engine API token | Kaggle Secrets only |
| LC0 process and GPU network memory | Reinitialized after each runtime restart |
| Quick Tunnel hostname | Regenerated and republished to `hooker.txt` each start |

A fresh Kaggle runtime downloads only the small Python server source. It copies the cached executable tools into writable session storage, uses the BT4 network directly from the read-only mounted output, and does not rebuild LC0 or download the network again. LC0 must still load BT4 into GPU memory after a restart.

Kaggle GPU sessions are finite and may be interrupted. Cloudflare Quick Tunnels are testing infrastructure: hostnames change and there is no uptime guarantee. The phone must treat `hooker.txt` as a discovery hint, validate `/health`, and handle `OFFLINE`, stale endpoints, timeouts and HTTP 429.
