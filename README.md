# LC0 Dual-T4 Kaggle Pairing — Private Tailscale Edition

This project keeps its dual-T4 LC0 engine and authenticated HTTP API, but replaces the public Cloudflare Quick Tunnel and GitHub URL pointer with a **private Tailscale tailnet connection**.

The Kaggle server joins your tailnet as an ephemeral node. Your phone joins the same tailnet using the official Tailscale app. The phone then calls the Kaggle node's Tailscale IP. No public IP, Cloudflare tunnel, GitHub pointer, or public endpoint is used for runtime pairing.

## What stays the same

- LC0 v0.32.1, CUDA FP16 DEMUX, both T4 GPUs, and the pinned BT4-it332 network.
- The persistent UCI engine process and the existing /health and /v1/analyze HTTP API.
- The bearer-token check, FEN validation, bounded analysis time, PV and score response, and HTTP 429 behavior for overlapping analyses.
- The two-notebook cache workflow: build the large artifacts once, then attach the saved Notebook Output to each fresh Server runtime.

## What changed

- Removed Cloudflare Quick Tunnel startup, downloads, cache files, URL parsing and publishing.
- Removed the GitHub Contents API write, hooker.txt, and all server-side GitHub secrets/source downloads.
- The Server notebook embeds and verifies the Python service inside its one-cell launcher; it does not fetch server source from GitHub.
- The Builder exports only lc0, BT4-it332.pb.gz, and manifest.json. Older cache outputs that still contain an unused cloudflared file remain compatible; the new server ignores that file.
- Tailscale runs in userspace networking mode, so Kaggle does not need a TUN device. The server attempts an explicit tailnet-only Tailscale Serve HTTP forwarder on port 8765 to http://127.0.0.1:8765 and verifies `tailscale serve status`; if Serve prerequisites are unavailable, it falls back to Tailscale userspace networking's same-port localhost forwarding. It also checks its local `/health` endpoint before declaring the API online.

## Stage 1 — build the cache once

1. Open Kaggle and create a notebook named **LC0 Dual T4 - Cache Builder**.
2. Select **Accelerator: GPU T4 x2** and **Internet: On**.
3. Paste the complete cache_builder_cell.py contents into a code cell and run it.
4. It builds LC0 v0.32.1 with CUDA compute capability 7.5, downloads BT4-it332, verifies the network's exact size and SHA256, and runs the dual-T4 UCI smoke tests.
5. The final line is CACHE BUILD COMPLETE. The output folder is /kaggle/working/lc0-dual-t4-cache and contains exactly three runtime artifacts: lc0, BT4-it332.pb.gz, and manifest.json.
6. Save a Kaggle notebook version with output files enabled. In the Server notebook, attach this saved folder through **Input → Add Input → Notebook Output Files**.

The Builder is a one-time preparation step and clones upstream LC0 source from GitHub. **The running Server/tunnel path does not contact GitHub.** If you already have the saved cache output from the older version, you do not need to rebuild it; the new server accepts the existing cache and ignores the extra Cloudflare binary.

## Stage 2 — prepare the Tailscale tailnet

1. Install the official **Tailscale** app on your Android phone and sign in to your tailnet.
2. Open the Tailscale admin console → **Keys** → **Generate auth key**.
3. Enable **Ephemeral** and **Pre-approved** (the latter appears if device approval is enabled). For repeated Kaggle restarts, you may enable **Reusable**; protect that secret carefully. Otherwise create a new one-off key when needed.
4. Keep the key private. Add it to Kaggle as a Secret named TS_AUTH_KEY.
5. Create a separate random bearer token for the chess API and save it as the Kaggle Secret ENGINE_API_TOKEN. Do not use the Tailscale auth key as the API token.

The server uses --state=mem:, so each Kaggle run is an ephemeral Tailscale node and receives a new Tailscale IP. On graceful shutdown it logs out and removes the node. If Kaggle kills the runtime abruptly, Tailscale automatically removes inactive ephemeral nodes later.

Tailscale auth keys register a device without an interactive browser login; ephemeral keys are intended for short-lived workloads such as containers. See the [official auth-key guide](https://tailscale.com/docs/features/access-control/auth-keys) and [ephemeral node guide](https://tailscale.com/docs/features/ephemeral-nodes).

## Stage 3 — launch the Server notebook

1. Create a notebook named **LC0 Dual T4 - Server**.
2. Set **Accelerator: GPU T4 x2** and **Internet: On**.
3. Attach the saved Builder notebook Output containing lc0, BT4-it332.pb.gz, and manifest.json.
4. Under **Add-ons → Secrets**, enable TS_AUTH_KEY and ENGINE_API_TOKEN.
5. Paste the complete kaggle_final_cell.py contents into one code cell and run it.
6. The cell verifies its embedded server source, installs python-chess only if needed, downloads the official Tailscale static Linux binaries, checks the archive against Tailscale's published SHA256, starts Tailscale in userspace mode and LC0, attempts to configure and verify `tailscale serve --bg --yes --http=8765 http://127.0.0.1:8765`, then checks the local `/health` endpoint. If Serve setup is unavailable, it retains userspace same-port forwarding and logs that fallback.
7. Wait until the cell prints **LC0 DUAL-T4 SERVICE ONLINE — PRIVATE TAILNET ONLY**. Copy the printed **Phone base URL** and use it in the phone client. It will look like http://100.x.y.z:8765; the actual address is assigned at runtime and is not predictable in advance.
8. Keep the notebook cell running while using the engine. Interrupt it when finished; the cell resets its Tailscale Serve configuration, logs out the ephemeral node, and stops LC0.

The Tailscale client version is pinned in lc0_kaggle_server.py to stable v1.104.7. Its static archive is checked against the published .sha256 file before execution.

## Phone connection

The phone must be connected to Tailscale and signed into the **same tailnet** as the Kaggle node. Use the Phone base URL printed by the Kaggle cell:

- Health: GET http://<KAGGLE_TAILSCALE_IP>:8765/health
- Analysis: POST http://<KAGGLE_TAILSCALE_IP>:8765/v1/analyze
- Header: Authorization: Bearer <ENGINE_API_TOKEN>
- JSON body: {"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","movetime_ms":1800}

Example Python request from the phone-side client:

~~~python
import requests

BASE_URL = "http://100.x.y.z:8765"  # replace with the URL printed in Kaggle
ENGINE_API_TOKEN = "your separate API bearer token"

health = requests.get(f"{BASE_URL}/health", timeout=5)
health.raise_for_status()

response = requests.post(
    f"{BASE_URL}/v1/analyze",
    headers={"Authorization": f"Bearer {ENGINE_API_TOKEN}"},
    json={
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "movetime_ms": 1800,
    },
    timeout=15,
)
response.raise_for_status()
print(response.json()["bestmove"])
~~~

Use the same bearer token stored in Kaggle Secrets; never hardcode it in a public repository. If the existing phone app uses HTTPS-only networking, allow this private HTTP endpoint for the Tailscale interface or adapt the app's network-security policy accordingly. Tailscale encrypts traffic between tailnet devices; the API bearer token remains an additional application-level check.

If your tailnet ACL/grants are restrictive, allow the phone device/user to connect to the Kaggle node on **TCP 8765**. A default allow-all tailnet policy permits same-tailnet device communication.

## Connectivity and limitations

- Tailscale tries to establish a direct peer-to-peer encrypted connection. Depending on the NAT/firewall conditions on the phone and Kaggle sides, it may use an encrypted DERP relay fallback; Tailscale chooses the path automatically. Check tailscale status or the admin console to inspect the connection path.
- The Kaggle node's Tailscale IP changes whenever a fresh ephemeral node is created. Always copy the current Phone base URL from the active notebook output.
- Kaggle sessions are finite and can be interrupted. A terminated notebook stops the engine and the API.
- Health is reachable only over the tailnet. POST /v1/analyze additionally requires the bearer token.
- Overlapping searches still return HTTP 429 rather than queueing stale positions.

## API response

The API returns bestmove and bestmove_san, UCI/SAN principal variation, score, depth, seldepth, nodes, NPS and elapsed time. Invalid FEN or time bounds return HTTP 400, missing/incorrect bearer tokens return HTTP 401, overlapping searches return HTTP 429, and engine faults return HTTP 503. Arbitrary UCI commands and shell execution are not exposed.


### Long-running Kaggle session notes

- The Kaggle node explicitly starts Tailscale with `--accept-dns=false` and `--accept-routes=false`; it does not accept tailnet DNS or subnet-route configuration.
- After the ONLINE banner, the foreground cell remains in its watchdog loop and prints an alive heartbeat every 30 seconds. A missing heartbeat means the Kaggle kernel/session was interrupted or terminated, not that the Python program intentionally completed.
- Kaggle controls interactive-session lifetime and GPU availability; a notebook cell cannot override platform session limits. Use the interactive editor session for a persistent foreground service and monitor Kaggle's session status.
