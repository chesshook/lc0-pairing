#!/usr/bin/env python3
"""Restart-safe LC0 dual-T4 HTTP service for a Kaggle Notebook.

- Reuses the cached LC0 binary and verified BT4 network from Notebook Output.
- Starts one persistent LC0 UCI process with the CUDA FP16 DEMUX backend.
- Exposes an authenticated JSON API on 127.0.0.1 only.
- Joins the user's tailnet using Tailscale userspace networking.
- Phone clients connect directly to the private Tailscale IPv4 address.
- Never downloads weights or compiles LC0 at server startup.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_WEIGHT_SIZE = 382_645_315
EXPECTED_WEIGHT_SHA256 = (
    "e6ada9d6c4a769bfab3aa0848d82caeb"
    "809aa45f83e6c605fc58a31d21bdd618"
)
PORT = int(os.environ.get("LC0_HTTP_PORT", "8765"))
THREADS = int(os.environ.get("LC0_THREADS", "4"))
MINIBATCH = int(os.environ.get("LC0_MINIBATCH", "256"))
NNCACHE = int(os.environ.get("LC0_NNCACHE", "1000000"))
DEFAULT_MOVETIME_MS = int(os.environ.get("LC0_DEFAULT_MOVETIME_MS", "1800"))
MAX_MOVETIME_MS = int(os.environ.get("LC0_MAX_MOVETIME_MS", "10000"))
TAILSCALE_VERSION = os.environ.get("LC0_TAILSCALE_VERSION", "1.104.7")
TAILSCALE_HOSTNAME = os.environ.get("LC0_TAILSCALE_HOSTNAME", "lc0-dual-t4-kaggle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_secret(client: Any, name: str, required: bool = True, default: str = "") -> str:
    try:
        value = client.get_secret(name)
    except Exception:
        value = None
    value = str(value).strip() if value is not None else ""
    if not value:
        value = default
    if required and not value:
        raise RuntimeError(
            f"Missing Kaggle Secret: {name}. Add it under Add-ons -> Secrets "
            "and enable it for this notebook."
        )
    return value


def find_cache() -> tuple[Path, dict[str, Any]]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise RuntimeError("Attach the saved LC0 cache notebook output as a Kaggle Input first.")

    for manifest_path in input_root.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("weights_sha256") == EXPECTED_WEIGHT_SHA256
                and manifest.get("weights_size") == EXPECTED_WEIGHT_SIZE
            ):
                root = manifest_path.parent
                lc0_path = root / manifest.get("lc0_file", "lc0")
                weights_path = root / manifest.get("weights_file", "BT4-it332.pb.gz")
                if lc0_path.is_file() and weights_path.is_file():
                    return root, manifest
        except Exception:
            continue

    raise RuntimeError(
        "No matching LC0 cache found in /kaggle/input. Attach the saved output "
        "containing lc0, BT4-it332.pb.gz and manifest.json. Existing cache outputs "
        "that also contain an unused cloudflared file are accepted; it is ignored."
    )


def verify_cache(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    lc0 = root / manifest.get("lc0_file", "lc0")
    weights = root / manifest.get("weights_file", "BT4-it332.pb.gz")

    if weights.stat().st_size != EXPECTED_WEIGHT_SIZE:
        raise RuntimeError("Mounted BT4 network has the wrong file size.")
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Mounted BT4 network failed SHA256 verification.")
    expected_lc0_hash = manifest.get("lc0_sha256")
    if expected_lc0_hash and sha256_file(lc0) != expected_lc0_hash:
        raise RuntimeError("Mounted LC0 executable failed SHA256 verification.")

    # Notebook Output Files are read-only. Stage only the small executable;
    # keep the 383 MB weights mounted read-only and avoid copying them.
    runtime_bin = Path("/kaggle/working/lc0-runtime-bin")
    runtime_bin.mkdir(parents=True, exist_ok=True)
    lc0_runtime = runtime_bin / "lc0"
    shutil.copy2(lc0, lc0_runtime)
    lc0_runtime.chmod(0o755)
    return lc0_runtime, weights


def check_t4x2() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20
    )
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 2 or not all("T4" in name.upper() for name in names):
        raise RuntimeError(
            "Expected exactly two Tesla T4 GPUs. Found: " + repr(names)
            + ". Set Kaggle Settings -> Accelerator -> GPU T4 x2."
        )
    return names


class Lc0Api:
    def __init__(self, engine: Any, engine_name: str, api_token: str, gpu_names: list[str]):
        self.engine = engine
        self.engine_name = engine_name
        self.api_token = api_token
        self.gpu_names = gpu_names
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.analysis_lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "online",
            "engine": self.engine_name,
            "gpus": self.gpu_names,
            "started_at": self.started_at,
            "api": "/v1/analyze",
        }

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        import chess
        import chess.engine

        fen = payload.get("fen")
        if not isinstance(fen, str) or not fen.strip() or len(fen) > 256:
            raise ValueError("Provide a valid FEN string in 'fen'.")
        board = chess.Board(fen.strip())
        if not board.is_valid():
            raise ValueError("FEN does not describe a valid standard chess position.")
        if not any(board.legal_moves):
            raise ValueError("Position has no legal moves.")

        try:
            movetime_ms = int(payload.get("movetime_ms", DEFAULT_MOVETIME_MS))
        except (TypeError, ValueError) as exc:
            raise ValueError("movetime_ms must be an integer.") from exc
        if not 100 <= movetime_ms <= MAX_MOVETIME_MS:
            raise ValueError(f"movetime_ms must be between 100 and {MAX_MOVETIME_MS}.")

        # One LC0 UCI process is shared. Reject overlapping searches rather than
        # letting stale board positions queue behind one another.
        if not self.analysis_lock.acquire(blocking=False):
            raise BlockingIOError("Engine is already analyzing another position; retry shortly.")
        started = time.perf_counter()
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(time=movetime_ms / 1000.0),
                info=chess.engine.INFO_ALL,
            )
        finally:
            self.analysis_lock.release()

        pv = info.get("pv") or []
        if not pv or pv[0] not in board.legal_moves:
            raise RuntimeError("LC0 returned no legal best move.")

        pv_uci: list[str] = []
        pv_san: list[str] = []
        cursor = board.copy(stack=False)
        for move in pv[:12]:
            if move not in cursor.legal_moves:
                break
            pv_uci.append(move.uci())
            pv_san.append(cursor.san(move))
            cursor.push(move)

        score_result = None
        score = info.get("score")
        if score is not None:
            relative = score.pov(board.turn)
            score_result = {"cp": relative.score(), "mate": relative.mate()}

        return {
            "ok": True,
            "engine": self.engine_name,
            "fen": board.fen(),
            "bestmove": pv_uci[0],
            "bestmove_san": pv_san[0] if pv_san else pv_uci[0],
            "pv_uci": pv_uci,
            "pv_san": pv_san,
            "score": score_result,
            "depth": info.get("depth"),
            "seldepth": info.get("seldepth"),
            "nodes": info.get("nodes"),
            "nps": info.get("nps"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }


def make_handler(api: Lc0Api):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print("[http] " + (fmt % args), flush=True)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # Keep HTTP/1.1 alive so the Android client can reuse its TLS/origin
            # connection for successive small FEN requests. Content-Length frames
            # each response, and BaseHTTPRequestHandler will read the next request.
            self.send_header("Connection", "keep-alive")
            self.send_header("Keep-Alive", "timeout=30, max=100")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, api.health())
            elif self.path == "/":
                self.send_json(200, {"service": "LC0 dual-T4", "health": "/health", "analyze": "POST /v1/analyze"})
            else:
                self.send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/v1/analyze":
                self.send_json(404, {"ok": False, "error": "not_found"})
                return
            auth = self.headers.get("Authorization", "")
            expected = "Bearer " + api.api_token
            if not hmac.compare_digest(auth, expected):
                self.send_json(401, {"ok": False, "error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 8192:
                self.send_json(413, {"ok": False, "error": "invalid_body_size"})
                return
            try:
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object.")
                result = api.analyze(payload)
                self.send_json(200, result)
            except BlockingIOError as exc:
                self.send_json(429, {"ok": False, "error": "engine_busy", "detail": str(exc)})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json(400, {"ok": False, "error": "bad_request", "detail": str(exc)})
            except Exception as exc:
                print("Analysis error:", repr(exc), flush=True)
                self.send_json(503, {"ok": False, "error": "engine_error"})

    return Handler


def download_tailscale() -> tuple[Path, Path]:
    """Download and checksum-verify Tailscale's official static Linux binaries."""
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    if arch is None:
        raise RuntimeError(f"Unsupported Kaggle CPU architecture for Tailscale: {machine}")

    if not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}", TAILSCALE_VERSION):
        raise RuntimeError("LC0_TAILSCALE_VERSION must be a stable x.y.z version.")

    runtime_dir = Path("/kaggle/working/tailscale-runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    archive_path = runtime_dir / f"tailscale_{TAILSCALE_VERSION}_{arch}.tgz"
    url = f"https://pkgs.tailscale.com/stable/tailscale_{TAILSCALE_VERSION}_{arch}.tgz"

    request = urllib.request.Request(url + ".sha256", headers={"User-Agent": "lc0-kaggle-tailscale/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        checksum_text = response.read(4096).decode("ascii", "strict")
    match = re.search(r"([0-9a-fA-F]{64})", checksum_text)
    if not match:
        raise RuntimeError("Tailscale package server returned an invalid SHA256 file.")
    expected_sha = match.group(1).lower()

    if not archive_path.is_file() or sha256_file(archive_path) != expected_sha:
        temporary = archive_path.with_suffix(".download")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "lc0-kaggle-tailscale/1.0"})
        print(f"Downloading verified Tailscale {TAILSCALE_VERSION} static binaries...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256_file(temporary) != expected_sha:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Tailscale archive SHA256 verification failed.")
        temporary.replace(archive_path)

    binaries: dict[str, Path] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for binary_name in ("tailscale", "tailscaled"):
            matches = [
                item for item in archive.getmembers()
                if PurePosixPath(item.name).name == binary_name and item.isfile()
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Expected exactly one {binary_name} binary in the Tailscale archive.")
            source = archive.extractfile(matches[0])
            if source is None:
                raise RuntimeError(f"Unable to read {binary_name} from the Tailscale archive.")
            destination = runtime_dir / binary_name
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            destination.chmod(0o755)
            binaries[binary_name] = destination

    version_result = subprocess.run(
        [str(binaries["tailscale"]), "version"],
        capture_output=True, text=True, timeout=20
    )
    if version_result.returncode != 0 or TAILSCALE_VERSION not in (version_result.stdout + version_result.stderr):
        raise RuntimeError("Downloaded Tailscale CLI failed its version check.")
    return binaries["tailscaled"], binaries["tailscale"]


def start_tailscale(auth_key: str) -> dict[str, Any]:
    """Start Tailscale without /dev/net/tun and attach this Kaggle runtime to the tailnet."""
    tailscaled_path, tailscale_path = download_tailscale()
    socket_path = Path("/tmp/lc0-tailscaled.sock")
    socket_path.unlink(missing_ok=True)
    log_path = Path("/kaggle/working/tailscaled.log")
    log_handle = log_path.open("w", encoding="utf-8")

    daemon = subprocess.Popen(
        [
            str(tailscaled_path),
            "--tun=userspace-networking",
            "--state=mem:",
            f"--socket={socket_path}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )

    try:
        deadline = time.monotonic() + 35
        while not socket_path.exists():
            if daemon.poll() is not None:
                raise RuntimeError(
                    f"tailscaled exited during startup (code {daemon.returncode}); "
                    f"see {log_path} for its log."
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for the Tailscale daemon socket.")
            time.sleep(0.25)

        # Tailscale supports --auth-key=file:<path>. This keeps the key out
        # of process argv and gives the temporary file owner-only permissions.
        auth_file = Path("/tmp/lc0-tailscale-authkey")
        auth_file.unlink(missing_ok=True)
        auth_file.write_text(auth_key, encoding="utf-8")
        auth_file.chmod(0o600)
        try:
            connected = subprocess.run(
                [
                    str(tailscale_path),
                    f"--socket={socket_path}",
                    "up",
                    f"--auth-key=file:{auth_file}",
                    f"--hostname={TAILSCALE_HOSTNAME}",
                    "--accept-dns=false",
                    "--accept-routes=false",
                ],
                capture_output=True, text=True, timeout=120
            )
        finally:
            auth_key = ""
            auth_file.unlink(missing_ok=True)

        if connected.returncode != 0:
            detail = (connected.stderr or connected.stdout).strip()[-2500:]
            raise RuntimeError("Tailscale authentication failed: " + detail)

        status = subprocess.run(
            [str(tailscale_path), f"--socket={socket_path}", "status", "--json"],
            capture_output=True, text=True, check=True, timeout=30,
        )
        payload = json.loads(status.stdout)
        self_info = payload.get("Self") or {}
        addresses = self_info.get("TailscaleIPs") or []
        tailscale_ip = next((ip for ip in addresses if "." in ip), "")
        if not tailscale_ip:
            raise RuntimeError("Tailscale connected but did not report a Tailscale IPv4 address.")
        dns_name = str(self_info.get("DNSName") or "").rstrip(".")

        print("[tailscale] private tailnet connection established.", flush=True)
        print("[tailscale] Kaggle IPv4:", tailscale_ip, flush=True)
        if dns_name:
            print("[tailscale] MagicDNS:", dns_name, flush=True)
        print(f"[tailscale] Phone API base URL: http://{tailscale_ip}:{PORT}", flush=True)
        return {
            "daemon": daemon,
            "cli": tailscale_path,
            "socket": socket_path,
            "log": log_handle,
            "ip": tailscale_ip,
            "dns_name": dns_name,
        }
    except Exception:
        if daemon.poll() is None:
            daemon.terminate()
            try:
                daemon.wait(timeout=8)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
        log_handle.close()
        raise


def stop_tailscale(state: dict[str, Any] | None) -> None:
    if not state:
        return
    daemon = state.get("daemon")
    try:
        subprocess.run(
            [
                str(state["cli"]),
                f"--socket={state['socket']}",
                "logout",
            ],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as exc:
        print("[tailscale] Logout warning:", repr(exc), flush=True)

    if daemon is not None and daemon.poll() is None:
        daemon.terminate()
        try:
            daemon.wait(timeout=8)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)
    log_handle = state.get("log")
    if log_handle:
        log_handle.close()


def main() -> None:
    # Secrets are kept in Kaggle Secrets; only the ephemeral node key and API
    # bearer token are needed. No GitHub or Cloudflare credentials are used.
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("This service must run inside Kaggle Notebooks.") from exc

    secrets_client = UserSecretsClient()
    tailscale_auth_key = get_secret(secrets_client, "TS_AUTH_KEY")
    engine_token = get_secret(secrets_client, "ENGINE_API_TOKEN")

    try:
        import chess
        import chess.engine
    except ImportError as exc:
        raise RuntimeError("python-chess is missing. Run: pip install python-chess==1.999") from exc

    cache_root, manifest = find_cache()
    lc0_path, weights_path = verify_cache(cache_root, manifest)
    gpu_names = check_t4x2()
    print("Cache:", cache_root)
    print("LC0:", manifest.get("lc0_help_first_line", "cached binary"))
    print("BT4 verified:", weights_path.stat().st_size, "bytes")
    print("GPUs:", gpu_names)

    command = [
        str(lc0_path),
        f"--weights={weights_path}",
        "--backend=demux",
        "--backend-opts=backend=cuda-fp16,(gpu=0),(gpu=1)",
        f"--threads={THREADS}",
        f"--minibatch-size={MINIBATCH}",
        f"--nncache={NNCACHE}",
    ]
    print("Starting one persistent dual-T4 UCI process...")
    engine = chess.engine.SimpleEngine.popen_uci(command, timeout=180)
    tailscale = None
    httpd = None

    try:
        engine.ping()
        warmup = engine.analyse(chess.Board(), chess.engine.Limit(time=1.0))
        warm_pv = warmup.get("pv") or []
        if not warm_pv or warm_pv[0] not in chess.Board().legal_moves:
            raise RuntimeError("LC0 warm-up did not return a legal move.")
        engine_name = str(engine.id.get("name", "Lc0"))
        print("UCI engine ready:", engine.id, "warmup bestmove:", warm_pv[0].uci())

        api = Lc0Api(engine, engine_name, engine_token, gpu_names)
        # Keep the HTTP listener private to this Kaggle runtime. Tailscale's
        # userspace netstack forwards authenticated tailnet TCP connections
        # arriving at the node's Tailscale IP to this loopback listener.
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(api))
        httpd.daemon_threads = True
        threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="lc0-http-server",
        ).start()

        tailscale = start_tailscale(tailscale_auth_key)

        print("\n" + "=" * 65)
        print("LC0 DUAL-T4 SERVICE ONLINE — PRIVATE TAILNET ONLY")
        print(f"Phone base URL: http://{tailscale['ip']}:{PORT}")
        print(f"Health URL: http://{tailscale['ip']}:{PORT}/health")
        print(f"Analysis route: http://{tailscale['ip']}:{PORT}/v1/analyze")
        if tailscale.get("dns_name"):
            print(f"MagicDNS base URL: http://{tailscale['dns_name']}:{PORT}")
        print("No public tunnel, public discovery pointer, or GitHub API is used.")
        print("Keep this notebook cell running. Interrupt it for graceful shutdown.")
        print("=" * 65, flush=True)

        while True:
            if tailscale["daemon"].poll() is not None:
                raise RuntimeError(
                    "tailscaled stopped unexpectedly; inspect /kaggle/working/tailscaled.log."
                )
            time.sleep(2)

    except KeyboardInterrupt:
        print("Shutdown requested by user.", flush=True)
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        stop_tailscale(tailscale)
        try:
            engine.quit()
        except Exception:
            pass
        print("LC0 service stopped.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FATAL:", str(error), file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
