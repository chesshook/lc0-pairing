#!/usr/bin/env python3
"""Restart-safe LC0 dual-T4 HTTP service for a Kaggle Notebook.

- Reads LC0 + BT4 + cloudflared from a mounted Kaggle Notebook Output dataset.
- Starts one persistent LC0 UCI process with the CUDA FP16 DEMUX backend.
- Exposes a small authenticated JSON API on localhost.
- Creates a Cloudflare Quick Tunnel and publishes the changing URL to hooker.txt.
- Never downloads weights or compiles LC0 at server startup.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue, Empty
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
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


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
        raise RuntimeError("/kaggle/input is missing; attach the LC0 cache notebook output as an Input.")

    candidates = list(input_root.rglob("manifest.json"))
    for manifest_path in candidates:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("weights_sha256") == EXPECTED_WEIGHT_SHA256
                and manifest.get("weights_size") == EXPECTED_WEIGHT_SIZE
            ):
                root = manifest_path.parent
                lc0_path = root / manifest.get("lc0_file", "lc0")
                weights_path = root / manifest.get("weights_file", "BT4-it332.pb.gz")
                tunnel_path = root / manifest.get("cloudflared_file", "cloudflared")
                if lc0_path.is_file() and weights_path.is_file() and tunnel_path.is_file():
                    return root, manifest
        except Exception:
            continue

    raise RuntimeError(
        "No matching LC0 cache found in /kaggle/input. Attach the notebook output "
        "containing lc0, BT4-it332.pb.gz, cloudflared and manifest.json. "
        "Do not run the build again; use cache_export.py once in the validated notebook."
    )


def verify_cache(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    lc0 = root / manifest.get("lc0_file", "lc0")
    weights = root / manifest.get("weights_file", "BT4-it332.pb.gz")
    cloudflared = root / manifest.get("cloudflared_file", "cloudflared")

    if weights.stat().st_size != EXPECTED_WEIGHT_SIZE:
        raise RuntimeError("Mounted BT4 network has the wrong file size.")
    if sha256_file(weights) != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Mounted BT4 network failed SHA256 verification.")
    expected_lc0_hash = manifest.get("lc0_sha256")
    if expected_lc0_hash and sha256_file(lc0) != expected_lc0_hash:
        raise RuntimeError("Mounted LC0 executable failed SHA256 verification.")
    expected_tunnel_hash = manifest.get("cloudflared_sha256")
    if expected_tunnel_hash and sha256_file(cloudflared) != expected_tunnel_hash:
        raise RuntimeError("Mounted cloudflared binary failed SHA256 verification.")

    # Notebook Output Files are mounted read-only. Stage only the two executable
    # tools (normally much smaller than the 383 MB network) into this runtime's
    # scratch directory so execute bits are guaranteed. The weights stay mounted
    # read-only and are not copied or downloaded.
    runtime_bin = Path("/kaggle/working/lc0-runtime-bin")
    runtime_bin.mkdir(parents=True, exist_ok=True)
    lc0_runtime = runtime_bin / "lc0"
    tunnel_runtime = runtime_bin / "cloudflared"
    shutil.copy2(lc0, lc0_runtime)
    shutil.copy2(cloudflared, tunnel_runtime)
    lc0_runtime.chmod(0o755)
    tunnel_runtime.chmod(0o755)
    return lc0_runtime, weights, tunnel_runtime


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


class GitHubPointer:
    """Serially create/update root hooker.txt using the GitHub Contents API."""
    def __init__(self, token: str, owner: str, repo: str, branch: str = "main"):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.branch = branch or "main"
        self.api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/hooker.txt"

    def _request(self, method: str, data: dict[str, Any] | None = None):
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(
            self.api_url + "?ref=" + self.branch if method == "GET" else self.api_url,
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": "Bearer " + self.token,
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "lc0-kaggle-ctf-service/1.0",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return response.status, json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", "replace")[:1500]
            if exc.code == 404 and method == "GET":
                return 404, {}
            raise RuntimeError(f"GitHub API {method} failed ({exc.code}): {error_body}") from exc

    def publish(self, value: str) -> None:
        content = base64.b64encode((value.strip() + "\n").encode("utf-8")).decode("ascii")
        for attempt in range(3):
            status, current = self._request("GET")
            payload: dict[str, Any] = {
                "message": "Publish Kaggle LC0 endpoint" if value != "OFFLINE" else "Mark Kaggle LC0 offline",
                "content": content,
                "branch": self.branch,
            }
            if status == 200 and current.get("sha"):
                payload["sha"] = current["sha"]
            try:
                self._request("PUT", payload)
                print("GitHub pointer updated:", value if value == "OFFLINE" else "ONLINE (URL stored)", flush=True)
                return
            except RuntimeError as exc:
                # A concurrent update may invalidate the Contents API SHA; refresh it and retry.
                if attempt == 2 or ("409" not in str(exc) and "422" not in str(exc)):
                    raise
                time.sleep(1.0 + attempt)
        raise RuntimeError("Unable to publish hooker.txt after three attempts.")


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
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

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


def wait_for_tunnel_url(process: subprocess.Popen[str], timeout_s: int = 90) -> str:
    lines: Queue[str] = Queue()

    def pump() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip())

    threading.Thread(target=pump, daemon=True, name="cloudflared-log-pump").start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"cloudflared exited early (code {process.returncode}).")
        try:
            line = lines.get(timeout=0.5)
        except Empty:
            continue
        print("[cloudflared]", line, flush=True)
        match = TUNNEL_URL_RE.search(line)
        if match:
            return match.group(0)
    raise TimeoutError("Timed out waiting for a trycloudflare.com URL.")


def main() -> None:
    # Kaggle Secrets are deliberately read at runtime, never stored in repo files.
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError as exc:
        raise RuntimeError("This service must run inside Kaggle Notebooks.") from exc

    secrets_client = UserSecretsClient()
    github_token = get_secret(secrets_client, "GITHUB_TOKEN")
    github_owner = get_secret(secrets_client, "GITHUB_REPO_OWNER")
    github_repo = get_secret(secrets_client, "GITHUB_REPO_NAME")
    engine_token = get_secret(secrets_client, "ENGINE_API_TOKEN")
    github_branch = get_secret(secrets_client, "GITHUB_BRANCH", required=False, default="main")

    # python-chess is the only engine-control dependency. It was already imported
    # by the user's successful validation notebook; fail clearly if environment differs.
    try:
        import chess
        import chess.engine
    except ImportError as exc:
        raise RuntimeError("python-chess is missing. Run: pip install python-chess==1.999") from exc

    cache_root, manifest = find_cache()
    lc0_path, weights_path, cloudflared_path = verify_cache(cache_root, manifest)
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
    tunnel = None
    httpd = None
    pointer = GitHubPointer(github_token, github_owner, github_repo, github_branch)
    published = False

    try:
        engine.ping()
        warmup = engine.analyse(chess.Board(), chess.engine.Limit(time=1.0))
        warm_pv = warmup.get("pv") or []
        if not warm_pv or warm_pv[0] not in chess.Board().legal_moves:
            raise RuntimeError("LC0 warm-up did not return a legal move.")
        engine_name = str(engine.id.get("name", "Lc0"))
        print("UCI engine ready:", engine.id, "warmup bestmove:", warm_pv[0].uci())

        api = Lc0Api(engine, engine_name, engine_token, gpu_names)
        httpd = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(api))
        httpd.daemon_threads = True
        threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.25},
            daemon=True,
            name="lc0-http-server",
        ).start()

        tunnel = subprocess.Popen(
            [str(cloudflared_path), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        public_url = wait_for_tunnel_url(tunnel)
        pointer.publish(public_url)
        published = True

        raw_pointer = (
            f"https://raw.githubusercontent.com/{github_owner}/{github_repo}/"
            f"{github_branch}/hooker.txt"
        )
        print("\n" + "=" * 65)
        print("LC0 DUAL-T4 SERVICE ONLINE")
        print("Public base URL:", public_url)
        print("Health URL:", public_url + "/health")
        print("Analysis route:", public_url + "/v1/analyze")
        print("GitHub discovery file:", raw_pointer)
        print("API token: loaded from Kaggle Secrets; not displayed")
        print("Keep this notebook cell running. Interrupt it for graceful shutdown.")
        print("=" * 65, flush=True)

        while True:
            if tunnel.poll() is not None:
                raise RuntimeError(f"cloudflared stopped (code {tunnel.returncode}).")
            time.sleep(2)

    except KeyboardInterrupt:
        print("Shutdown requested by user.", flush=True)
    finally:
        if published:
            try:
                pointer.publish("OFFLINE")
            except Exception as exc:
                print("Could not publish OFFLINE marker:", repr(exc), flush=True)
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=8)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
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
