# ============================================================
# LC0 DUAL-T4 — FINAL ONE-CELL KAGGLE SERVER
#
# Prerequisites:
#   - Kaggle Accelerator: GPU T4 x2
#   - Internet: ON
#   - Attach the saved "lc0-dual-t4-cache" notebook output as Input
#   - Enable these Kaggle Secrets:
#       GITHUB_TOKEN
#       GITHUB_REPO_OWNER
#       GITHUB_REPO_NAME
#       GITHUB_BRANCH
#       ENGINE_API_TOKEN
#
# Downloads only this small Python service source.
# Never downloads BT4 weights or compiles LC0 at startup.
# ============================================================

from pathlib import Path
import importlib.util
import hashlib
import runpy
import shutil
import subprocess
import sys
import urllib.request

from kaggle_secrets import UserSecretsClient

REPO_OWNER = "chesshook"
REPO_NAME = "lc0-pairing"
BRANCH = "main"

SERVER_URL = (
    "https://raw.githubusercontent.com/"
    + REPO_OWNER + "/" + REPO_NAME + "/" + BRANCH
    + "/lc0_kaggle_server.py"
)
SERVER_PATH = Path("/kaggle/working/lc0_kaggle_server.py")
TEMP_PATH = SERVER_PATH.with_suffix(".download")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Confirm that credentials are present without printing or exposing values.
secrets = UserSecretsClient()
required_secrets = (
    "GITHUB_TOKEN",
    "GITHUB_REPO_OWNER",
    "GITHUB_REPO_NAME",
    "ENGINE_API_TOKEN",
)
for name in required_secrets:
    value = secrets.get_secret(name)
    if value is None or not str(value).strip():
        raise RuntimeError(
            f"Missing Kaggle Secret: {name}. Add it under Add-ons -> Secrets "
            "and enable it for this notebook."
        )

# Reuse this small file if it is already present in the current runtime.
# A new Kaggle runtime downloads only ~20 KB of source code.
source_is_ready = False
if SERVER_PATH.is_file():
    source_is_ready = (
        SERVER_PATH.stat().st_size == 20235
    )

if not source_is_ready:
    request = urllib.request.Request(
        SERVER_URL,
        headers={"User-Agent": "lc0-kaggle-bootstrap/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, TEMP_PATH.open("wb") as out:
        shutil.copyfileobj(response, out)
    TEMP_PATH.replace(SERVER_PATH)

if SERVER_PATH.stat().st_size != 20235:
    raise RuntimeError(
        "Unexpected server source size. Check the public GitHub source file."
    )

# Import python-chess only if this Kaggle environment does not already have it.
if importlib.util.find_spec("chess") is None:
    print("Installing python-chess (small Python dependency only)...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "python-chess==1.999"],
        check=True,
    )

print("LC0 server source ready:", SERVER_PATH)
print("Using saved Kaggle Output cache; no LC0 build or BT4 download.")
print("Starting one persistent dual-T4 UCI process in this notebook cell.")
print("Keep this cell running. Interrupt it to request graceful shutdown.\n", flush=True)

# IMPORTANT: run in-process so KeyboardInterrupt triggers the service's
# finally block, which stops the tunnel, HTTP server and LC0 engine.
runpy.run_path(str(SERVER_PATH), run_name="__main__")
