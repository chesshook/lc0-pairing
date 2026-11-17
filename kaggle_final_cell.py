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
# Downloads only this small, SHA256-pinned Python service source.
# Never downloads BT4 weights or compiles LC0 at startup.
# ============================================================

from pathlib import Path
import hashlib
import importlib.util
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
SERVER_SHA256 = "b566d20c2e5ddfa020dcc573f4b378cf7072d694c89a00dbcd55b1f8a93a1a33"
SERVER_PATH = Path("/kaggle/working/lc0_kaggle_server.py")
TEMP_PATH = SERVER_PATH.with_suffix(".download")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Confirm all secrets are attached without printing their values.
secrets = UserSecretsClient()
for name in (
    "GITHUB_TOKEN",
    "GITHUB_REPO_OWNER",
    "GITHUB_REPO_NAME",
    "ENGINE_API_TOKEN",
):
    value = secrets.get_secret(name)
    if value is None or not str(value).strip():
        raise RuntimeError(
            f"Missing Kaggle Secret: {name}. Add it under Add-ons -> Secrets "
            "and enable it for this notebook."
        )

# Reuse the cached small source file only if it matches the pinned SHA256.
if not SERVER_PATH.is_file() or sha256_file(SERVER_PATH) != SERVER_SHA256:
    request = urllib.request.Request(
        SERVER_URL,
        headers={"User-Agent": "lc0-kaggle-bootstrap/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, TEMP_PATH.open("wb") as out:
        shutil.copyfileobj(response, out)
    TEMP_PATH.replace(SERVER_PATH)

if sha256_file(SERVER_PATH) != SERVER_SHA256:
    raise RuntimeError("LC0 server source SHA256 mismatch; refusing to execute it.")

# Validate Python syntax before starting any engine process.
source = SERVER_PATH.read_text(encoding="utf-8")
compile(source, str(SERVER_PATH), "exec")

if importlib.util.find_spec("chess") is None:
    print("Installing python-chess (small Python dependency only)...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "python-chess==1.999"],
        check=True,
    )

print("LC0 server source verified:", SERVER_SHA256)
print("Using attached Kaggle Output cache; no LC0 build or BT4 download.")
print("Starting one persistent dual-T4 UCI process in this notebook cell.")
print("Keep this cell running. Interrupt it to request graceful shutdown.\n", flush=True)

# Run in-process so KeyboardInterrupt reaches the service cleanup block.
runpy.run_path(str(SERVER_PATH), run_name="__main__")
