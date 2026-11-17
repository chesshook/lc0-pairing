# ============================================================
# LC0 DUAL-T4 — ONE-TIME CLEAN BUILD + CACHE EXPORT
#
# Use this in the FIRST Kaggle notebook. This is the step you
# have NOT done yet: it downloads and compiles LC0, downloads
# BT4 once, tests the dual-T4 UCI engine, and creates
# /kaggle/working/lc0-dual-t4-cache for persistent Notebook Output.
#
# Settings required BEFORE running:
#   1. Accelerator = GPU T4 x2
#   2. Internet = ON
#   3. No Kaggle Secrets are required for this builder.
#
# Run this cell only once for the initial cache build.
# ============================================================

from pathlib import Path
import hashlib
import runpy
import shutil
import urllib.request

BUILDER_URL = (
    "https://raw.githubusercontent.com/chesshook/lc0-pairing/"
    "main/bootstrap_cache.py"
)
BUILDER_SHA256 = "cead1f605fd42a4f666079d26969b5b50c51f13d48e405739d60462757df1e4f"

BUILDER = Path("/kaggle/working/bootstrap_cache.py")
TEMP = BUILDER.with_suffix(".download")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not BUILDER.is_file() or sha256_file(BUILDER) != BUILDER_SHA256:
    request = urllib.request.Request(
        BUILDER_URL,
        headers={"User-Agent": "lc0-kaggle-cache-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, TEMP.open("wb") as out:
        shutil.copyfileobj(response, out)
    TEMP.replace(BUILDER)

if sha256_file(BUILDER) != BUILDER_SHA256:
    raise RuntimeError("Builder SHA256 mismatch; refusing to execute.")

source = BUILDER.read_text(encoding="utf-8")
compile(source, str(BUILDER), "exec")

print("Verified clean-start builder:", BUILDER_SHA256)
print("Starting one-time LC0 v0.32.1 + CUDA build and BT4 cache export.")
print("Do not interrupt the cell during the build/download/smoke test.")
print("The script creates /kaggle/working/lc0-dual-t4-cache.\n", flush=True)

runpy.run_path(str(BUILDER), run_name="__main__")
