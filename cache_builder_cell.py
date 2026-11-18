# ============================================================
# LC0 DUAL-T4 — ONE-TIME CLEAN BUILD + CACHE EXPORT
#
# FIRST KAGGLE NOTEBOOK
#
# This downloads the build script only, verifies it against the
# GitHub Git-blob SHA1, then executes it. The BT4 network itself
# is separately verified by SHA256 inside bootstrap_cache.py.
# ============================================================

from pathlib import Path
import hashlib
import runpy
import shutil
import urllib.request

BUILDER_URL = (
    "https://raw.githubusercontent.com/chesshook/lc0-pairing/"
    "tailscale-direct/bootstrap_cache.py"
)

# Exact GitHub Contents API blob SHA for bootstrap_cache.py.
EXPECTED_BUILDER_BLOB_SHA1 = "adf5a8063df685ab0b037e76417b3aff3d44014d"

BUILDER = Path("/kaggle/working/bootstrap_cache.py")
TEMP = BUILDER.with_suffix(".download")


def git_blob_sha1(path):
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


if (
    not BUILDER.is_file()
    or git_blob_sha1(BUILDER) != EXPECTED_BUILDER_BLOB_SHA1
):
    request = urllib.request.Request(
        BUILDER_URL,
        headers={"User-Agent": "lc0-kaggle-cache-builder/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, TEMP.open("wb") as out:
        shutil.copyfileobj(response, out)
    TEMP.replace(BUILDER)


actual_blob_sha1 = git_blob_sha1(BUILDER)

if actual_blob_sha1 != EXPECTED_BUILDER_BLOB_SHA1:
    raise RuntimeError(
        "Builder Git blob SHA1 mismatch. "
        f"Expected {EXPECTED_BUILDER_BLOB_SHA1}, got {actual_blob_sha1}. "
        "No build was started."
    )

source = BUILDER.read_text(encoding="utf-8")
compile(source, str(BUILDER), "exec")

print("Verified builder Git blob SHA1:", actual_blob_sha1)
print("Starting one-time LC0 v0.32.1 + CUDA build and BT4 cache export.")
print("Do not interrupt the cell during the build/download/smoke test.")
print("The script creates /kaggle/working/lc0-dual-t4-cache.\n", flush=True)

runpy.run_path(str(BUILDER), run_name="__main__")
