#!/usr/bin/env python3
"""Export the already-built, validated LC0 + BT4 artifacts as Kaggle notebook output.

Run this in the SAME Kaggle notebook session that currently contains the validated
artifacts under /kaggle/working/lc0_dual_t4_lab. It does not rebuild LC0.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ROOT = Path(os.environ.get(
    "LC0_SOURCE_ROOT", "/kaggle/working/lc0_dual_t4_lab"
))
OUTPUT_ROOT = Path(os.environ.get(
    "LC0_CACHE_OUTPUT", "/kaggle/working/lc0-dual-t4-cache"
))
EXPECTED_WEIGHT_SIZE = 382_645_315
EXPECTED_WEIGHT_SHA256 = (
    "e6ada9d6c4a769bfab3aa0848d82caeb"
    "809aa45f83e6c605fc58a31d21bdd618"
)
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cloudflared(destination: Path) -> str:
    if destination.is_file() and os.access(destination, os.X_OK):
        probe = subprocess.run(
            [str(destination), "--version"], capture_output=True,
            text=True, timeout=20
        )
        if probe.returncode == 0:
            return (probe.stdout or probe.stderr).strip().splitlines()[0]

    temporary = destination.with_suffix(".download")
    request = urllib.request.Request(
        CLOUDFLARED_URL,
        headers={"User-Agent": "lc0-kaggle-ctf-cache/1.0"},
    )
    print("Downloading the small Cloudflare tunnel client once for the cache...")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    temporary.chmod(0o755)
    temporary.replace(destination)

    probe = subprocess.run(
        [str(destination), "--version"], capture_output=True,
        text=True, timeout=20
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "Downloaded cloudflared did not execute: " + (probe.stderr or probe.stdout)
        )
    return (probe.stdout or probe.stderr).strip().splitlines()[0]


def main() -> None:
    lc0_source = SOURCE_ROOT / "lc0" / "build" / "release" / "lc0"
    weights_source = SOURCE_ROOT / "BT4-it332.pb.gz"

    if not lc0_source.is_file():
        raise FileNotFoundError(f"Validated LC0 binary not found: {lc0_source}")
    if not weights_source.is_file():
        raise FileNotFoundError(f"Validated BT4 network not found: {weights_source}")

    actual_size = weights_source.stat().st_size
    actual_hash = sha256(weights_source)
    if actual_size != EXPECTED_WEIGHT_SIZE:
        raise RuntimeError(
            f"BT4 size mismatch: expected {EXPECTED_WEIGHT_SIZE}, got {actual_size}"
        )
    if actual_hash != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(
            "BT4 SHA256 mismatch; refusing to cache an unverified weights file: "
            + actual_hash
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lc0_target = OUTPUT_ROOT / "lc0"
    weights_target = OUTPUT_ROOT / "BT4-it332.pb.gz"
    tunnel_target = OUTPUT_ROOT / "cloudflared"

    shutil.copy2(lc0_source, lc0_target)
    shutil.copy2(weights_source, weights_target)
    lc0_target.chmod(0o755)
    cloudflared_version = download_cloudflared(tunnel_target)

    lc0_probe = subprocess.run(
        [str(lc0_target), "--help"], capture_output=True,
        text=True, timeout=30
    )
    if lc0_probe.returncode != 0:
        raise RuntimeError("Cached LC0 binary failed --help: " + lc0_probe.stderr[-2000:])
    help_text = (lc0_probe.stdout or "") + "\n" + (lc0_probe.stderr or "")
    for required in ("cuda-fp16", "demux"):
        if required not in help_text:
            raise RuntimeError(f"Cached LC0 binary does not advertise required backend: {required}")

    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lc0_source_root": str(SOURCE_ROOT),
        "lc0_file": "lc0",
        "lc0_sha256": sha256(lc0_target),
        "lc0_help_first_line": next(
            (line.strip() for line in help_text.splitlines() if line.strip()), "unknown"
        ),
        "weights_file": "BT4-it332.pb.gz",
        "weights_size": actual_size,
        "weights_sha256": actual_hash,
        "cloudflared_file": "cloudflared",
        "cloudflared_version": cloudflared_version,
        "cloudflared_sha256": sha256(tunnel_target),
        "validated_engine": "Lc0 v0.32.1 / CUDA FP16 / DEMUX / T4x2",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print("\nCACHE EXPORT COMPLETE")
    print("Output folder:", OUTPUT_ROOT)
    print("Files:")
    for item in sorted(OUTPUT_ROOT.iterdir()):
        print(f"  {item.name:24s} {item.stat().st_size:>12,} bytes")
    print("BT4 SHA256:", actual_hash)
    print("LC0 SHA256:", manifest["lc0_sha256"])
    print("cloudflared:", cloudflared_version)
    print("\nNext: save this notebook version, then use its Output files as an Input")
    print("to the separate LC0 server notebook. No build is performed by this script.")


if __name__ == "__main__":
    main()
