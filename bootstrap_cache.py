#!/usr/bin/env python3
"""One-time clean-start builder for the Kaggle LC0 dual-T4 cache.

Run this only in the first Kaggle GPU T4x2 notebook. It builds LC0 v0.32.1
with its native CUDA FP16 backend, downloads the pinned BT4-it332 weights, smoke-tests the dual-GPU UCI
engine, and writes a small reusable folder under /kaggle/working/lc0-dual-t4-cache.

After successfully saving this notebook version's Output Files, future server
notebooks attach that folder as an Input. They do not run this build script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LC0_TAG = "v0.32.1"
LC0_GIT_URL = "https://github.com/LeelaChessZero/lc0.git"
WEIGHTS_URL = (
    "https://storage.lczero.org/files/networks-contrib/"
    "BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
)
WEIGHTS_NAME = "BT4-it332.pb.gz"
EXPECTED_WEIGHT_SIZE = 382_645_315
EXPECTED_WEIGHT_SHA256 = (
    "e6ada9d6c4a769bfab3aa0848d82caeb"
    "809aa45f83e6c605fc58a31d21bdd618"
)
WORK = Path("/kaggle/working")
SOURCE_ROOT = WORK / "lc0_dual_t4_lab"
OUTPUT_ROOT = WORK / "lc0-dual-t4-cache"


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env=None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("\n$", " ".join(str(x) for x in command), flush=True)
    return subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        check=True,
        capture_output=capture,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_size: int, expected_sha: str) -> None:
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256_file(destination) == expected_sha:
            print("Reusing already verified file:", destination.name)
            return
        destination.unlink()

    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "lc0-kaggle-cache-builder/1.0"},
    )
    print(f"Downloading {destination.name} ...", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_report = 0
        with temporary.open("wb") as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_report >= 32 * 1024 * 1024 or (total and downloaded == total):
                    if total:
                        percent = 100.0 * downloaded / total
                        print(
                            f"  Download progress: {downloaded:,}/{total:,} bytes ({percent:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"  Downloaded: {downloaded:,} bytes", flush=True)
                    last_report = downloaded

    actual_size = temporary.stat().st_size
    actual_sha = sha256_file(temporary)
    print("Downloaded bytes:", actual_size)
    print("Downloaded SHA256:", actual_sha)

    if actual_size != expected_size or actual_sha != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{destination.name} integrity check failed. "
            f"Expected {expected_size} bytes / {expected_sha}; "
            f"received {actual_size} bytes / {actual_sha}."
        )
    temporary.replace(destination)


def verify_hardware_and_cuda() -> tuple[list[str], str]:
    gpu = run([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ], capture=True).stdout
    gpu_names = [line.strip() for line in gpu.splitlines() if line.strip()]
    print("Detected GPUs:", gpu_names, flush=True)
    if len(gpu_names) != 2 or not all("T4" in name.upper() for name in gpu_names):
        raise RuntimeError(
            "This builder requires Kaggle GPU T4 x2. Found: " + repr(gpu_names)
            + ". In Notebook Settings choose Accelerator -> GPU T4 x2."
        )

    cuda_bin = Path("/usr/local/cuda/bin")
    if cuda_bin.is_dir():
        os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError(
            "nvcc is not available. Confirm GPU T4 x2 is selected and the Kaggle "
            "GPU runtime has initialized. LC0's native CUDA backend requires nvcc."
        )
    nvcc_version = run([nvcc, "--version"], capture=True).stdout
    print("CUDA compiler:", nvcc)
    print(nvcc_version)
    run(["g++", "--version"])
    return gpu_names, nvcc


def install_build_dependencies() -> None:
    # Kaggle GPU images normally have the compiler and CUDA toolkit already.
    # Install the small source-build tools and headers required by Meson.
    run(["apt-get", "update", "-qq"])
    run([
        "apt-get", "install", "-y", "-qq",
        "build-essential", "git", "pkg-config", "zlib1g-dev",
    ])
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "--upgrade", "meson", "ninja",
    ])


def build_lc0() -> Path:
    if SOURCE_ROOT.exists():
        shutil.rmtree(SOURCE_ROOT)

    run([
        "git", "clone",
        "--branch", LC0_TAG,
        "--depth", "1",
        "--recurse-submodules",
        "--shallow-submodules",
        LC0_GIT_URL,
        SOURCE_ROOT,
    ])

    # The T4 is compute capability 7.5. Pin the CUDA code target explicitly.
    # plain_cuda builds LC0's native CUDA backend, which exposes cuda-fp16.
    build_env = os.environ.copy()
    cuda_bin = "/usr/local/cuda/bin"
    build_env["PATH"] = cuda_bin + os.pathsep + build_env.get("PATH", "")

    run([
        "./build.sh", "release",
        "-Dgtest=false",
        "-Donnx=false",
        "-Dplain_cuda=true",
        "-Dcudnn=false",
        "-Dcc_cuda=75",
        "-Dnative_cuda=false",
        "-Dblas=false",
        "-Dopenblas=false",
    ], cwd=SOURCE_ROOT, env=build_env)

    binary = SOURCE_ROOT / "build" / "release" / "lc0"
    if not binary.is_file():
        raise RuntimeError(f"LC0 build finished without expected binary: {binary}")
    binary.chmod(0o755)

    help_result = subprocess.run(
        [str(binary), "--help"],
        cwd=str(SOURCE_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    help_text = (help_result.stdout or "") + "\n" + (help_result.stderr or "")
    if help_result.returncode != 0:
        raise RuntimeError("Newly built LC0 failed --help: " + help_text[-4000:])
    missing = [name for name in ("cuda-fp16", "demux") if name not in help_text]
    if missing:
        raise RuntimeError(
            "LC0 build is missing required backend(s): " + ", ".join(missing)
            + ". Check nvcc, cuBLAS, CUDA headers and Meson output above."
        )
    print("LC0 help/backend check passed.")
    return binary


def smoke_test_dual(binary: Path, weights: Path, gpu_names: list[str]) -> dict:
    if importlib.util.find_spec("chess") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "python-chess==1.999"])

    import chess
    import chess.engine

    command = [
        str(binary),
        f"--weights={weights}",
        "--backend=demux",
        "--backend-opts=backend=cuda-fp16,(gpu=0),(gpu=1)",
        "--threads=4",
        "--minibatch-size=256",
        "--nncache=1000000",
    ]

    print("\nStarting dual-T4 UCI smoke test ...", flush=True)
    engine = chess.engine.SimpleEngine.popen_uci(command, timeout=180)
    checks = []
    try:
        engine.ping()
        positions = [
            ("Starting position", chess.Board()),
            ("Ruy Lopez", chess.Board()),
            ("Pawn endgame", chess.Board("8/5pk1/6p1/8/4P3/6P1/5PK1/8 w - - 0 40")),
        ]
        for name, board in positions:
            started = __import__("time").perf_counter()
            info = engine.analyse(board, chess.engine.Limit(time=2.0))
            elapsed = __import__("time").perf_counter() - started
            pv = info.get("pv") or []
            if not pv or pv[0] not in board.legal_moves:
                raise RuntimeError(f"LC0 did not return a legal move for {name}.")
            result = {
                "position": name,
                "bestmove_uci": pv[0].uci(),
                "bestmove_san": board.san(pv[0]),
                "depth": info.get("depth"),
                "nodes": info.get("nodes"),
                "nps": info.get("nps"),
                "seconds": round(elapsed, 3),
                "legal": True,
            }
            print("PASS:", result, flush=True)
            checks.append(result)
    finally:
        engine.quit()

    # Record GPU inventory after successful dual backend initialization.
    nvidia = run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.used",
        "--format=csv,noheader",
    ], capture=True).stdout
    return {
        "engine_id": "Lc0 v0.32.1",
        "gpu_names": gpu_names,
        "positions": checks,
        "nvidia_smi_after_test": nvidia.strip(),
    }


def main() -> None:
    print("=" * 72)
    print("LC0 v0.32.1 — CLEAN BUILD AND PERSISTENT CACHE EXPORT")
    print("=" * 72)

    gpu_names, _ = verify_hardware_and_cuda()
    install_build_dependencies()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Remove artifacts from older Cloudflare-based cache builds.
    (OUTPUT_ROOT / "cloudflared").unlink(missing_ok=True)
    weights = OUTPUT_ROOT / WEIGHTS_NAME
    download_verified(
        WEIGHTS_URL, weights,
        EXPECTED_WEIGHT_SIZE, EXPECTED_WEIGHT_SHA256,
    )

    binary_source = build_lc0()
    binary_target = OUTPUT_ROOT / "lc0"
    shutil.copy2(binary_source, binary_target)
    binary_target.chmod(0o755)

    test_report = smoke_test_dual(binary_target, weights, gpu_names)

    manifest = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tag": LC0_TAG,
        "lc0_file": "lc0",
        "lc0_sha256": sha256_file(binary_target),
        "lc0_help_first_line": "Lc0 v0.32.1",
        "weights_file": WEIGHTS_NAME,
        "weights_size": weights.stat().st_size,
        "weights_sha256": sha256_file(weights),
        "validated_engine": "Lc0 v0.32.1 / CUDA FP16 / DEMUX / T4x2",
        "smoke_test": test_report,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # The source tree and intermediate build files are no longer needed.
    # Keep only the three runtime artifacts in /kaggle/working for Output Files.
    shutil.rmtree(SOURCE_ROOT, ignore_errors=True)

    print("\n" + "=" * 72)
    print("CACHE BUILD COMPLETE")
    print("Output folder:", OUTPUT_ROOT)
    print("Files to preserve as Notebook Output:")
    for item in sorted(OUTPUT_ROOT.iterdir()):
        print(f"  {item.name:24s} {item.stat().st_size:>12,} bytes")
    print("BT4 SHA256:", manifest["weights_sha256"])
    print("LC0 SHA256:", manifest["lc0_sha256"])
    print("Dual-T4 smoke-test positions:", len(test_report["positions"]))
    print("\nNEXT: Save a Kaggle Notebook Version with Save Output enabled.")
    print("Then attach this notebook's Output Files to the separate server notebook.")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
