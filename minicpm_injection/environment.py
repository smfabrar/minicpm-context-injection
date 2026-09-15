"""Download the audio-only models and build the pinned CUDA runtime.

The notebook calls these functions so setup code remains ordinary, reviewable
Python. Long builds emit a heartbeat and write their complete output to a log.
"""
from __future__ import annotations

import hashlib
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RUNTIME_REPOSITORY = "https://github.com/tc-mb/llama.cpp-omni.git"
RUNTIME_COMMIT = "64d092c60db4b4ee45768476bd752f03fdcc98ea"
MODEL_REPOSITORY = "openbmb/MiniCPM-o-4_5-gguf"
MODEL_REVISION = "db25077c33951fe163b42986fba0132e279872a2"
MODEL_FILES = (
    "MiniCPM-o-4_5-Q4_K_M.gguf",
    "audio/MiniCPM-o-4_5-audio-F16.gguf",
    "tts/MiniCPM-o-4_5-tts-F16.gguf",
    "tts/MiniCPM-o-4_5-projector-F16.gguf",
    "token2wav-gguf/encoder.gguf",
    "token2wav-gguf/flow_extra.gguf",
    "token2wav-gguf/flow_matching.gguf",
    "token2wav-gguf/hifigan2.gguf",
    "token2wav-gguf/prompt_cache.gguf",
)


@dataclass(frozen=True)
class Workspace:
    """All large and generated files live outside the cloned evidence repo."""

    root: Path
    repository: Path

    @property
    def runtime(self) -> Path:
        return self.root / "runtime" / "llama.cpp-omni"

    @property
    def models(self) -> Path:
        return self.root / "models" / "MiniCPM-o-4_5-gguf"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def bridge(self) -> Path:
        return self.repository / "native" / "build" / "native-session"

    def create(self) -> "Workspace":
        for path in (self.root, self.runtime.parent, self.models, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        return self


def _command_text(command: Iterable[object]) -> str:
    return " ".join(str(part) for part in command)


def run(command: Iterable[object], *, cwd: Path | None = None, capture: bool = False) -> str:
    """Run a short command and show exactly what was executed."""

    command = [str(part) for part in command]
    print("$", _command_text(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    return completed.stdout or ""


def run_with_heartbeat(
    command: Iterable[object],
    *,
    cwd: Path | None,
    log_path: Path,
    heartbeat_seconds: int = 20,
) -> None:
    """Stream command output and report elapsed time during quiet compilations."""

    command = [str(part) for part in command]
    print("$", _command_text(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    messages: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started = time.monotonic()
    reader_finished = False
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {_command_text(command)}\n")
        while not reader_finished:
            try:
                message = messages.get(timeout=heartbeat_seconds)
            except queue.Empty:
                elapsed = time.monotonic() - started
                message = f"[still running: {elapsed:.0f} seconds elapsed]\n"
            if message is None:
                reader_finished = True
            else:
                print(message, end="", flush=True)
                log.write(message)
                log.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    print(f"Completed in {time.monotonic() - started:.1f} seconds. Log: {log_path}")


def require_nvidia() -> dict[str, str]:
    """Return the assigned GPU description or fail before expensive setup."""

    if not shutil.which("nvidia-smi"):
        raise RuntimeError(
            "No NVIDIA GPU is attached. In Colab choose Runtime > Change runtime type > NVIDIA GPU."
        )
    fields = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).strip().split(", ")
    if len(fields) != 4:
        raise RuntimeError(f"Could not parse nvidia-smi output: {fields}")
    gpu = dict(zip(("name", "memory_mib", "driver", "compute_capability"), fields))
    print("Assigned GPU:", gpu)
    return gpu


def download_models(workspace: Workspace, token: str | None = None) -> dict[str, object]:
    """Download the fixed audio-only allow-list. Vision is deliberately absent."""

    from huggingface_hub import snapshot_download

    workspace.create()
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        allow_patterns=list(MODEL_FILES),
        local_dir=workspace.models,
        token=token or None,
    )
    missing = [name for name in MODEL_FILES if not (workspace.models / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete model download: {missing}")
    vision_files = [path for path in workspace.models.glob("vision/**/*") if path.is_file()]
    if vision_files:
        raise RuntimeError(f"Vision files were unexpectedly downloaded: {vision_files}")
    size = sum((workspace.models / name).stat().st_size for name in MODEL_FILES)
    result = {
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "files": list(MODEL_FILES),
        "size_gib": size / 2**30,
        "vision_downloaded": False,
    }
    print(f"Verified {len(MODEL_FILES)} files ({result['size_gib']:.2f} GiB); vision absent.")
    return result


def prepare_runtime(workspace: Workspace) -> dict[str, str]:
    """Check out the pinned runtime and apply the small injection transport patch."""

    workspace.create()
    if not (workspace.runtime / ".git").is_dir():
        run(["git", "clone", "--filter=blob:none", RUNTIME_REPOSITORY, workspace.runtime])
    run(["git", "fetch", "origin", RUNTIME_COMMIT], cwd=workspace.runtime)
    run(["git", "checkout", "--detach", RUNTIME_COMMIT], cwd=workspace.runtime)
    head = run(["git", "rev-parse", "HEAD"], cwd=workspace.runtime, capture=True).strip()
    if head != RUNTIME_COMMIT:
        raise RuntimeError(f"Wrong runtime revision: {head}")

    patch_path = workspace.repository / "patches" / "context-injection.patch"
    patch = patch_path.read_text()
    patch_hash = hashlib.sha256(patch.encode()).hexdigest()
    forward = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        cwd=workspace.runtime,
        input=patch,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if forward.returncode == 0:
        applied = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=workspace.runtime,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if applied.returncode:
            raise RuntimeError(applied.stdout)
    else:
        reverse = subprocess.run(
            ["git", "apply", "--reverse", "--check", "--whitespace=nowarn", "-"],
            cwd=workspace.runtime,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if reverse.returncode:
            raise RuntimeError("Runtime is neither clean nor exactly patched:\n" + forward.stdout)

    actual = run(
        ["git", "diff", "--", "tools/omni/omni.cpp", "tools/omni/omni.h"],
        cwd=workspace.runtime,
        capture=True,
    )
    if hashlib.sha256(actual.encode()).hexdigest() != patch_hash:
        raise RuntimeError("Applied runtime patch differs from the reviewed patch")
    print("Pinned runtime and context-injection patch verified.")
    return {"repository": RUNTIME_REPOSITORY, "commit": head, "patch_sha256": patch_hash}


def build_cuda(workspace: Workspace) -> dict[str, object]:
    """Build only for the attached GPU and then compile the JSONL bridge."""

    build = workspace.runtime / "build-cuda"
    log = workspace.root / "build.log"
    launchers: list[str] = []
    if shutil.which("ccache"):
        launchers = [
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ]
    configure = [
        "cmake",
        "-S",
        workspace.runtime,
        "-B",
        build,
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA=ON",
        "-DGGML_METAL=OFF",
        "-DGGML_NATIVE=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=native",
        *launchers,
    ]
    run_with_heartbeat(configure, cwd=workspace.runtime, log_path=log)
    jobs = str(max(1, min(4, os.cpu_count() or 2)))
    run_with_heartbeat(
        ["cmake", "--build", build, "--target", "omni", "-j", jobs],
        cwd=workspace.runtime,
        log_path=log,
    )
    cuda_libraries = list(build.rglob("libggml-cuda.so*"))
    if not cuda_libraries:
        raise RuntimeError("CUDA was requested but libggml-cuda.so was not built")

    native_build = workspace.repository / "native" / "build"
    run_with_heartbeat(
        [
            "cmake",
            "-S",
            workspace.repository / "native",
            "-B",
            native_build,
            "-G",
            "Ninja",
            f"-DRUNTIME={workspace.runtime}",
            f"-DRUNTIME_BUILD={build}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=workspace.repository,
        log_path=log,
    )
    run_with_heartbeat(
        ["cmake", "--build", native_build, "-j", jobs],
        cwd=workspace.repository,
        log_path=log,
    )
    if not workspace.bridge.is_file():
        raise RuntimeError("Native session bridge was not built")
    links = run(["ldd", workspace.bridge], capture=True)
    if "not found" in links:
        raise RuntimeError("Native bridge has unresolved libraries:\n" + links)
    print("CUDA backend:", cuda_libraries[0])
    print("Native bridge:", workspace.bridge)
    return {
        "build_directory": str(build),
        "bridge": str(workspace.bridge),
        "jobs": int(jobs),
        "native_gpu_architecture_only": True,
        "log": str(log),
    }
