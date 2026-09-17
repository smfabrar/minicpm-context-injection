"""Check dependencies while handling Decord 0.6.0's documented wheel-tag defect.

Upstream: https://github.com/dmlc/decord/issues/356
The published Linux py3 wheel contains a stale cp36 tag in its WHEEL metadata.
No installed metadata or binaries are changed. All other pip errors remain fatal.
"""
from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys

DECORD_ERROR = "decord 0.6.0 is not supported on this platform"
STALE_TAG = "Tag: cp36-cp36m-manylinux2010_x86_64"


def known_decord_metadata_error(output, returncode, *, system, machine, version, wheel):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    tags = [line.strip() for line in wheel.splitlines() if line.startswith("Tag:")]
    return (returncode == 1 and lines == [DECORD_ERROR] and system == "linux"
            and machine == "x86_64" and version == "0.6.0" and tags == [STALE_TAG])


def check_dependencies():
    completed = subprocess.run([sys.executable, "-m", "pip", "check"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode == 0:
        print(completed.stdout.strip(), flush=True)
        return
    try:
        distribution = importlib.metadata.distribution("decord")
        version = distribution.version
        wheel = distribution.read_text("WHEEL") or ""
    except importlib.metadata.PackageNotFoundError:
        version, wheel = "", ""
    if not known_decord_metadata_error(completed.stdout, completed.returncode,
                                      system=sys.platform, machine=platform.machine(),
                                      version=version, wheel=wheel):
        raise RuntimeError("Dependency check failed:\n" + completed.stdout)
    # Import loads the prebuilt native library. A real binary/loader failure is
    # fatal even when pip's only complaint is the known metadata defect.
    import decord
    if decord.__version__ != "0.6.0":
        raise RuntimeError("Imported Decord differs from the installed distribution")
    print("Dependency check passed. Decord 0.6.0 native library imported; "
          "its documented stale Python 3.6 wheel tag is the only pip-check error.", flush=True)


if __name__ == "__main__":
    check_dependencies()
