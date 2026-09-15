"""Generate the short Colab notebook; implementation stays in the repository."""
from pathlib import Path

import nbformat as n


REPOSITORY_URL = "https://github.com/smfabrar/minicpm-context-injection.git"
IMPLEMENTATION_COMMIT = "046220688e105c54f1de8328419bd685bc9da4a8"
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    ROOT / "notebooks" / "context_injection_colab.ipynb",
    ROOT.parent / "colab_gpu_evidence_tests.ipynb",
)

notebook = n.v4.new_notebook()
notebook.metadata = {
    "accelerator": "GPU",
    "colab": {"name": "MiniCPM-o context injection on CUDA"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
markdown = n.v4.new_markdown_cell
code = n.v4.new_code_cell

notebook.cells = [
    markdown(
        """# Can MiniCPM-o accept new context while a duplex conversation is running?

This notebook performs fresh native inference on an NVIDIA GPU. It tests whether text that arrives **after a MiniCPM-o 4.5 session has started** can enter the persistent language-model context during speech generation and change a later answer.

No model answer is stored in this notebook. Each run creates raw logs, event timestamps, generated speech, and a machine-readable verdict.

Vision is excluded from the download and disabled at runtime. The experiment uses the Q4 language model, audio encoder, TTS model, and Token2Wav vocoder.
"""
    ),
    markdown(
        """## What happens inside the experiment

MiniCPM receives one-second audio units. The patched runtime lets one of those units carry optional text:

```text
one-second audio ──> audio encoder ──> persistent LLM KV context ──> text and TTS
                                             ▲
                                             │
                                  optional injected text
```

The runtime records four separate kinds of evidence:

| Question | Required evidence |
|---|---|
| Did the text reach the LLM? | Native callback reports successful evaluation and a larger KV position |
| Did it happen while speech was being generated? | The callback falls between two TTS chunks from the same response |
| Did the model use the information? | A later answer contains the exact new room code and excludes the cancelled code |
| Could the code appear anyway? | A matched control receives the same question and seed but not the new code |

These checks are deliberately independent. KV growth is transport evidence; the model's later answer is semantic evidence.
"""
    ),
    markdown(
        """## 1. Load the reviewed implementation

The notebook contains only orchestration and presentation. This cell clones normal Python/C++ source files from GitHub at one fixed commit, then imports them. You can inspect every source file in the cloned directory before running the model.
"""
    ),
    code(
        f'''from pathlib import Path
import json, shutil, subprocess, sys, time

EVIDENCE_REPOSITORY = "{REPOSITORY_URL}"
EVIDENCE_COMMIT = "{IMPLEMENTATION_COMMIT}"
REPOSITORY = Path("/content/minicpm-context-injection")

if not (REPOSITORY / ".git").is_dir():
    subprocess.run(["git", "clone", EVIDENCE_REPOSITORY, str(REPOSITORY)], check=True)
subprocess.run(["git", "-C", str(REPOSITORY), "checkout", "--detach", EVIDENCE_COMMIT], check=True)
actual_commit = subprocess.check_output(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
assert actual_commit == EVIDENCE_COMMIT
sys.path.insert(0, str(REPOSITORY))
print("Loaded implementation commit:", actual_commit)
'''
    ),
    markdown(
        """## 2. Install tools and check the GPU

Compilation runs on Colab's CPU and may be quiet during a large C++ or CUDA file. The build function used later prints a heartbeat every 20 seconds and saves the full output. CUDA accelerates inference after compilation finishes.
"""
    ),
    code(
        """subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run([
    "apt-get", "install", "-y", "-qq", "build-essential", "cmake", "ninja-build",
    "ccache", "git", "ffmpeg", "espeak-ng",
], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPOSITORY / "requirements.txt")], check=True)

from minicpm_injection import (
    Workspace, archive_runs, audit_results, build_cuda, download_models,
    prepare_runtime, require_nvidia, run_mid_generation_pair,
    run_multi_turn_update, show_run, write_combined_report,
)

workspace = Workspace(
    root=Path("/content/minicpm-context-injection-work"),
    repository=REPOSITORY,
).create()
gpu_info = require_nvidia()
free_gib = shutil.disk_usage("/content").free / 2**30
print(f"Free disk: {free_gib:.1f} GiB")
if free_gib < 13:
    raise RuntimeError("At least 13 GiB free disk is required")
"""
    ),
    markdown(
        """## 3. Download only the audio pipeline

The allow-list contains nine files: the Q4 language model, audio encoder, TTS model and projector, and five Token2Wav files. There is no vision path in the list. The downloader verifies all nine files and fails if any vision file appears.

The repository is public. If Hugging Face asks for authentication, add `HF_TOKEN` under Colab's key icon and rerun this cell. The token is never printed.
"""
    ),
    code(
        """try:
    from google.colab import userdata
    try:
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        hf_token = None
except ImportError:
    hf_token = None

model_info = download_models(workspace, token=hf_token)
model_info
"""
    ),
    markdown(
        """## 4. Build the pinned CUDA runtime

This checks out one `llama.cpp-omni` commit and applies a small reviewed patch that transports optional text to the serialized LLM worker and records whether evaluation changed the KV position.

The CUDA build targets only the GPU attached to this runtime:

```text
GGML_CUDA=ON
GGML_METAL=OFF
GGML_NATIVE=ON
CMAKE_CUDA_ARCHITECTURES=native
```

Targeting the attached architecture avoids compiling kernels for every supported NVIDIA generation. The first build can still take several minutes on Colab's two CPU cores. A line saying `[still running: ...]` means the compiler is alive but has not completed its current file. Rerunning uses Ninja and `ccache`.
"""
    ),
    code(
        """runtime_info = prepare_runtime(workspace)
build_info = build_cuda(workspace)
print(json.dumps({"runtime": runtime_info, "build": build_info}, indent=2))
"""
    ),
    markdown(
        """## 5. Experiment A — inject a correction during generated speech

The injected condition follows this timeline:

1. Start one session and provide `Room=B742` with the first audio unit.
2. Ask for the assigned room.
3. Wait for a non-final TTS audio chunk.
4. Generate a previously unknown replacement code and attach the correction to the next input unit.
5. Confirm the text evaluation occurred before a later chunk of that same TTS response.
6. Ask for the current room in the same session.

The control repeats the setup and seed but withholds the replacement code. Every condition is retained, including failures.
"""
    ),
    code(
        """experiment_started = time.time()
mid_generation_results = run_mid_generation_pair(workspace, seed=42)

import pandas as pd
pd.DataFrame([{
    "condition": result.get("condition"),
    "verdict": result.get("verdict"),
    "new code": result.get("new_code"),
    "injection between TTS chunks": result.get("injection_between_tts_chunks"),
    "follow-up answer": result.get("followup_text"),
} for result in mid_generation_results])
"""
    ),
    markdown(
        """A `PASS` in the injected row requires both the timing evidence and the exact replacement code in the later answer. Successful KV evaluation followed by the wrong answer is a semantic `FAIL`. If the model never starts a suitable TTS response, the timing hypothesis is `NOT TESTED` rather than silently failed or passed.
"""
    ),
    markdown(
        """## 6. Experiment B — correction and recall across turns

One native model process remains open for three complete exchanges:

1. inject an initial random room and ask for it;
2. after that exchange finishes, inject a different room and ask again;
3. ask for the current room later without injecting anything.

The test requires the replacement in both later answers, two checked KV updates, and exactly one native session.
"""
    ),
    code(
        """multi_turn_result = run_multi_turn_update(workspace, seed=42)
print(json.dumps(multi_turn_result, indent=2))
"""
    ),
    markdown(
        """## 7. Audit raw evidence and export it

The audit reopens each run directory. It requires the manifest, result, native log, and both event traces; confirms one audio-only session; counts checked context evaluations; and summarizes frame latency. It does not change a model verdict.
"""
    ),
    code(
        """all_results = mid_generation_results + [multi_turn_result]
audits = audit_results(all_results)
display(pd.DataFrame(audits))

report_path = write_combined_report(
    workspace,
    hardware=gpu_info,
    model=model_info,
    runtime=runtime_info,
    build=build_info,
    results=all_results,
)
archive_path = archive_runs(workspace)
print(f"Experiment time: {time.time() - experiment_started:.1f} seconds")
print("Report:", report_path)
print("All runs:", archive_path)

# To download after review:
# from google.colab import files
# files.download(str(report_path))
# files.download(str(archive_path))
"""
    ),
    markdown(
        """## 8. Inspect the timeline and listen to generated speech

Choose any retained run. The table shows fact arrival, checked context evaluation, generated text, TTS chunks, and turn boundaries on one clock. The players use the float32 audio emitted directly by native Token2Wav.
"""
    ),
    code(
        """RUN_TO_REVIEW = mid_generation_results[0]["run_directory"]
show_run(RUN_TO_REVIEW)
"""
    ),
    markdown(
        """## How to state the conclusion

- One injected `PASS` demonstrates feasibility for that retained trial.
- Repeated paired trials are required to estimate reliability.
- GPU latency and semantic reliability are separate results.
- “Between TTS chunks” proves overlap with native speech generation. Hosted Colab cannot prove simultaneous physical browser microphone capture and speaker playback because those devices are outside the remote VM. That final device-level claim requires a WebRTC/WebSocket client or the local hardware notebook.
"""
    ),
]

n.validate(notebook)
for output in OUTPUTS:
    output.parent.mkdir(parents=True, exist_ok=True)
    n.write(notebook, output)
    print(output)
