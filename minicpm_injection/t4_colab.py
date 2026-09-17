"""Small notebook-facing helpers; model inference runs in isolated Python 3.11."""
from __future__ import annotations

import html
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColabWorkspace:
    repository: Path
    root: Path

    @property
    def python(self):
        return self.root / "venv" / "bin" / "python"


def _logged_command(command, log_path, *, env=None, verbose=False):
    """Save complete output; notebook shows experiment progress and heartbeats."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(c) for c in command], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    messages = queue.Queue()

    def read():
        for line in process.stdout:
            messages.put(line)
        messages.put(None)

    threading.Thread(target=read, daemon=True).start()
    started = time.monotonic()
    try:
        with log_path.open("a") as log:
            while True:
                try:
                    line = messages.get(timeout=20)
                except queue.Empty:
                    print(f"Still running · {time.monotonic() - started:.0f}s · full log: {log_path}", flush=True)
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if verbose or line.startswith(("baseline:", "followup:", "recall:", "initial:", "update:")):
                    print(line, end="", flush=True)
            status = process.wait()
        if status:
            tail = log_path.read_text().splitlines()[-12:]
            print("\n".join(tail))
            raise RuntimeError(f"Command failed (exit {status}). Complete log: {log_path}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def setup_environment(workspace):
    """Use native wheels and allow packaging the pure Python MiniCPM utils."""
    from IPython.display import HTML, display

    if not shutil.which("nvidia-smi"):
        raise RuntimeError("Select Runtime > Change runtime type > T4 GPU")
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    workspace.root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(workspace.root).free < 22 * 2**30 and not workspace.python.exists():
        raise RuntimeError("At least 22 GiB free disk is required for first setup")
    log = workspace.root / "setup.log"
    commands = [
        ("Install environment manager", [sys.executable, "-m", "pip", "install", "--only-binary=:all:", "uv==0.8.22"]),
    ]
    for label, command in commands:
        print(label, flush=True)
        _logged_command(command, log)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is missing after installation")
    print("Download prebuilt Python 3.11", flush=True)
    _logged_command([uv, "python", "install", "3.11"], log)
    if not workspace.python.exists():
        _logged_command([uv, "venv", "--python", "3.11", "--seed", workspace.root / "venv"], log)
    print("Install matching PyTorch and GPTQ wheels", flush=True)
    _logged_command([workspace.python, "-m", "pip", "install", "--only-binary=:all:",
                     "--no-binary=minicpmo-utils",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu121",
                     "-r", workspace.repository / "requirements-python.lock"], log)
    print("Install audio tools", flush=True)
    _logged_command(["apt-get", "update", "-qq"], log)
    _logged_command(["apt-get", "install", "-y", "-qq", "ffmpeg", "espeak-ng", "libsndfile1"], log)
    _logged_command([workspace.python, "-m", "pip", "check"], log)
    _logged_command([workspace.python, "-c",
        "import torch,transformers,gptqmodel,stepaudio2; "
        "assert torch.cuda.is_available(); "
        "print('PyTorch:',torch.__version__,'Transformers:',transformers.__version__); "
        "print('GPU:',torch.cuda.get_device_name(0))"], log, verbose=True)
    display(HTML(_table(["GPU", "Precision", "Vision", "CUDA source build"],
                        [[gpu, "GPTQ 4-bit / FP16", "Disabled", "None"]])))
    print("Setup complete. Full installation log:", log)


def run_evidence(workspace, *, mid_speech=False, token=None):
    """Run a fresh pair and load its report, retaining all error logs."""
    import fcntl

    label = "mid_speech" if mid_speech else "correction_and_recall"
    runs = workspace.root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / ".inference.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An experiment is already running. Wait for it to finish.")
        root = runs / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_{uuid.uuid4().hex[:6]}"
        root.mkdir()
        child_env = os.environ.copy()
        if token:
            child_env["HF_TOKEN"] = token
        command = [workspace.python, "-u", workspace.repository / "minicpm_injection" / "official_t4.py",
                   "--root", root]
        if mid_speech:
            command.append("--mid-speech")
        print("Loading the official model, then running injected and control conditions.", flush=True)
        print("Run folder:", root, flush=True)
        _logged_command(command, root / "console.log", env=child_env)
        report = json.loads((root / "combined_report.json").read_text())
    report["run_root"] = str(root)
    return report


def _table(headers, rows):
    def cell(value, tag="td"):
        text = "—" if value is None else str(value)
        return f"<{tag} style='padding:8px 12px;text-align:left;border-bottom:1px solid #ddd'>{html.escape(text)}</{tag}>"
    return ("<table style='border-collapse:collapse;width:100%'><thead><tr>"
            + "".join(cell(h, "th") for h in headers) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(cell(v) for v in row) + "</tr>" for row in rows)
            + "</tbody></table>")


def results_html(report, title):
    """Render conclusions and actual answers without exposing raw JSON."""
    results = report.get("results", [])
    paired_pass = len(results) == 2 and all(r.get("verdict") == "PASS" for r in results)
    conclusion = ("PASS · Both the injected condition and matched control met all checks."
                  if paired_pass else "Not established · Review the condition verdicts and answers below.")
    blocks = [f"<h3>{html.escape(title)}</h3><p><strong>{conclusion}</strong></p>"]
    blocks.append(_table(["Condition", "Verdict", "Text evaluated", "Same session", "Between speech chunks"], [
        ["Injected" if r.get("injected", index == 0) else "Control", r.get("verdict"),
         r.get("transport_verified"), r.get("session_continuity"),
         r.get("injection_between_audio_chunks") if r.get("mid_speech") else "Not required"]
        for index, r in enumerate(results)]))
    for index, result in enumerate(results):
        injected = result.get("injected", index == 0)
        condition = "Injected" if injected else "Control"
        old, new = result.get("old_code"), result.get("new_code")
        blocks.append(f"<h4>{condition}</h4><p>Initial room: <strong>{html.escape(str(old))}</strong>. "
                      + (f"Replacement supplied: <strong>{html.escape(str(new))}</strong>." if injected
                         else f"Replacement withheld: <strong>{html.escape(str(new))}</strong>.") + "</p>")
        answers = result.get("answers", {})
        blocks.append(_table(["Exchange", "Expected room", "Actual model answer"], [
            ["Before correction", old, answers.get("baseline") or "No completed answer"],
            ["After correction", new if injected else old, answers.get("followup") or "No completed answer"],
            ["Later recall", new if injected else old, answers.get("recall") or "No completed answer"],
        ]))
        if result.get("error"):
            blocks.append(f"<p><strong>Error:</strong> {html.escape(result['error'])}</p>")
    return "\n".join(blocks)


def show_results(report, title):
    from IPython.display import HTML, display
    display(HTML(results_html(report, title)))
    print("Raw evidence:", report["run_root"])


def show_timeline(report, condition=0):
    from IPython.display import HTML, display

    result = report["results"][condition]
    path = Path(result["run_directory"]) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    origin = events[0]["t"] if events else 0
    rows = []
    for event in events:
        kind = event["kind"]
        if kind == "context_applied":
            detail = f"{event['text']} · KV {event['kv_before']} → {event['kv_after']} · schema match: {event['schema_matches']}"
        elif kind == "output" and (event.get("text") or event.get("audio_samples")):
            detail = f"{event['text']} · speech samples: {event['audio_samples']} · turn {event['turn']}"
        elif kind in ("fact_arrival", "exchange_complete", "exchange_timeout", "session_start", "error"):
            detail = event.get("message", event.get("replacement_code", event.get("phase", "Session prepared once")))
        else:
            continue
        rows.append([round(event["t"] - origin, 3), event.get("phase", ""), kind, detail])
    display(HTML(_table(["Seconds", "Exchange", "Event", "Evidence"], rows)))


def play_answers(report, condition=0):
    """Join original speech chunks per exchange for three understandable players."""
    import wave
    from IPython.display import Audio, HTML, display

    directory = Path(report["results"][condition]["run_directory"])
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    for phase, label in (("baseline", "Before correction"), ("followup", "After correction"), ("recall", "Later recall")):
        paths = [directory / e["audio_path"] for e in events if e["kind"] == "output"
                 and e["phase"] == phase and e.get("audio_path")]
        if not paths:
            continue
        combined = directory / f"{phase}_answer.wav"
        with wave.open(str(combined), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 24000):
                        raise RuntimeError(f"Unexpected speech audio format: {path}")
                    output.writeframes(source.readframes(source.getnframes()))
        display(HTML(f"<p><strong>{label}</strong></p>"))
        display(Audio(filename=str(combined)))


def save_results_notebook(workspace, reports):
    """Save real report outputs into a shareable notebook, including on reruns."""
    template = workspace.repository / "notebooks" / "official_t4_context_injection_colab.ipynb"
    if not template.is_file():
        raise FileNotFoundError("The published implementation is missing its notebook template")
    notebook = json.loads(template.read_text())
    # The implementation commit contains the initial notebook template. Pin
    # exported copies to the code actually used, even after a notebook-only update.
    import re
    actual_commit = subprocess.check_output(
        ["git", "-C", str(workspace.repository), "rev-parse", "HEAD"], text=True).strip()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            source = re.sub(r'^EVIDENCE_COMMIT = "[^"]+"$', f'EVIDENCE_COMMIT = "{actual_commit}"',
                            source, flags=re.MULTILINE)
            cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["evidence"] = {
        "implementation_commit": actual_commit,
        "saved_at_unix": time.time(),
        "note": "Report tables from actual retained trials; other cells are the runnable experiment template.",
    }
    for cell in notebook["cells"]:
        key = cell.get("metadata", {}).get("evidence_result")
        if key in reports:
            report = reports[key]
            rendered = results_html(report, "Correction and recall" if key == "correction" else "Update during speech")
            cell["outputs"] = [dict(output_type="display_data", metadata={}, data={
                "text/html": rendered, "text/plain": json.dumps(report, indent=2)})]
            cell["execution_count"] = None
    path = workspace.root / "context_injection_results.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n")
    (workspace.root / "displayed_reports.json").write_text(json.dumps(reports, indent=2))
    print("Notebook with actual result tables saved:", path)
    return path


def export_evidence(workspace, reports):
    notebook_path = save_results_notebook(workspace, reports)
    archive = Path(shutil.make_archive(str(workspace.root / "all_t4_evidence"), "zip",
                                       root_dir=workspace.root, base_dir="runs"))
    return notebook_path, archive
