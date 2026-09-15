"""Audit retained evidence and build a machine-readable combined report."""
from __future__ import annotations

import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Iterable

from .environment import Workspace


def _read_json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_results(results: Iterable[dict]) -> list[dict]:
    """Verify each result against its raw event trace and summarize latency."""

    audits = []
    for result in results:
        directory = Path(result["run_directory"])
        required = (
            directory / "manifest.json",
            directory / "result.json",
            directory / "host_events.jsonl",
            directory / "native_events.jsonl",
            directory / "native.log",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing evidence files: {missing}")
        events = _read_json_lines(directory / "host_events.jsonl")
        ready = [event for event in events if event["kind"] == "ready"]
        if len(ready) != 1:
            raise RuntimeError(f"Expected one persistent session in {directory}")
        if ready[0]["vision_loaded"] is not False:
            raise RuntimeError(f"Vision was unexpectedly loaded in {directory}")
        context_events = [
            event
            for event in events
            if event["kind"] == "context_applied"
            and event["ok"]
            and event["kv_after"] > event["kv_before"]
        ]
        latencies = [
            float(event["latency_ms"])
            for event in events
            if event["kind"] == "frame_done" and "latency_ms" in event
        ]
        sorted_latencies = sorted(latencies)
        p95_index = min(len(sorted_latencies) - 1, int(0.95 * len(sorted_latencies)))
        audits.append(
            {
                "test": result.get("test"),
                "condition": result.get("condition", "single session"),
                "verdict": result["verdict"],
                "successful_context_evaluations": len(context_events),
                "frames_processed": len(latencies),
                "median_frame_latency_ms": statistics.median(latencies) if latencies else None,
                "p95_frame_latency_ms": sorted_latencies[p95_index] if latencies else None,
                "vision_loaded": False,
                "run_directory": str(directory),
            }
        )
    return audits


def write_combined_report(
    workspace: Workspace,
    *,
    hardware: dict,
    model: dict,
    runtime: dict,
    build: dict,
    results: list[dict],
) -> Path:
    """Write conclusions together with configuration and raw-run locations."""

    report = {
        "created_at_unix": time.time(),
        "question": "Can text arriving after session start affect MiniCPM-o during native duplex speech?",
        "hardware": hardware,
        "model": model,
        "runtime": runtime,
        "build": build,
        "vision": "disabled and not downloaded",
        "results": results,
        "artifact_audits": audit_results(results),
        "evidence_boundaries": {
            "native_duplex_stream_and_tts": "tested",
            "text_evaluation_in_persistent_kv_context": "tested",
            "semantic_use_of_injected_fact": "tested by exact-code follow-up",
            "physical_browser_microphone_and_speaker_overlap": "NOT TESTED in hosted Colab",
        },
    }
    path = workspace.root / "combined_report.json"
    path.write_text(json.dumps(report, indent=2))
    print("Combined report:", path)
    return path


def show_run(run_directory: str | Path) -> None:
    """Display decisive events and play every native TTS chunk in a notebook."""

    import numpy as np
    import pandas as pd
    from IPython.display import Audio, display

    directory = Path(run_directory)
    events = _read_json_lines(directory / "host_events.jsonl")
    important = {
        "ready",
        "initial_fact",
        "new_fact_arrived",
        "fact_arrived",
        "correction_created_after_initial_turn",
        "context_applied",
        "audio",
        "turn_start",
        "turn_complete",
        "turn_timeout",
    }
    origin = events[0]["t"] if events else 0
    rows = []
    for event in events:
        if event["kind"] in important or (event["kind"] == "frame_done" and event.get("text")):
            details = {key: value for key, value in event.items() if key not in ("t", "kind", "path")}
            rows.append(
                {"seconds": round(event["t"] - origin, 3), "event": event["kind"], "details": details}
            )
    display(pd.DataFrame(rows))

    native_events = _read_json_lines(directory / "native_events.jsonl")
    for event in native_events:
        if event["kind"] == "audio":
            print(f"TTS turn {event['turn']}, chunk {event['audio_id']}, final={event['final']}")
            display(Audio(np.fromfile(event["path"], dtype=np.float32), rate=event["sample_rate"]))


def archive_runs(workspace: Workspace) -> Path:
    """Create one downloadable ZIP containing every retained condition."""

    archive = Path(
        shutil.make_archive(
            str(workspace.root / "minicpm_context_injection_runs"),
            "zip",
            root_dir=workspace.root,
            base_dir="runs",
        )
    )
    print("Runs archive:", archive)
    return archive
