"""Run the two semantic context-injection experiments.

One native process owns one MiniCPM duplex session. Python supplies one-second
audio units and optional text at unit boundaries; it never edits model output.
Native callbacks record when text changes the LLM KV context and when TTS emits
audio. Those timestamps let the evaluator distinguish injection during speech
generation from injection before or after a response.
"""
from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .environment import Workspace


ROOM_CODES = ("C583", "D419", "G726", "J358", "K614", "M927", "R465", "T831", "V294")


def contains_code(text: str, code: str) -> bool:
    """Match one exact code while allowing spaces between streamed characters."""

    pattern = r"(?<![A-Za-z0-9])" + r"\s*".join(re.escape(char) for char in code)
    pattern += r"(?!\s*[A-Za-z0-9])"
    return bool(code) and re.search(pattern, text, re.IGNORECASE) is not None


def choose_code(excluding: str | None = None) -> str:
    choices = [code for code in ROOM_CODES if code != excluding]
    return secrets.choice(choices)


class NativeSession:
    """Persistent JSON-lines connection to one native MiniCPM duplex session."""

    def __init__(self, workspace: Workspace, label: str, seed: int = 42):
        if not workspace.bridge.is_file():
            raise FileNotFoundError("Build the CUDA runtime before starting an experiment")
        workspace.runs.mkdir(parents=True, exist_ok=True)
        self.lock_file = (workspace.runs / ".native.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            raise RuntimeError("Another native model session is still running")

        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_{uuid.uuid4().hex[:8]}"
        self.directory = workspace.runs / run_id
        self.directory.mkdir(parents=True)
        self.events: list[dict] = []
        self.condition = threading.Condition()
        self.command_lock = threading.Lock()
        self.sequence = 0
        self.closed = False
        self.host_log = (self.directory / "host_events.jsonl").open("w")
        self.native_log = (self.directory / "native.log").open("w")
        command = [str(workspace.bridge), str(workspace.models), str(self.directory), str(seed), "4096"]
        manifest = {
            "command": command,
            "run_id": run_id,
            "runtime_commit": subprocess.check_output(
                ["git", "-C", str(workspace.runtime), "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "-C", str(workspace.runtime), "diff"])
            ).hexdigest(),
            "bridge_sha256": hashlib.sha256(workspace.bridge.read_bytes()).hexdigest(),
            "seed": seed,
            "context_size": 4096,
        }
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self.process = subprocess.Popen(
            command,
            cwd=workspace.runtime,
            stdin=subprocess.PIPE,
            stdout=self.native_log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_native_events, daemon=True)
        self.reader.start()
        atexit.register(self.close)
        print(f"Loading one audio-only MiniCPM session. Native log: {self.directory/'native.log'}")
        try:
            ready = self.wait_for(lambda event: event["kind"] == "ready", timeout=600)
            if ready["vision_loaded"] or not ready["tts"]:
                raise RuntimeError("Expected vision disabled and TTS enabled")
        except BaseException:
            self.close()
            raise
        print("Session ready:", run_id)

    def record(self, kind: str, **data: object) -> dict:
        event = {"t": time.time(), "kind": kind, **data}
        with self.condition:
            self.events.append(event)
            self.host_log.write(json.dumps(event) + "\n")
            self.host_log.flush()
            self.condition.notify_all()
        return event

    def _read_native_events(self) -> None:
        path = self.directory / "native_events.jsonl"
        while not path.exists() and self.process.poll() is None:
            time.sleep(0.05)
        if not path.exists():
            return
        with path.open() as source:
            while True:
                line = source.readline()
                if line:
                    event = json.loads(line)
                    self.record(event.pop("kind"), **event)
                elif self.process.poll() is not None:
                    break
                else:
                    time.sleep(0.02)
        with self.condition:
            self.condition.notify_all()

    def snapshot(self) -> list[dict]:
        with self.condition:
            return list(self.events)

    def wait_for(self, predicate: Callable[[dict], bool], timeout: float = 120) -> dict:
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                for event in self.events:
                    if event["kind"] == "error":
                        raise RuntimeError(event["message"])
                    if predicate(event):
                        return event
                if self.process.poll() is not None:
                    raise RuntimeError(f"Native process exited; inspect {self.directory/'native.log'}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for native evidence")
                self.condition.wait(min(remaining, 0.5))

    def send_audio_unit(self, audio: Path, *, text: str = "", phase: str = "") -> dict:
        """Submit one audio unit and optional text to the next native boundary."""

        with self.command_lock:
            self.sequence += 1
            command = {"op": "frame", "audio": str(audio.resolve()), "text": text, "phase": phase}
            self.record(
                "submit",
                frame=self.sequence,
                phase=phase,
                text=text,
                audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
            )
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
            result = self.wait_for(
                lambda event: event["kind"] == "frame_done" and event["frame"] == self.sequence
            )
            if not result["ok"]:
                raise RuntimeError("Native frame inference failed")
            if result["text"]:
                print(f"  {phase}, unit {result['frame']}: {result['text']}")
            return result

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.process.poll() is None:
                try:
                    assert self.process.stdin is not None
                    self.process.stdin.write('{"op":"stop"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=90)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.reader.join(timeout=5)
            self.record("process_exit", return_code=self.process.returncode)
        finally:
            self.closed = True
            if self.process.stdin:
                self.process.stdin.close()
            self.native_log.close()
            self.host_log.close()
            self.lock_file.close()
            atexit.unregister(self.close)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16_000) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def _question_units(session: NativeSession, question: str, phase: str) -> tuple[list[Path], Path]:
    """Synthesize a repeatable question and split it into one-second units."""

    directory = session.directory / "input" / phase
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "question.wav"
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    ffmpeg = shutil.which("ffmpeg")
    if not espeak or not ffmpeg:
        raise RuntimeError("The experiment requires espeak-ng and ffmpeg")
    subprocess.run([espeak, "-w", str(source), question], check=True)
    normalized = directory / "question_16k.wav"
    subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", str(normalized)],
        check=True,
    )
    with wave.open(str(normalized), "rb") as audio:
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
    units: list[Path] = []
    for index, start in enumerate(range(0, len(samples), 16_000)):
        unit = np.zeros(16_000, dtype=np.float32)
        source_unit = samples[start : start + 16_000]
        unit[: len(source_unit)] = source_unit
        path = directory / f"input_{index:04d}.wav"
        _write_wav(path, unit)
        units.append(path)
    silence = directory / "silence.wav"
    _write_wav(silence, np.zeros(16_000, dtype=np.float32))
    session.record(
        "scripted_question",
        phase=phase,
        question=question,
        units=len(units),
        wav_sha256=hashlib.sha256(normalized.read_bytes()).hexdigest(),
    )
    return units, silence


def run_turn(
    session: NativeSession,
    question: str,
    phase: str,
    *,
    text_at_boundary: Callable[[NativeSession], str] | None = None,
    maximum_units: int = 60,
) -> dict:
    """Feed a question at one unit per second until one spoken response finishes."""

    question_units, silence = _question_units(session, question, phase)
    started = time.time()
    old_final_audio = {
        event["audio_id"]
        for event in session.snapshot()
        if event["kind"] == "audio" and event.get("final")
    }
    session.record("turn_start", phase=phase)
    response_seen = False
    listening_units = 0
    frame_results = []
    for index in range(len(question_units) + maximum_units):
        due = started + index
        time.sleep(max(0, due - time.time()))
        text = text_at_boundary(session) if text_at_boundary else ""
        session.record("input_due", phase=phase, due=due, lag_s=max(0, time.time() - due))
        audio = question_units[index] if index < len(question_units) else silence
        result = session.send_audio_unit(audio, text=text, phase=phase)
        frame_results.append(result)
        response_seen = response_seen or result["speak"]
        listening_units = listening_units + 1 if response_seen and not result["speak"] else 0
        new_final_audio = any(
            event["kind"] == "audio"
            and event.get("final")
            and event["audio_id"] not in old_final_audio
            for event in session.snapshot()
        )
        if index >= len(question_units) and listening_units >= 2 and new_final_audio:
            session.record("turn_complete", phase=phase)
            return {
                "complete": True,
                "response_seen": response_seen,
                "text": "".join(frame["text"] for frame in frame_results),
            }
        if index % 5 == 0 and not result["text"]:
            state = "SPEAK" if result["speak"] else "LISTEN"
            print(f"  {phase}: {index + 1} units processed; state={state}")
    # Audio synthesis runs in its own native worker. If the language model has
    # already returned to LISTEN, allow a short bounded drain for the final TTS
    # callback instead of misclassifying a nearly completed exchange.
    if response_seen and listening_units >= 2:
        try:
            session.wait_for(
                lambda event: event["kind"] == "audio"
                and event.get("final")
                and event["audio_id"] not in old_final_audio,
                timeout=30,
            )
            session.record("turn_complete", phase=phase, completion="tts_drain_grace")
            return {
                "complete": True,
                "response_seen": True,
                "text": "".join(frame["text"] for frame in frame_results),
            }
        except TimeoutError:
            pass
    session.record("turn_timeout", phase=phase)
    return {
        "complete": False,
        "response_seen": response_seen,
        "text": "".join(frame["text"] for frame in frame_results),
    }


def injection_overlapped_tts(events: list[dict], payload: str) -> bool:
    """Require successful text evaluation between two chunks of one TTS turn."""

    acknowledgements = [
        event
        for event in events
        if event["kind"] == "context_applied"
        and event["text"] == payload
        and event["ok"]
        and event["kv_after"] > event["kv_before"]
    ]
    if not acknowledgements:
        return False
    applied_at = acknowledgements[0]["t"]
    earlier_audio = [event for event in events if event["kind"] == "audio" and event["t"] < applied_at]
    if not earlier_audio or earlier_audio[-1]["final"]:
        return False
    speech_turn = earlier_audio[-1]["turn"]
    return any(
        event["kind"] == "audio" and event["turn"] == speech_turn and event["t"] > applied_at
        for event in events
    )


def _save_result(session: NativeSession, result: dict) -> dict:
    result["run_directory"] = str(session.directory)
    (session.directory / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def run_mid_generation_pair(workspace: Workspace, seed: int = 42) -> list[dict]:
    """Run an injected trial and a matched trial that withholds the correction."""

    old_code = "B742"
    new_code: str | None = None
    results = []
    for injected in (True, False):
        condition = "injected" if injected else "control"
        session = NativeSession(workspace, f"mid_generation_{condition}", seed)
        state = {"initial_sent": False, "correction_sent": False, "payload": ""}
        try:
            def boundary(current: NativeSession) -> str:
                nonlocal new_code
                if not state["initial_sent"]:
                    state["initial_sent"] = True
                    current.record("initial_fact", value=old_code)
                    return f"Room={old_code}."
                if state["correction_sent"]:
                    return ""
                audio = [event for event in current.snapshot() if event["kind"] == "audio"]
                if audio and not audio[-1]["final"]:
                    state["correction_sent"] = True
                    if injected:
                        new_code = choose_code(excluding=old_code)
                        state["payload"] = f"Room {old_code} is cancelled. Current room={new_code}."
                        current.record("new_fact_arrived", value=new_code)
                        return state["payload"]
                    current.record("control_boundary", withheld_code=new_code)
                return ""

            first = run_turn(
                session,
                "What is the assigned room? Please answer with only the room code.",
                "response_during_update",
                text_at_boundary=boundary,
            )
            followup = run_turn(
                session,
                "What is the current assigned room? Answer with only its code.",
                "followup",
            )
            events = session.snapshot()
            timing_proved = injection_overlapped_tts(events, state["payload"]) if injected else None
            new_seen = bool(new_code) and contains_code(followup["text"], new_code)
            old_seen = contains_code(followup["text"], old_code)
            prerequisite = state["correction_sent"] and new_code is not None
            if injected:
                tested = prerequisite and timing_proved
                semantic_success = new_seen and not old_seen
            else:
                tested = prerequisite and followup["response_seen"]
                semantic_success = old_seen and not new_seen
            verdict = ("PASS" if semantic_success else "FAIL") if tested else "NOT TESTED"
            results.append(
                _save_result(
                    session,
                    {
                        "test": "mid_generation_context_update",
                        "condition": condition,
                        "verdict": verdict,
                        "old_code": old_code,
                        "new_code": new_code,
                        "injection_triggered": prerequisite,
                        "injection_between_tts_chunks": timing_proved,
                        "first_turn_complete": first["complete"],
                        "followup_complete": followup["complete"],
                        "followup_text": followup["text"],
                        "new_code_in_followup": new_seen,
                        "old_code_in_followup": old_seen,
                    },
                )
            )
        except Exception as error:
            results.append(
                _save_result(
                    session,
                    {"test": "mid_generation_context_update", "condition": condition, "verdict": "ERROR", "error": str(error)},
                )
            )
        finally:
            session.close()
    return results


def run_multi_turn_update(workspace: Workspace, seed: int = 42) -> dict:
    """Inject a fact, replace it next turn, then test recall without reinjection."""

    first_code = choose_code()
    replacement = choose_code(excluding=first_code)
    session = NativeSession(workspace, "multi_turn_update", seed)

    def inject_once(payload: str) -> Callable[[NativeSession], str]:
        pending = [payload]

        def boundary(current: NativeSession) -> str:
            if not pending:
                return ""
            current.record("fact_arrived", payload=pending[0])
            return pending.pop()

        return boundary

    try:
        initial = run_turn(
            session,
            "What is the assigned room? Please answer with only the room code.",
            "initial",
            text_at_boundary=inject_once(f"Room={first_code}."),
        )
        if not initial["complete"]:
            return _save_result(
                session,
                {"test": "multi_turn_update", "verdict": "NOT TESTED", "reason": "Initial exchange did not finish"},
            )
        session.record("correction_created_after_initial_turn", old=first_code, new=replacement)
        corrected = run_turn(
            session,
            "What is the assigned room now? Answer with only the current room code.",
            "correction",
            text_at_boundary=inject_once(
                f"The previous room {first_code} is cancelled. Current room={replacement}."
            ),
        )
        if not corrected["complete"]:
            return _save_result(
                session,
                {"test": "multi_turn_update", "verdict": "NOT TESTED", "reason": "Correction exchange did not finish"},
            )
        recall = run_turn(
            session,
            "Remind me of the current assigned room. Answer with only its code.",
            "recall_without_injection",
        )
        events = session.snapshot()
        acknowledgements = [
            event
            for event in events
            if event["kind"] == "context_applied"
            and event["ok"]
            and event["kv_after"] > event["kv_before"]
        ]
        checks = {
            "initial_answer_used_first_code": contains_code(initial["text"], first_code),
            "correction_answer_used_only_replacement": contains_code(corrected["text"], replacement)
            and not contains_code(corrected["text"], first_code),
            "later_recall_used_only_replacement": contains_code(recall["text"], replacement)
            and not contains_code(recall["text"], first_code),
            "two_context_evaluations_changed_kv": len(acknowledgements) == 2,
            "one_persistent_session": sum(event["kind"] == "ready" for event in events) == 1,
        }
        verdict = ("PASS" if all(checks.values()) else "FAIL") if recall["complete"] else "NOT TESTED"
        return _save_result(
            session,
            {
                "test": "multi_turn_update",
                "verdict": verdict,
                "first_code": first_code,
                "replacement_code": replacement,
                "checks": checks,
                "responses": {
                    "initial": initial["text"],
                    "correction": corrected["text"],
                    "recall": recall["text"],
                },
            },
        )
    except Exception as error:
        return _save_result(
            session,
            {"test": "multi_turn_update", "verdict": "ERROR", "error": str(error)},
        )
    finally:
        session.close()
