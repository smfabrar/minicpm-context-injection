"""Context-injection evidence through the official MiniCPM-o GPTQ duplex API.

Runs in a separate Python 3.11 environment on Colab. No C++ runtime or patch.
Imports of GPU/audio libraries are deferred so evaluators can be tested locally.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import secrets
import subprocess
import time
import traceback
from pathlib import Path

MODEL_ID = "openbmb/MiniCPM-o-4_5-GPTQ"
MODEL_REVISION = "02b54c54c36f8b48e97501bb0fde5d178c62df38"
SYSTEM_PROMPT = (
    "Streaming Audio Conversation. You are a helpful assistant. "
    "External context supplies the current room assignment. A newer assignment "
    "replaces the older one. When asked for the room, spell its letter and digits."
)


def contains_code(text, code):
    pattern = r"(?<![A-Za-z0-9])" + r"\s*".join(re.escape(c) for c in code)
    return bool(code) and re.search(pattern + r"(?!\s*[A-Za-z0-9])", text, re.I) is not None


def evaluate(events, old_code, new_code, injected, mid_speech):
    """Separate transport, timing, baseline and semantic evidence."""
    contexts = [e for e in events if e["kind"] == "context_applied"]
    transport = bool(contexts) and all(
        e["success"] and e["kv_after"] - e["kv_before"] == e["expected_kv_growth"]
        and e["schema_matches"] for e in contexts
    )
    updates = [e for e in contexts if e["phase"] == "update"]
    overlap = any(
        a["kind"] == "output" and a["audio_samples"] > 0 and not a["is_listen"]
        and not a["end_of_turn"] and a["seq"] < update["seq"] < b["seq"]
        and b["kind"] == "output" and b["audio_samples"] > 0 and not b["is_listen"]
        and a["turn"] == b["turn"]
        for update in updates for a in events for b in events
    )
    answers = {
        phase: "".join(e["text"] for e in events if e["kind"] == "output" and e["phase"] == phase)
        for phase in ("baseline", "followup", "recall")
    }
    completed = {e["phase"] for e in events if e["kind"] == "exchange_complete"}
    baseline_ok = contains_code(answers["baseline"], old_code)
    expected = new_code if injected else old_code
    forbidden = old_code if injected else new_code
    semantic = all(contains_code(answers[p], expected) and not contains_code(answers[p], forbidden)
                   for p in ("followup", "recall"))
    continuity = len([e for e in events if e["kind"] == "session_start"]) == 1
    verdict = "PASS"
    if not continuity or not transport or (injected and not updates):
        verdict = "ERROR"
    elif not {"baseline", "followup", "recall"}.issubset(completed):
        verdict = "NOT TESTED"
    elif mid_speech and not overlap:
        verdict = "NOT TESTED"
    elif not baseline_ok or not semantic:
        verdict = "FAIL"
    return dict(verdict=verdict, transport_verified=transport, session_continuity=continuity,
                injection_between_audio_chunks=overlap, baseline_verified=baseline_ok,
                semantic_verified=semantic, answers=answers, old_code=old_code,
                new_code=new_code, injected=injected, mid_speech=mid_speech)


class EvidenceSession:
    def __init__(self, duplex, directory, ref_audio, ref_path):
        self.duplex = duplex
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        (self.directory / "audio").mkdir()
        self.events = []
        self.turn = 0
        self.ref_path = str(ref_path)
        self.log = (self.directory / "events.jsonl").open("w")
        duplex.prepare(prefix_system_prompt=SYSTEM_PROMPT, ref_audio=ref_audio,
                       prompt_wav_path=self.ref_path)
        self.record("session_start", decoder_id=id(duplex.decoder))

    def record(self, kind, **data):
        event = dict(seq=len(self.events), t=time.time(), kind=kind, **data)
        self.events.append(event)
        self.log.write(json.dumps(event) + "\n")
        self.log.flush()
        return event

    def step(self, phase, *, audio=None, text=None, output_phase=None):
        import numpy as np
        import soundfile as sf
        import torch

        # Separate text-only units avoid the official mixed-mode path's audio
        # logits, which are computed before the new text is appended.
        if text is not None and audio is not None:
            raise ValueError("Use a separate text-only unit for context injection")
        before = self.duplex.decoder.get_cache_length()
        expected_ids = self.duplex.tokenizer.encode(text, add_special_tokens=False) if text else []
        self.record("input", phase=phase, text=text, audio_samples=0 if audio is None else len(audio))
        torch.cuda.synchronize()
        start = time.monotonic()
        prefill = self.duplex.streaming_prefill(audio_waveform=audio, frame_list=[],
                                               text_list=[text] if text is not None else None)
        torch.cuda.synchronize()
        after = self.duplex.decoder.get_cache_length()
        if text is not None:
            schema = self.duplex.prefill_schema_tokens[-1] if prefill["success"] else []
            self.record("context_applied", phase=phase, text=text, success=bool(prefill["success"]),
                        kv_before=before, kv_after=after, expected_kv_growth=1 + len(expected_ids),
                        schema_matches=schema == [self.duplex.unit_token_id] + expected_ids,
                        seconds=time.monotonic() - start)
        if not prefill["success"]:
            raise RuntimeError(f"Official streaming_prefill failed: {prefill}")
        result = self.duplex.streaming_generate(prompt_wav_path=self.ref_path,
                                               max_new_speak_tokens_per_chunk=20,
                                               decode_mode="greedy")
        torch.cuda.synchronize()
        waveform = result["audio_waveform"]
        # Listening outputs contain synthetic silence and are not TTS evidence.
        samples = 0 if waveform is None or result["is_listen"] else int(np.asarray(waveform).size)
        audio_path = None
        if samples:
            audio_path = f"audio/{len(self.events):04d}.wav"
            sf.write(self.directory / audio_path, np.asarray(waveform), 24000)
        event = self.record("output", phase=output_phase or phase, turn=self.turn, text=result["text"],
                            is_listen=bool(result["is_listen"]), end_of_turn=bool(result["end_of_turn"]),
                            audio_samples=samples, audio_path=audio_path,
                            seconds=time.monotonic() - start, kv_length=self.duplex.decoder.get_cache_length())
        print(f"{phase}: {'listen' if result['is_listen'] else result['text']}", flush=True)
        if result["end_of_turn"] and not result["is_listen"]:
            self.turn += 1
        return event

    def exchange(self, phase, question_audio, after_chunk=None, max_units=50):
        import numpy as np

        outputs = []
        units = [question_audio[i:i + 16000] for i in range(0, len(question_audio), 16000)]
        units = [np.pad(unit, (0, 16000 - len(unit))) for unit in units]
        for index in range(len(units) + max_units):
            audio = units[index] if index < len(units) else np.zeros(16000, dtype=np.float32)
            event = self.step(phase, audio=audio)
            outputs.append(event)
            if after_chunk and event["audio_samples"] and not event["end_of_turn"]:
                outputs.append(after_chunk())
                after_chunk = None
            # Require a completed speaking turn after all question units arrived.
            latest = outputs[-1]
            if index >= len(units) - 1 and latest["end_of_turn"] and not latest["is_listen"]:
                self.record("exchange_complete", phase=phase)
                return True
        self.record("exchange_timeout", phase=phase)
        return False


def synthesize_question(text, path):
    import librosa
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "145", "-w", str(path), text], check=True)
    waveform, _ = librosa.load(str(path), sr=16000, mono=True)
    return waveform


def run_condition(duplex, root, ref_audio, ref_path, *, old_code, new_code=None,
                  injected=True, mid_speech=False, seed=42):
    import torch
    label = ("mid_speech" if mid_speech else "between_turns") + ("_injected" if injected else "_control")
    directory = Path(root) / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}_{secrets.token_hex(3)}"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    session = EvidenceSession(duplex, directory, ref_audio, ref_path)
    manifest = dict(model_id=MODEL_ID, model_revision=MODEL_REVISION, seed=seed,
                    condition=label, dtype="float16", init_vision=False, attention="sdpa",
                    sliding_window="off", decoder_source_sha256=hashlib.sha256(
                        Path(inspect.getfile(type(duplex))).read_bytes()).hexdigest())
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        session.step("initial", text=f"External context: current room={old_code}. Do not answer yet.")
        baseline = synthesize_question(
            "What is my assigned room? Spell the code, then explain in several sentences how to find the room.",
            directory / "baseline_question.wav")
        question = synthesize_question("What is my current room? Spell only the room code.",
                                       directory / "followup_question.wav")

        def update():
            nonlocal new_code
            if new_code is None:
                new_code = secrets.choice("CDGJKMRTV") + str(secrets.randbelow(900) + 100)
                while new_code == old_code:
                    new_code = secrets.choice("CDGJKMRTV") + str(secrets.randbelow(900) + 100)
            session.record("fact_arrival", phase="update", replacement_code=new_code,
                           withheld=not injected)
            payload = (f"External context: {old_code} is cancelled. Current room={new_code}."
                       if injected else "External context: the current room assignment is unchanged.")
            return session.step("update", text=payload, output_phase="baseline" if mid_speech else "update")

        completed = session.exchange("baseline", baseline, after_chunk=update if mid_speech else None)
        if completed:
            if not mid_speech:
                update()
            session.exchange("followup", question)
            session.exchange("recall", question)
        result = evaluate(session.events, old_code, new_code or "WITHHELD", injected, mid_speech)
    except Exception as error:
        session.record("error", message=str(error), traceback=traceback.format_exc())
        result = dict(verdict="ERROR", error=str(error), old_code=old_code, new_code=new_code)
    finally:
        session.log.close()
    result["run_directory"] = str(directory)
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mid-speech", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    import importlib.metadata
    import librosa
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, GPTQConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Select an NVIDIA GPU runtime in Colab")
    print("GPU:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0), flush=True)
    # Download the vocoder and reference voice at the same revision as the model.
    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                  ignore_patterns=["*.mp4", "*.mp3", "*.png", "*.md"],
                                  token=None)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.float16, device_map={"": 0},
        init_vision=False, init_audio=True, init_tts=True,
        quantization_config=GPTQConfig(bits=4, group_size=128, desc_act=False, sym=True,
                                      use_exllama=False, block_name_to_quantize="llm.model.layers"),
    ).eval()
    # All model modules stay on GPU; automatic CPU offload can break duplex TTS.
    duplex = model.as_duplex(enable_float16=True, sliding_window_mode="off")
    if "text_list" not in inspect.signature(duplex.streaming_prefill).parameters:
        raise RuntimeError("This model revision does not expose duplex text_list")
    if hasattr(model, "vpm"):
        raise RuntimeError("Vision unexpectedly loaded")
    quantized = [name for name, module in model.named_modules() if hasattr(module, "qweight")]
    if not quantized:
        raise RuntimeError("No GPTQ quantized layers loaded")
    ref_path = Path(model_dir) / "assets" / "HT_ref_audio.wav"
    ref_audio, _ = librosa.load(str(ref_path), sr=16000, mono=True)
    environment = dict(gpu=torch.cuda.get_device_name(0), model_revision=MODEL_REVISION,
                       quantized_layers=len(quantized), memory_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                       packages={name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "gptqmodel", "minicpmo-utils", "optimum")})
    (args.root / "environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment, indent=2), flush=True)
    old_code = "B" + str(secrets.randbelow(900) + 100)
    injected = run_condition(duplex, args.root, ref_audio, ref_path, old_code=old_code,
                             mid_speech=args.mid_speech)
    results = [injected]
    if injected.get("new_code"):
        results.append(run_condition(duplex, args.root, ref_audio, ref_path, old_code=old_code,
                                     new_code=injected["new_code"], injected=False, mid_speech=args.mid_speech))
    report = dict(environment=environment, results=results,
                  paired_pass=len(results) == 2 and all(r["verdict"] == "PASS" for r in results),
                  boundary="Recorded audio; injection between generated chunks. Browser microphone/playback overlap and real-time throughput are not tested.")
    (args.root / "combined_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
