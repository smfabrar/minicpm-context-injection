# MiniCPM-o context injection during duplex speech

For a **Colab T4 without a native CUDA build**, use
[`notebooks/official_t4_context_injection_colab.ipynb`](notebooks/official_t4_context_injection_colab.ipynb).
It loads OpenBMB's [official GPTQ 4-bit model](https://huggingface.co/openbmb/MiniCPM-o-4_5-GPTQ)
through Transformers, disables vision, uses FP16 and SDPA, and installs wheels in
an isolated Python 3.11 environment. Large code lives in the repository; the
notebook contains short cells and explanations. The official card estimates about 11 GB GPU
memory; duplex/TTS headroom and throughput still require measurement on the T4.
This setup has not yet been validated by a GPU run.

The notebook uses the official duplex `streaming_prefill(text_list=[...])` API.
Separate text-only units avoid the mixed audio/text path's pending logits being
computed before text is appended. It checks exact text-token schema and KV
growth, baseline recall, corrected recall across two later exchanges, and a
matched control. An additional experiment inserts the update between generated
audio chunks of the same turn. Each condition prepares its session once.
Dependency, GPU-loading and runtime failures are preserved as errors. Native
dependencies use wheels; `minicpmo-utils==1.0.6` is allowed to package its pure
Python source because that release has no published wheel. No CUDA compilation
is requested. The notebook clones a published implementation commit, displays
verdict and answer tables plus speech players, and saves
`context_injection_results.ipynb` with actual result outputs after each experiment.
Regenerate it with `python tools/make_t4_notebook.py` (requires `nbformat`).

The native GGUF experiment below remains available as a separate implementation.

This repository asks one narrow question:

> Can new text be inserted into an already-running MiniCPM-o 4.5 conversation,
> while its duplex speech pipeline is active, and affect a later answer?

The experiment uses the audio-only GGUF model and the native
[`llama.cpp-omni`](https://github.com/tc-mb/llama.cpp-omni) runtime. Vision is
not downloaded or loaded.

## What “context injection” means here

MiniCPM normally receives one-second audio units. The runtime encodes each unit,
adds it to the language model's persistent key/value (KV) context, and decides
whether to listen or speak. This experiment adds one optional text field to an
audio unit:

```text
microphone/recorded audio ──> audio encoder ──> persistent LLM context ──> text/TTS
                                                   ▲
                                                   │
                                  new text at an audio-unit boundary
```

The injected text is evaluated by the same LLM worker that owns the conversation.
The model process and conversation are not restarted. A native callback records
the KV position before and after evaluation. A larger KV position proves that
the tokens were evaluated; it does not prove that the model understood or used
them. The follow-up answer is the separate semantic test.

## The two experiments

### A. Update during generated speech

1. Start one native duplex session and inject an initial assignment,
   `Room=B742`.
2. Ask for the room using generated 16 kHz speech.
3. Wait until native TTS emits a non-final audio chunk. Only then choose a new,
   unpredictable room code.
4. Submit “B742 is cancelled; the current room is NEW_CODE” with the next
   one-second input unit.
5. Require a successful text-evaluation callback between two TTS chunks from the
   same speech turn.
6. Ask for the current room later in the same session.
7. Run a matched control with the same seed and questions while withholding the
   new code.

The injected condition passes only when the timing condition occurs and the
follow-up contains the complete new code without the cancelled code. The control
passes only when it stays with the old code and does not produce the withheld
code.

### B. Correction and recall across turns

1. Inject an initial random room and complete an exchange.
2. Create and inject a different room assignment after that exchange finishes.
3. Ask for the current room.
4. Ask again in a later turn without another injection.

This experiment passes only if both context evaluations change the KV position,
the correction answer and later recall use only the replacement code, and all
three exchanges use one persistent native session.

## What this does and does not test

| Claim | Evidence in this repository |
|---|---|
| Text reached the native LLM context | Checked callback with KV positions before/after |
| Injection occurred during TTS generation | Callback timestamp falls between chunks of one TTS turn |
| Model used the new fact | Exact code in a later answer, cancelled code absent |
| Code was not produced without injection | Matched control condition |
| Update survived another turn | Correction and recall experiment |
| Vision stayed off | Model allow-list plus native `vision_loaded=false` event |
| Browser microphone and speaker overlapped | **Not tested by hosted Colab** |

Colab runs the native duplex model and TTS queues on a remote GPU. Its Python
process cannot directly open the browser's microphone and speaker as PortAudio
devices. Testing physical input/output overlap requires a browser audio transport
such as WebRTC or WebSocket, or the separate local-device notebook. This boundary
does not prevent Colab from testing native stream ordering and semantic uptake.

## Run in Google Colab

Open [`notebooks/context_injection_colab.ipynb`](notebooks/context_injection_colab.ipynb)
in Colab and select **Runtime → Change runtime type → NVIDIA GPU**. Then run cells
from top to bottom.

The model repository is public. If Hugging Face requests authentication, create
a Colab secret called `HF_TOKEN`; the notebook reads it through Colab's secret
API and never prints it.

The first CUDA build uses the runtime's attached-GPU mode:

```text
GGML_CUDA=ON
GGML_METAL=OFF
GGML_NATIVE=ON
CMAKE_CUDA_ARCHITECTURES=native
```

`GGML_NATIVE=ON` matters on Colab. Turning it off makes the runtime compile CUDA
kernels for many architectures, which is much slower on the usual two CPU cores.
The build function prints a heartbeat every 20 seconds, saves `build.log`, and
uses `ccache` for faster reruns. The GPU is normally idle during compilation;
CUDA accelerates inference after the binary exists.

The audio-only weights use about 7.2 GiB of disk. A 16 GB T4 should normally fit
the Q4 model, while L4 or A100 provides more memory and speed. Colab hardware is
assigned dynamically, so the notebook records the actual device.

## Reading a result

Each condition receives one of three verdicts:

- `PASS`: every stated prerequisite occurred and the exact semantic expectation
  was met.
- `FAIL`: the prerequisite occurred, but the answer did not meet the semantic
  expectation.
- `NOT TESTED`: a required event never happened—for example, the model did not
  start speaking, so an update could not be inserted between TTS chunks.

An `ERROR` indicates a runtime, model, or harness failure and includes the log
location. Every run is retained; the notebook never selects only a successful
trial.

Artifacts are written under the Colab workspace:

```text
runs/<run-id>/
├── manifest.json          exact binary, runtime, seed, and context settings
├── host_events.jsonl      ordered host and native events
├── native_events.jsonl    events written directly by the C++ process
├── native.log             complete native stdout/stderr
├── input/                 generated questions split into one-second units
├── audio/                 raw float32 chunks produced by native TTS
└── result.json            condition-specific checks and verdict
```

`combined_report.json` records the GPU, model revision, runtime commit, patch
hash, build settings, results, latency summary, and evidence boundaries.

## Repository layout

```text
minicpm_injection/environment.py   model download, pinned runtime, CUDA build
minicpm_injection/experiment.py    persistent session and experimental protocol
minicpm_injection/reporting.py     artifact audit and combined report
native/native_session.cpp          small JSON-lines bridge to llama.cpp-omni
patches/context-injection.patch    reviewed runtime transport/callback change
tests/test_logic.py                exact-code and event-order evaluator tests
notebooks/                         readable Colab entry point
```

The evaluator tests use constructed events and verify only the verdict logic.
They are not model evidence. Actual evidence comes only from fresh native runs.

## Interpreting feasibility

One passing injected trial establishes that the mechanism is possible in that
trial. It does not establish reliability. Reliability requires repeated paired
trials across codes and seeds, with every failure retained and the success rate
reported. GPU speed addresses latency; it cannot guarantee that the model will
semantically follow an injected correction.
