"""Generate the short teaching notebook; setup and inference live in Git."""
from pathlib import Path

import nbformat as n

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/smfabrar/minicpm-context-injection.git"
IMPLEMENTATION_COMMIT = "b2420e3b66bd545b7ad88f0d74f0c081f8731941"
markdown = n.v4.new_markdown_cell
code = n.v4.new_code_cell
notebook = n.v4.new_notebook()
notebook.metadata = {
    "accelerator": "GPU",
    "colab": {"name": "MiniCPM-o — context injection on T4"},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
notebook.cells = [
    markdown("""# Can MiniCPM-o use new information during a conversation?

We want to give a speech assistant a new fact **after the conversation has started**, without restarting its memory. For example, a room assignment changes while the assistant is speaking. Can it remember the replacement when asked later?

This notebook tests that question using **MiniCPM-o 4.5's official GPTQ 4-bit model** on a **Colab T4**. We run two experiments, each with an injected condition and a control. You will see actual answers, evidence checks, verdicts and generated speech below the experiment cells.

**Before you start:** select **Runtime → Change runtime type → T4 GPU**, then run from top to bottom. Use a fresh runtime if the previous native notebook is still running.

The official GPTQ card estimates about 11 GB of model memory. TTS and conversation memory need extra space, so this notebook also checks whether the configuration actually runs on your assigned GPU.

[Official model](https://huggingface.co/openbmb/MiniCPM-o-4_5-GPTQ) · [Official streaming API](https://huggingface.co/openbmb/MiniCPM-o-4_5) · [Experiment source](https://github.com/smfabrar/minicpm-context-injection)
"""),
    markdown("""## What we mean by context injection

MiniCPM's duplex model processes audio in short units and keeps its language-model memory between units. The official API also accepts text through `streaming_prefill(text_list=[...])`. We use that input to supply a room assignment and, later, a correction.

```text
Recorded question → audio encoder → conversation memory → text + speech
                                          ↑
                              new external room assignment
```

Three checks answer different questions:

| Check | What it establishes |
|---|---|
| Text tokens appear in the prefill record and increase memory by the expected amount | The new text reached the model |
| Later answers contain the replacement room and omit the old room | The model used the new information |
| A control keeps the old room when the replacement is withheld | The replacement was not supplied through the questions |

The second experiment adds a timing check: the correction must enter memory between two generated speech chunks from the same response. Each condition starts one session and keeps it for all exchanges.
"""),
    markdown("""## 1. Load the experiment from Git

The notebook shows the experiment steps and results. Setup, model loading, evidence evaluation and playback helpers live in normal Python files in the repository. This cell checks out the published implementation and prints its exact commit.
"""),
    code(f'''from pathlib import Path
import subprocess, sys

EVIDENCE_REPOSITORY = "{REPOSITORY_URL}"
EVIDENCE_COMMIT = "{IMPLEMENTATION_COMMIT}"
REPOSITORY = Path("/content/minicpm-context-injection")

if not (REPOSITORY / ".git").is_dir():
    subprocess.run(["git", "clone", EVIDENCE_REPOSITORY, str(REPOSITORY)], check=True)
subprocess.run(["git", "-C", str(REPOSITORY), "fetch", "--quiet", "origin", EVIDENCE_COMMIT], check=True)
subprocess.run(["git", "-C", str(REPOSITORY), "checkout", "--detach", EVIDENCE_COMMIT], check=True)
actual_commit = subprocess.check_output(
    ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
if EVIDENCE_COMMIT != "main":
    assert actual_commit == EVIDENCE_COMMIT
sys.path.insert(0, str(REPOSITORY))

from minicpm_injection.t4_colab import (
    ColabWorkspace, export_evidence, play_answers, run_evidence,
    save_results_notebook, setup_environment, show_results, show_timeline,
)
workspace = ColabWorkspace(REPOSITORY, Path("/content/minicpm-official-t4"))
reports = {{}}
print("Implementation commit:", actual_commit)
'''),
    markdown("""## 2. Prepare the T4 environment

We load the quantized language model with its audio encoder and speech output. Vision is disabled. The T4 uses **FP16** and **SDPA attention**.

The GPTQ wheel needs matching Python, PyTorch and CUDA versions. Setup downloads prebuilt Python 3.11 and installs the pinned CUDA libraries from wheels. MiniCPM's utility package is distributed as pure Python source and is allowed to package itself; **no CUDA source is compiled**.

This cell shows setup progress and a GPU summary. If installation fails, it shows the actual error and full log path. The notebook kernel stays unchanged; GPU inference runs in the isolated environment.

Decord's published Linux wheel has a known stale Python 3.6 tag. Setup accepts that one metadata error only after confirming the tag and importing the native library. Every other dependency error stops setup. Rerunning setup reuses installed packages and downloads.
"""),
    code("setup_environment(workspace)\n"),
    markdown("""## 3. Optional Hugging Face authentication

The model is public. If a download requires authentication, add a Colab secret named `HF_TOKEN` under the key icon, enable notebook access, then rerun this cell. The token is passed to the downloader without printing it.
"""),
    code("""hf_token = None
try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
except Exception:
    pass
print("Hugging Face token available:", hf_token is not None)
"""),
    markdown("""## 4. Experiment A — correction and later recall

**Question:** does a correction replace an older fact and survive another exchange?

1. Start one duplex session and supply a random room code.
2. Ask for the assigned room and verify the initial answer.
3. After that exchange, choose a different random code and inject the correction.
4. Ask for the current room.
5. Ask again without injecting anything else.

The **control** repeats the questions and initial room with the same seed, but receives no replacement code. Questions never contain either room code.

**Expected result:** the injected condition uses the new room in both later answers; the control keeps the old room. The tables show both conditions, actual answers and evidence checks. The notebook automatically saves these result tables for sharing.
"""),
    code("""reports["correction"] = run_evidence(workspace, token=hf_token)
show_results(reports["correction"], "Experiment A — correction and recall")
results_notebook = save_results_notebook(workspace, reports)
"""),
    markdown("""### Listen to the three answers

These players contain the model's generated speech for the injected condition: before correction, after correction and later recall. The transcript table above is evaluated against exact room codes; speech pronunciation is a separate observation.
"""),
    code('play_answers(reports["correction"])\n'),
    markdown("""## 5. Experiment B — update during an ongoing speech response

**Question:** can the correction enter the same session while a speech response is still being produced?

This time, the first question requests a longer answer. We wait for a generated speech chunk that is not the final chunk. **Only then** do we choose the replacement code and inject it at the next unit boundary.

The timing test requires another speech chunk from the **same response** after the text evaluation. Later questions test corrected recall. The control uses the same procedure while withholding the replacement code.

**Expected result:** verified text evaluation between two speech chunks, followed by the new room in both later injected answers and the old room in the control. If the model finishes speaking before a suitable update is possible, the timing claim is **NOT TESTED**.
"""),
    code("""reports["speech"] = run_evidence(workspace, mid_speech=True, token=hf_token)
show_results(reports["speech"], "Experiment B — update during speech")
results_notebook = save_results_notebook(workspace, reports)
"""),
    markdown("""### Inspect the event order

The timeline shows when the new fact arrived, when its text was evaluated, how the conversation memory grew, and which speech turn produced each chunk. This lets you check timing evidence against the generated answers.
"""),
    code('show_timeline(reports["speech"])\n'),
    markdown("""## 6. Interpret and share the results

| Verdict | Meaning |
|---|---|
| PASS | Every required transport, timing and answer check passed for that condition |
| FAIL | Completed answers did not satisfy the semantic checks |
| NOT TESTED | A required completed exchange or speech timing event did not occur |
| ERROR | Setup, model loading or an evidence prerequisite failed |

An experiment establishes feasibility only when **both injected and control conditions pass**. A single pair does not establish reliability; repeated pairs are needed. The experiment uses recorded questions and remote GPU speech generation. Simultaneous physical microphone/speaker operation and real-time throughput are not measured here.

The export includes a **notebook with actual result tables saved in its outputs** and a ZIP containing every retained trial, logs, transcripts, input audio, generated speech and manifests. Until inference runs successfully, there are no T4 model results to claim.
"""),
    code("""results_notebook, evidence_archive = export_evidence(workspace, reports)
print("Notebook with results:", results_notebook)
print("All trial evidence:", evidence_archive)

# To download:
# from google.colab import files
# files.download(str(results_notebook))
# files.download(str(evidence_archive))
"""),
]
notebook.cells[9].metadata["evidence_result"] = "correction"
notebook.cells[13].metadata["evidence_result"] = "speech"
n.validate(notebook)
for output in (ROOT / "notebooks" / "official_t4_context_injection_colab.ipynb",
               ROOT.parent / "official_t4_context_injection_colab.ipynb"):
    n.write(notebook, output)
    print(output)
