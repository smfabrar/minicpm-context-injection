"""Constructed traces test evidence logic, not model feasibility."""
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "minicpm_injection" / "official_t4.py"
spec = importlib.util.spec_from_file_location("official_t4", path)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

presentation_path = path.with_name("t4_colab.py")
presentation_spec = importlib.util.spec_from_file_location("t4_colab", presentation_path)
presentation = importlib.util.module_from_spec(presentation_spec)
import sys
sys.modules[presentation_spec.name] = presentation
presentation_spec.loader.exec_module(presentation)


def trace():
    events = [
        dict(kind="session_start"),
        dict(kind="context_applied", phase="initial", success=True, kv_before=10,
             kv_after=14, expected_kv_growth=4, schema_matches=True),
        dict(kind="output", phase="baseline", turn=0, audio_samples=24000,
             is_listen=False, end_of_turn=False, text="Room B742."),
        dict(kind="context_applied", phase="update", success=True, kv_before=20,
             kv_after=24, expected_kv_growth=4, schema_matches=True),
        dict(kind="output", phase="baseline", turn=0, audio_samples=24000,
             is_listen=False, end_of_turn=True, text=" Follow the signs."),
        dict(kind="exchange_complete", phase="baseline"),
        dict(kind="output", phase="followup", turn=1, audio_samples=24000,
             is_listen=False, end_of_turn=True, text="C583."),
        dict(kind="exchange_complete", phase="followup"),
        dict(kind="output", phase="recall", turn=2, audio_samples=24000,
             is_listen=False, end_of_turn=True, text="C58 3."),
        dict(kind="exchange_complete", phase="recall"),
    ]
    return [dict(seq=index, **event) for index, event in enumerate(events)]


class OfficialEvidenceTests(unittest.TestCase):
    def evaluate(self, events, injected=True):
        return harness.evaluate(events, "B742", "C583", injected, True)

    def test_complete_injected_trace_passes(self):
        self.assertEqual(self.evaluate(trace())["verdict"], "PASS")

    def test_kv_growth_without_text_schema_is_not_proof(self):
        events = trace()
        events[3]["schema_matches"] = False
        self.assertEqual(self.evaluate(events)["verdict"], "ERROR")
        events = trace()
        events[3]["kv_after"] += 1
        self.assertEqual(self.evaluate(events)["verdict"], "ERROR")

    def test_different_turn_and_silence_do_not_prove_overlap(self):
        events = trace()
        events[4]["turn"] = 1
        self.assertEqual(self.evaluate(events)["verdict"], "NOT TESTED")
        events = trace()
        events[2]["is_listen"] = True
        self.assertEqual(self.evaluate(events)["verdict"], "NOT TESTED")

    def test_recall_must_exclude_cancelled_code(self):
        events = trace()
        events[8]["text"] = "C583 replaces B742."
        self.assertEqual(self.evaluate(events)["verdict"], "FAIL")

    def test_control_must_keep_old_code_and_exclude_withheld_code(self):
        events = trace()
        for index in (6, 8):
            events[index]["text"] = "B742."
        self.assertEqual(self.evaluate(events, injected=False)["verdict"], "PASS")
        events[8]["text"] = "C583."
        self.assertEqual(self.evaluate(events, injected=False)["verdict"], "FAIL")

    def test_missing_exchange_and_session_restart_cannot_pass(self):
        events = trace()
        events.pop()
        self.assertEqual(self.evaluate(events)["verdict"], "NOT TESTED")
        events = trace()
        events.append(dict(seq=10, kind="session_start"))
        self.assertEqual(self.evaluate(events)["verdict"], "ERROR")

    def test_partial_or_longer_code_is_not_exact_match(self):
        self.assertFalse(harness.contains_code("C58", "C583"))
        self.assertFalse(harness.contains_code("C5830", "C583"))
        self.assertTrue(harness.contains_code("C 5 8 3.", "C583"))


class NotebookResultsTests(unittest.TestCase):
    def test_single_condition_cannot_be_presented_as_paired_pass(self):
        report = dict(paired_pass=True, results=[dict(verdict="PASS", injected=True,
                       old_code="B742", new_code="C583", answers=dict(followup="<script>"))])
        rendered = presentation.results_html(report, "Actual results")
        self.assertIn("Not established", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)

    def test_export_keeps_actual_verdict_and_pins_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, work = root / "repository", root / "work"
            (repo / "notebooks").mkdir(parents=True)
            work.mkdir()
            template = dict(cells=[
                dict(cell_type="code", source=['EVIDENCE_COMMIT = "main"\n'], metadata={}, outputs=[]),
                dict(cell_type="code", source=["show_results(report)\n"],
                     metadata={"evidence_result": "correction"}, outputs=[]),
            ], metadata={})
            (repo / "notebooks" / "official_t4_context_injection_colab.ipynb").write_text(json.dumps(template))
            report = dict(results=[dict(verdict="FAIL", old_code="B742", new_code="C583",
                                       answers=dict(followup="B742."))])
            with patch.object(presentation.subprocess, "check_output", return_value="a" * 40 + "\n"):
                output = presentation.save_results_notebook(presentation.ColabWorkspace(repo, work),
                                                           {"correction": report})
            saved = json.loads(output.read_text())
            self.assertIn("a" * 40, "".join(saved["cells"][0]["source"]))
            rendered = saved["cells"][1]["outputs"][0]["data"]["text/html"]
            self.assertIn("FAIL", rendered)
            self.assertIn("B742.", rendered)
            self.assertEqual(json.loads((work / "displayed_reports.json").read_text())["correction"], report)


if __name__ == "__main__":
    unittest.main()
