"""Logic tests use constructed events; they are not model evidence."""
import unittest

from minicpm_injection.experiment import contains_code, injection_overlapped_tts


class VerdictLogicTests(unittest.TestCase):
    def test_exact_code_match_allows_stream_spacing(self):
        self.assertTrue(contains_code("The room is C58 3", "C583"))
        self.assertFalse(contains_code("The room is C58", "C583"))
        self.assertFalse(contains_code("The room is C5830", "C583"))

    def test_injection_must_be_between_chunks_of_same_turn(self):
        payload = "Current room=C583."
        valid = [
            {"kind": "audio", "t": 1, "turn": 0, "final": False},
            {
                "kind": "context_applied",
                "t": 2,
                "text": payload,
                "ok": True,
                "kv_before": 10,
                "kv_after": 16,
            },
            {"kind": "audio", "t": 3, "turn": 0, "final": True},
        ]
        self.assertTrue(injection_overlapped_tts(valid, payload))

        after_speech = [dict(event) for event in valid]
        after_speech[0]["final"] = True
        self.assertFalse(injection_overlapped_tts(after_speech, payload))

        failed_evaluation = [dict(event) for event in valid]
        failed_evaluation[1]["ok"] = False
        self.assertFalse(injection_overlapped_tts(failed_evaluation, payload))


if __name__ == "__main__":
    unittest.main()
