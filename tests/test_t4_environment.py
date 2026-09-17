"""A known metadata exception must not hide dependency or binary failures."""
import importlib.util
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "minicpm_injection" / "check_t4_environment.py"
spec = importlib.util.spec_from_file_location("check_t4_environment", path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class DependencyCheckTests(unittest.TestCase):
    def metadata(self, **overrides):
        values = dict(system="linux", machine="x86_64", version="0.6.0",
                      wheel="Wheel-Version: 1.0\n" + checker.STALE_TAG + "\n")
        values.update(overrides)
        return values

    def test_exact_documented_error_is_recognized(self):
        self.assertTrue(checker.known_decord_metadata_error(checker.DECORD_ERROR + "\n", 1,
                                                           **self.metadata()))

    def test_extra_dependency_error_is_fatal(self):
        output = checker.DECORD_ERROR + "\ntorch requires a package which is not installed.\n"
        self.assertFalse(checker.known_decord_metadata_error(output, 1, **self.metadata()))

    def test_different_architecture_version_or_tag_is_fatal(self):
        for override in (dict(machine="aarch64"), dict(system="darwin"),
                         dict(version="0.7.0"), dict(wheel="Tag: py3-none-other\n")):
            with self.subTest(override=override):
                self.assertFalse(checker.known_decord_metadata_error(checker.DECORD_ERROR, 1,
                                                                    **self.metadata(**override)))

    def test_native_import_error_propagates(self):
        completed = subprocess.CompletedProcess([], 1, checker.DECORD_ERROR)
        distribution = types.SimpleNamespace(version="0.6.0", read_text=lambda name: checker.STALE_TAG)
        with patch.object(checker.subprocess, "run", return_value=completed), \
             patch.object(checker.importlib.metadata, "distribution", return_value=distribution), \
             patch.object(checker.sys, "platform", "linux"), \
             patch.object(checker.platform, "machine", return_value="x86_64"), \
             patch.dict("sys.modules", {"decord": None}):
            with self.assertRaises(ModuleNotFoundError):
                checker.check_dependencies()


if __name__ == "__main__":
    unittest.main()
