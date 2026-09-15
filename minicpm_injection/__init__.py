"""Reproducible MiniCPM-o context-injection experiment."""

from .environment import Workspace, build_cuda, download_models, prepare_runtime, require_nvidia
from .experiment import run_mid_generation_pair, run_multi_turn_update
from .reporting import audit_results, archive_runs, show_run, write_combined_report

__all__ = [
    "Workspace",
    "audit_results",
    "archive_runs",
    "build_cuda",
    "download_models",
    "prepare_runtime",
    "require_nvidia",
    "run_mid_generation_pair",
    "run_multi_turn_update",
    "show_run",
    "write_combined_report",
]
