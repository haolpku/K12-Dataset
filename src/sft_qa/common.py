"""Shared path helpers for SFT QA scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def resolve_input_path(config: Any, subject_stage: str, input_json: Optional[str]) -> Path:
    if input_json:
        return Path(input_json).resolve()
    return (config.output_dir / "subject_stage_kg" / f"{subject_stage}.json").resolve()


def resolve_workspace_root(config: Any, subject_stage: str, workspace_dir: Optional[str]) -> Path:
    if workspace_dir:
        return Path(workspace_dir).resolve()
    return (config.workspace_dir / "sft_qa" / subject_stage).resolve()
