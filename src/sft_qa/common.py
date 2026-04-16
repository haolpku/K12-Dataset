"""Shared helpers for the ``sft_qa`` scripts (paths, env bootstrap).

Keeps path resolution consistent across generators and provides a thin wrapper for
loading repository-level environment defaults used by LLM clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def load_openai_env(config_path: Optional[str]) -> Path:
    """Load ``<repo>/config/.env`` into ``os.environ`` without overriding existing keys.

    The ``config_path`` argument is retained for backwards compatibility with older
    call sites; the on-disk file used is always ``config/.env`` at the repo root.
    """
    _ = config_path
    from utils.config import load_repo_dotenv

    return load_repo_dotenv()


def resolve_input_path(config: Any, subject_stage: str, input_json: Optional[str]) -> Path:
    if input_json:
        return Path(input_json).resolve()
    return (config.output_dir / "subject_stage_kg" / f"{subject_stage}.json").resolve()


def resolve_workspace_root(config: Any, subject_stage: str, workspace_dir: Optional[str]) -> Path:
    if workspace_dir:
        return Path(workspace_dir).resolve()
    return (config.workspace_dir / "sft_qa" / subject_stage).resolve()
