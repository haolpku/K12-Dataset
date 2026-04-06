"""Load and resolve the unified pipeline configuration (config/default.yaml)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR}`` placeholders with environment variable values."""
    def _repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_VAR_RE.sub(_repl, value)


def _resolve_value(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env(value)
    if isinstance(value, dict):
        return {k: _resolve_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v) for v in value]
    return value


class PipelineConfig:
    """Parsed pipeline config with resolved, absolute paths."""

    def __init__(self, raw: Dict[str, Any], config_dir: Path) -> None:
        self._raw = _resolve_value(raw)
        self._config_dir = config_dir.resolve()

    def _resolve_path(self, rel: str) -> Path:
        """Resolve a path relative to *config_dir*."""
        p = Path(rel)
        if p.is_absolute():
            return p
        return (self._config_dir / p).resolve()

    # ---- path helpers -------------------------------------------------------

    @property
    def books_yaml(self) -> Path:
        return self._resolve_path(self._raw.get("paths", {}).get("books_yaml", "../books.yaml"))

    @property
    def input_dir(self) -> Path:
        return self._resolve_path(self._raw.get("paths", {}).get("input_dir", "../input"))

    @property
    def workspace_dir(self) -> Path:
        return self._resolve_path(self._raw.get("paths", {}).get("workspace_dir", "../workspace"))

    @property
    def output_dir(self) -> Path:
        return self._resolve_path(self._raw.get("paths", {}).get("output_dir", "../data"))

    @property
    def chapter_kg_dir(self) -> Path:
        return self.output_dir / "chapter_kg"

    @property
    def book_kg_dir(self) -> Path:
        return self.output_dir / "book_kg"

    @property
    def subject_stage_kg_dir(self) -> Path:
        return self.output_dir / "subject_stage_kg"

    @property
    def subject_kg_dir(self) -> Path:
        return self.output_dir / "subject_kg"

    @property
    def global_kg_dir(self) -> Path:
        return self.output_dir / "global_kg"

    @property
    def segmentation_workspace_dir(self) -> Path:
        return self.workspace_dir / "segmentation"

    @property
    def build_graph_workspace_dir(self) -> Path:
        return self.workspace_dir / "build_graph"

    @property
    def check_graph_workspace_dir(self) -> Path:
        return self.workspace_dir / "check_graph"

    # ---- LLM settings -------------------------------------------------------

    @property
    def llm(self) -> Dict[str, Any]:
        return self._raw.get("llm", {})

    # ---- merge settings ------------------------------------------------------

    @property
    def merge_normalize(self) -> str:
        return self._raw.get("merge", {}).get("normalize", "nfkc_lower")

    @property
    def merge_dedup_labels(self) -> List[str]:
        return self._raw.get("merge", {}).get("dedup_labels", ["Concept", "Skill", "Experiment", "Exercise"])

    # ---- books.yaml loading --------------------------------------------------

    def load_books(self, *, require_source: bool = True) -> List[Dict[str, Any]]:
        """Return validated book records from ``books.yaml``."""
        with open(self.books_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        books = data.get("books", [])
        if not isinstance(books, list):
            raise ValueError("books.yaml must contain a top-level 'books' list")

        required = {"book_prefix", "subject", "stage", "grade", "publisher"}
        if require_source:
            required.add("source")
        normalized: List[Dict[str, Any]] = []
        seen_prefixes: set[str] = set()

        for idx, raw_book in enumerate(books, start=1):
            if not isinstance(raw_book, dict):
                raise ValueError(f"books[{idx}] must be an object")
            missing = [key for key in required if not str(raw_book.get(key, "")).strip()]
            if missing:
                raise ValueError(f"books[{idx}] missing required fields: {', '.join(missing)}")

            book = dict(raw_book)
            book["book_prefix"] = str(book["book_prefix"]).strip()
            if book["book_prefix"] in seen_prefixes:
                raise ValueError(f"duplicate book_prefix in books.yaml: {book['book_prefix']}")
            seen_prefixes.add(book["book_prefix"])
            normalized.append(book)

        return normalized

    def resolve_book_source(self, book: Dict[str, Any]) -> Path:
        source = Path(str(book["source"]))
        if source.is_absolute():
            return source
        return (self.input_dir / source).resolve()

    def segmentation_book_dir(self, book_prefix: str) -> Path:
        return self.segmentation_workspace_dir / book_prefix

    def sections_dir_for(self, book_prefix: str) -> Path:
        return self.segmentation_book_dir(book_prefix) / "sections"

    def sections_index_for(self, book: Dict[str, Any]) -> Path:
        return self.segmentation_book_dir(str(book["book_prefix"])) / "sections_index.json"

    def build_graph_book_dir(self, book_prefix: str) -> Path:
        return self.build_graph_workspace_dir / book_prefix


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a YAML file.

    If *config_path* is ``None``, the default ``config/default.yaml`` relative
    to the repository root is used.
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        config_path_obj = repo_root / "config" / "default.yaml"
    else:
        config_path_obj = Path(config_path)

    with open(config_path_obj, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(raw, config_path_obj.parent)
