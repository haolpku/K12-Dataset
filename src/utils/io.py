"""Small JSON / JSONL helpers shared across pipeline modules.

Provides strict-ish readers and writers used by KG merge, benchmark export, and SFT
utilities to keep serialization consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    """Write *obj* as pretty-printed JSON.  Parent dirs are created automatically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a JSON-Lines file (one JSON object per line)."""
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{p} line {line_no} invalid json: {exc}") from exc
                if isinstance(payload, dict):
                    records.append(payload)
    return records


def write_jsonl(path: Union[str, Path], records: Iterable[Dict[str, Any]]) -> None:
    """Write *records* as JSON-Lines.  Parent dirs are created automatically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
