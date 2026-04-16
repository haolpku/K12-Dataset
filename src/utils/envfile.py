"""Load KEY=VALUE environment files into ``os.environ``.

Used for repo ``config/.env`` and for ``eval/configs/.env`` (K12 multiselect eval).
Does not override keys that are already set in the process environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> bool:
    """Parse a minimal dotenv-style file and apply entries to ``os.environ``.

    Returns ``True`` if *path* existed and was read, ``False`` if missing.
    """
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("\"", "'")) and value.endswith(("\"", "'")) and len(value) >= 2:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return True
