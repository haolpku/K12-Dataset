#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_FILE="${SCRIPT_DIR}/generate_exercise_json.py"
ENV_FILE="${REPO_ROOT}/src/exercise/.openai.env"

if [[ ! -f "$PY_FILE" ]]; then
  echo "ERROR: not found: $PY_FILE" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

python3 "$PY_FILE" "$@"
