#!/usr/bin/env bash
# Launch vLLM for GLM-4.7-Flash (matches configs/models/glm4.7flash_vllm.yaml).
# Env overrides: VLLM_MODEL_ID, VLLM_SERVED_NAME, VLLM_HOST, VLLM_PORT, VLLM_TP_SIZE,
# VLLM_MAX_MODEL_LEN, VLLM_GPU_MEM, VLLM_LOG_DIR, VLLM_TOOL_CALL_PARSER (glm47|glm45).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
LOG_DIR="${VLLM_LOG_DIR:-${ROOT}/workspace/benchmark_output/logs}"
mkdir -p "${LOG_DIR}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found in PATH" >&2
  exit 127
fi

MODEL_ID="${VLLM_MODEL_ID:-zai-org/GLM-4.7-Flash}"
SERVED_NAME="${VLLM_SERVED_NAME:-glm-4.7-flash-local}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP_SIZE:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEM="${VLLM_GPU_MEM:-0.90}"
LOG_FILE="${LOG_DIR}/${SERVED_NAME}.log"

TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-glm47}"
if ! vllm serve --help 2>/dev/null | grep -q "glm47"; then
  TOOL_CALL_PARSER="glm45"
  echo "[WARN] glm47 tool parser not available in this vLLM build; using glm45" >&2
fi

echo "[INFO] model=${MODEL_ID} served=${SERVED_NAME} host=${HOST} port=${PORT} tp=${TP}"
echo "[INFO] tool_call_parser=${TOOL_CALL_PARSER} reasoning_parser=glm45"
echo "[INFO] log_file=${LOG_FILE}"

vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  2>&1 | tee "${LOG_FILE}"
