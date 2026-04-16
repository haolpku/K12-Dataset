#!/usr/bin/env bash
# Launch vLLM for Ministral-3-14B-Instruct (matches configs/models/ministral3_14b_instruct_vllm.yaml).
# Env overrides: VLLM_MODEL_ID, VLLM_SERVED_NAME, VLLM_HOST, VLLM_PORT, VLLM_TP_SIZE,
# VLLM_MAX_MODEL_LEN, VLLM_GPU_MEM, VLLM_LOG_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
LOG_DIR="${VLLM_LOG_DIR:-${ROOT}/workspace/benchmark_output/logs}"
mkdir -p "${LOG_DIR}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found in PATH" >&2
  exit 127
fi

MODEL_ID="${VLLM_MODEL_ID:-mistralai/Ministral-3-14B-Instruct-2512}"
SERVED_NAME="${VLLM_SERVED_NAME:-ministral3-14b-instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP_SIZE:-2}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-40960}"
GPU_MEM="${VLLM_GPU_MEM:-0.9}"
LOG_FILE="${LOG_DIR}/${SERVED_NAME}.log"

echo "[INFO] model=${MODEL_ID} served=${SERVED_NAME} host=${HOST} port=${PORT} tp=${TP}"
echo "[INFO] log_file=${LOG_FILE}"

vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  2>&1 | tee "${LOG_FILE}"
