#!/usr/bin/env bash
# Launch vLLM for Llama-3-8B-Instruct (matches configs/models/llama3_8b_instruct_vllm.yaml).
# Env overrides: VLLM_MODEL_ID, VLLM_SERVED_NAME, VLLM_HOST, VLLM_PORT, VLLM_TP_SIZE,
# VLLM_MAX_MODEL_LEN, VLLM_GPU_MEM, VLLM_LOG_DIR.
# Hugging Face gated models: export HF_TOKEN before launch if downloads fail.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
LOG_DIR="${VLLM_LOG_DIR:-${ROOT}/workspace/benchmark_output/logs}"
mkdir -p "${LOG_DIR}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "[ERROR] vllm not found in PATH" >&2
  exit 127
fi

MODEL_ID="${VLLM_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
SERVED_NAME="${VLLM_SERVED_NAME:-llama3-8b-instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP_SIZE:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
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
  2>&1 | tee "${LOG_FILE}"
