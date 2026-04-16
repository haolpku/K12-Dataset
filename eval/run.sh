#!/usr/bin/env bash
# 在任意目录执行均可；仅传模型名时使用脚本内默认输入/输出路径。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?用法: $0 <model> [input_dir] [output_dir]}"
shift || true
if [ "$#" -eq 0 ]; then
  exec python3 "$ROOT/eval/eval_multiselect.py" --model "$MODEL"
elif [ "$#" -eq 1 ]; then
  exec python3 "$ROOT/eval/eval_multiselect.py" --model "$MODEL" -i "$1"
else
  exec python3 "$ROOT/eval/eval_multiselect.py" --model "$MODEL" -i "$1" -o "$2"
fi
