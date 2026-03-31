#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXTRACT_PY="${SCRIPT_DIR}/extract_kg_from_textbook.py"
ENV_FILE="${SCRIPT_DIR}/.openai.env"
ROOT_OUTPUT="${REPO_ROOT}/output"

usage() {
  cat <<USAGE_EOF
Usage:
  $(basename "$0") --subject 数学 --grade 七年级上册[,八年级上册] [--book-prefix math_rjb_1a] [--book-prefix-list p1,p2] [--chapters ch1_s1,ch2_s3] [--stage 初中] [--publisher 人教版] [--prompt prompt.txt] [--force]

Only supported mode:
  --grade 支持逗号分隔列表（例如 必修一,必修二），按精确匹配目录名
  --book-prefix 可选；单本或多本共用同一前缀
  --book-prefix-list 可选；按 --grade 列表顺序一一对应（例如 bx1,bx2）
  --chapters 可选；给了就按 out_sections 文件名筛选，不给就处理全部 md（自动排除 exercise.md）
  未传 --chapters 且未 --force 时：会自动跳过 raw_output 中已有对应 json 的文件，只跑剩余文件
USAGE_EOF
}

SUBJECT=""
GRADE=""
CHAPTERS=""
STAGE=""
PUBLISHER=""
PROMPT=""
BOOK_PREFIX=""
BOOK_PREFIX_LIST=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="${2:-}"; shift 2 ;;
    --grade) GRADE="${2:-}"; shift 2 ;;
    --chapters) CHAPTERS="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --publisher) PUBLISHER="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --book-prefix) BOOK_PREFIX="${2:-}"; shift 2 ;;
    --book-prefix-list) BOOK_PREFIX_LIST="${2:-}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$SUBJECT" && -n "$GRADE" ]] || {
  echo "ERROR: --subject --grade 必填" >&2
  usage
  exit 1
}

if [[ -z "$BOOK_PREFIX" && -z "$BOOK_PREFIX_LIST" ]]; then
  echo "ERROR: --book-prefix 与 --book-prefix-list 至少提供一个" >&2
  usage
  exit 1
fi

[[ -f "$EXTRACT_PY" ]] || { echo "ERROR: not found: $EXTRACT_PY" >&2; exit 1; }
[[ -f "$ENV_FILE" ]] || { echo "ERROR: not found: $ENV_FILE" >&2; exit 1; }

if [[ -z "$PROMPT" ]]; then
  PROMPT="${SCRIPT_DIR}/prompt.txt"
fi
[[ -f "$PROMPT" ]] || { echo "ERROR: prompt not found: $PROMPT" >&2; exit 1; }

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "ERROR: OPENAI_API_KEY missing in $ENV_FILE" >&2; exit 1; }

trim_space() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

IFS=',' read -r -a GRADE_LIST_RAW <<< "$GRADE"
GRADE_LIST=()
for g in "${GRADE_LIST_RAW[@]}"; do
  g="$(trim_space "$g")"
  [[ -n "$g" ]] && GRADE_LIST+=("$g")
done

[[ "${#GRADE_LIST[@]}" -gt 0 ]] || {
  echo "ERROR: --grade 为空" >&2
  exit 1
}

grade_match() {
  local cand="$1"
  local g
  for g in "${GRADE_LIST[@]}"; do
    [[ "$cand" == "$g" ]] && return 0
  done
  return 1
}

declare -A BOOK_PREFIX_BY_GRADE=()
if [[ -n "$BOOK_PREFIX_LIST" ]]; then
  IFS=',' read -r -a PREFIX_LIST_RAW <<< "$BOOK_PREFIX_LIST"
  PREFIX_LIST=()
  for p in "${PREFIX_LIST_RAW[@]}"; do
    p="$(trim_space "$p")"
    [[ -n "$p" ]] && PREFIX_LIST+=("$p")
  done

  if [[ "${#PREFIX_LIST[@]}" -ne "${#GRADE_LIST[@]}" ]]; then
    echo "ERROR: --book-prefix-list 数量必须与 --grade 数量一致" >&2
    echo "  grades:  ${#GRADE_LIST[@]} -> ${GRADE_LIST[*]}" >&2
    echo "  prefixes:${#PREFIX_LIST[@]} -> ${PREFIX_LIST[*]}" >&2
    exit 1
  fi

  for i in "${!GRADE_LIST[@]}"; do
    BOOK_PREFIX_BY_GRADE["${GRADE_LIST[$i]}"]="${PREFIX_LIST[$i]}"
  done
fi

book_dirs=()
for s in "$ROOT_OUTPUT"/*; do
  [[ -d "$s" ]] || continue
  [[ "$(basename "$s")" == *"$SUBJECT"* ]] || continue
  for st in "$s"/*; do
    [[ -d "$st" ]] || continue
    [[ -z "$STAGE" || "$(basename "$st")" == *"$STAGE"* ]] || continue
    for p in "$st"/*; do
      [[ -d "$p" ]] || continue
      [[ "$(basename "$p")" == *"$PUBLISHER"* ]] || continue
      for b in "$p"/*; do
        [[ -d "$b" ]] || continue
        grade_match "$(basename "$b")" || continue
        [[ -d "$b/out_sections" ]] || continue
        book_dirs+=("$b")
      done
    done
  done
done

if [[ "${#book_dirs[@]}" -eq 0 ]]; then
  echo "ERROR: 未找到匹配书本" >&2
  exit 1
fi

echo "Matched books: ${#book_dirs[@]}"

for book_dir in "${book_dirs[@]}"; do
  echo "==> Processing: $book_dir"

  grade_name="$(basename "$book_dir")"
  if [[ -n "$BOOK_PREFIX_LIST" ]]; then
    current_book_prefix="${BOOK_PREFIX_BY_GRADE[$grade_name]:-}"
    [[ -n "$current_book_prefix" ]] || {
      echo "ERROR: 未找到 grade=$grade_name 对应的 book-prefix（请检查 --book-prefix-list 顺序）" >&2
      exit 1
    }
  else
    current_book_prefix="$BOOK_PREFIX"
  fi

  sections_dir="$book_dir/out_sections"
  output_dir="$book_dir/raw_output"
  mkdir -p "$output_dir"

  tmp_sections_dir="$(mktemp -d "${TMPDIR:-/tmp}/kg_sections.XXXXXX")"

  python3 - "$sections_dir" "$tmp_sections_dir" "$CHAPTERS" "$output_dir" "$FORCE" <<'PY'
import json
import os
import shutil
import sys

src_dir, dst_dir, raw, output_dir, force = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
index_path = os.path.join(src_dir, "sections_index.json")
with open(index_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sections = data.get("sections", [])

want = None
raw = (raw or "").strip()
if raw:
    want = set()
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        if x.endswith(".md"):
            x = x[:-3]
        want.add(x)
    if not want:
        raise ValueError("--chapters 为空")

picked = []
skipped_existing = 0
for item in sections:
    name = os.path.basename(item.get("file", ""))
    stem, _ = os.path.splitext(name)
    if name == "exercise.md":
        continue
    if "整理与提升" in name:
        continue
    if want is not None and stem not in want:
        continue
    if want is None and force != 1 and os.path.exists(os.path.join(output_dir, f"{stem}.json")):
        skipped_existing += 1
        continue
    picked.append(item)

if want is not None and not picked:
    raise ValueError(f"未匹配到章节文件: {sorted(want)}")

picked_existing = []
skipped_missing = 0
for item in picked:
    src = os.path.join(src_dir, item["file"])
    if not os.path.exists(src):
        skipped_missing += 1
        continue
    picked_existing.append(item)

os.makedirs(dst_dir, exist_ok=True)
new_data = dict(data)
new_data["sections"] = picked_existing
with open(os.path.join(dst_dir, "sections_index.json"), "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

for item in picked_existing:
    src = os.path.join(src_dir, item["file"])
    dst = os.path.join(dst_dir, item["file"])
    shutil.copy2(src, dst)

print(f"Filtered sections: {len(picked_existing)} (skipped existing: {skipped_existing}, skipped missing: {skipped_missing})")
PY

  pending_count="$(python3 - "$tmp_sections_dir" <<'PY'
import json
import sys
from pathlib import Path
idx = Path(sys.argv[1]) / "sections_index.json"
with idx.open("r", encoding="utf-8") as f:
    data = json.load(f)
print(len(data.get("sections", [])))
PY
)"

  if [[ "$pending_count" -eq 0 ]]; then
    echo "SKIP: 没有待处理章节（未传 --chapters 时会自动跳过已有输出）"
    rm -rf "$tmp_sections_dir"
    continue
  fi

  publisher_arg="$(basename "$(dirname "$book_dir")")"
  [[ "$publisher_arg" == *"人教版"* ]] && publisher_arg="人民教育出版社"

  export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
  cmd=(
    python3 "$EXTRACT_PY"
    --sections-dir "$tmp_sections_dir"
    --prompt "$PROMPT"
    --model "$OPENAI_MODEL"
    --api-key "$OPENAI_API_KEY"
    --grade "$grade_name"
    --publisher "$publisher_arg"
    --subject "$SUBJECT"
    --output-dir "$output_dir"
    --book-prefix "$current_book_prefix"
  )
  [[ -n "${OPENAI_BASE_URL:-}" ]] && cmd+=(--base-url "$OPENAI_BASE_URL")

  "${cmd[@]}"
  rm -rf "$tmp_sections_dir"
done
