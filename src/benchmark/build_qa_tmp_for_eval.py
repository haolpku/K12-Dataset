#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


LABELS = ("A", "B", "C", "D")
K_VALUES = (1, 2, 3)
# Preferred ratio for non-forced questions (between 2 and 3 correct options).
TARGET_RATIO_NON1 = {2: 0.6, 3: 0.4}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {i} invalid json: {e}") from e
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def clean_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for x in values:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def seed_from_id(qid: str, salt: str = "") -> int:
    h = hashlib.sha256(f"{qid}|{salt}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def shuffled_by_id(items: Sequence[str], qid: str, salt: str) -> List[str]:
    arr = list(items)
    rnd = random.Random(seed_from_id(qid, salt))
    rnd.shuffle(arr)
    return arr


def choose_k(
    feasible: List[int],
    global_counts: Dict[int, int],
    seen_count: int,
    qid: str,
    target_ratio: Dict[int, float],
) -> int:
    # Choose the k with the largest deficit from target ratio.
    best_k = feasible[0]
    best_score = None
    for k in feasible:
        target = target_ratio[k] * (seen_count + 1)
        deficit = target - global_counts[k]
        tie = seed_from_id(qid, f"k_tie_{k}") / float(2**64)
        score = (deficit, tie)
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
    return best_k


def combos_for_k(k: int) -> List[Tuple[int, ...]]:
    if k == 1:
        return [(0,), (1,), (2,), (3,)]
    if k == 2:
        return [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    if k == 3:
        return [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    raise ValueError(f"unsupported k={k}")


def choose_combo(
    k: int,
    combo_counts: Dict[int, Dict[Tuple[int, ...], int]],
    qid: str,
) -> Tuple[int, ...]:
    combos = combos_for_k(k)
    min_count = min(combo_counts[k][c] for c in combos)
    candidates = [c for c in combos if combo_counts[k][c] == min_count]
    candidates = sorted(
        candidates,
        key=lambda c: seed_from_id(qid, f"combo_{k}_{c}"),
    )
    return candidates[0]


def convert_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    rows = read_jsonl(input_path)
    out_rows: List[Dict[str, Any]] = []

    k_counts = {1: 0, 2: 0, 3: 0}
    combo_counts: Dict[int, Dict[Tuple[int, ...], int]] = {
        1: {c: 0 for c in combos_for_k(1)},
        2: {c: 0 for c in combos_for_k(2)},
        3: {c: 0 for c in combos_for_k(3)},
    }
    skipped = 0
    forced_one = 0

    for row in rows:
        qid = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        answers = clean_text_list(row.get("answer"))
        candidates = clean_text_list(row.get("candidates"))
        candidates = [x for x in candidates if x not in set(answers)]

        feasible = [k for k in K_VALUES if len(answers) >= k and len(candidates) >= (4 - k)]
        if not feasible:
            skipped += 1
            continue

        # Prefer k in {2,3}. Only use k=1 when the sample cannot support 2/3.
        feasible_non1 = [k for k in feasible if k in (2, 3)]
        if feasible_non1:
            target_ratio = {k: TARGET_RATIO_NON1[k] for k in feasible_non1}
            ratio_sum = sum(target_ratio.values())
            target_ratio = {k: v / ratio_sum for k, v in target_ratio.items()}
            k = choose_k(feasible_non1, k_counts, len(out_rows), qid, target_ratio)
        else:
            forced_one += 1
            k = 1

        # If answer has more than needed, sample deterministically.
        selected_answers = shuffled_by_id(answers, qid, "answers")[:k]
        # If candidates are more than needed, keep front ones as requested.
        selected_wrong = candidates[: (4 - k)]

        combo = choose_combo(k, combo_counts, qid)
        options = [""] * 4

        # Put correct options into chosen slots, order randomized deterministically.
        ans_for_slots = shuffled_by_id(selected_answers, qid, "answer_slots")
        for idx, opt_pos in enumerate(combo):
            options[opt_pos] = ans_for_slots[idx]

        # Fill remaining slots with wrong options in current order.
        wrong_iter = iter(selected_wrong)
        for i in range(4):
            if not options[i]:
                options[i] = next(wrong_iter)

        answer_labels = [LABELS[i] for i in combo]
        answer_labels.sort()

        out_rows.append(
            {
                "id": qid,
                "question": question,
                "options": {LABELS[i]: options[i] for i in range(4)},
                "answer": answer_labels,
            }
        )

        k_counts[k] += 1
        combo_counts[k][combo] += 1

    written = write_jsonl(output_path, out_rows)
    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_rows": len(rows),
        "written_rows": written,
        "skipped_rows": skipped,
        "forced_one_rows": forced_one,
        "k_counts": k_counts,
        "combo_counts": {str(k): {str(c): v for c, v in combo_counts[k].items()} for k in K_VALUES},
    }


def should_include_file(file_name: str) -> bool:
    if file_name == "task5.jsonl":
        return False
    if file_name.startswith("task3_") and file_name != "task3_subtask3.jsonl":
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 4-option QA jsonl for evaluation.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input jsonl directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output jsonl directory.")
    parser.add_argument("--glob", default="*.jsonl", help="Input file glob pattern.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted([p for p in args.input_dir.glob(args.glob) if p.is_file() and should_include_file(p.name)])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: List[Dict[str, Any]] = []
    total_in = 0
    total_out = 0
    total_skip = 0
    total_forced_one = 0
    agg_k = {1: 0, 2: 0, 3: 0}

    for fp in files:
        out_path = args.output_dir / fp.name
        rep = convert_file(fp, out_path)
        reports.append(rep)
        total_in += rep["input_rows"]
        total_out += rep["written_rows"]
        total_skip += rep["skipped_rows"]
        total_forced_one += rep["forced_one_rows"]
        for k in K_VALUES:
            agg_k[k] += rep["k_counts"][k]

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "file_count": len(files),
        "total_input_rows": total_in,
        "total_written_rows": total_out,
        "total_skipped_rows": total_skip,
        "total_forced_one_rows": total_forced_one,
        "target_ratio_non1": TARGET_RATIO_NON1,
        "global_k_counts": agg_k,
        "global_k_ratio_realized": {
            str(k): (agg_k[k] / total_out if total_out else 0.0) for k in K_VALUES
        },
        "files": reports,
    }

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
