#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


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
                raise ValueError(f"{path} line {i}: invalid json: {e}") from e
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


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text.strip() if not ch.isspace())


def char_ngrams(text: str, n: int = 3) -> Counter:
    t = normalize_text(text)
    if not t:
        return Counter()
    if len(t) < n:
        return Counter({t: 1})
    return Counter(t[i : i + n] for i in range(len(t) - n + 1))


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, v in a.items():
        if k in b:
            dot += v * b[k]
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def text_similarity(a: str, b: str, ngram_n: int) -> float:
    # Mix short and long character n-grams to reduce all-zero scores on short Chinese terms.
    sim_1 = cosine_counter(char_ngrams(a, 1), char_ngrams(b, 1))
    sim_2 = cosine_counter(char_ngrams(a, 2), char_ngrams(b, 2))
    sim_n = cosine_counter(char_ngrams(a, ngram_n), char_ngrams(b, ngram_n))
    return 0.2 * sim_1 + 0.3 * sim_2 + 0.5 * sim_n


def clean_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for x in values:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if t:
            out.append(t)
    return out


def score_candidate(
    candidate: str,
    query_text: str,
    answers: List[str],
    ngram_n: int,
) -> Dict[str, float]:
    sim_query = text_similarity(candidate, query_text, ngram_n)
    sim_answer = 0.0
    for ans in answers:
        sim_answer = max(sim_answer, text_similarity(candidate, ans, ngram_n))
    score = (sim_query + sim_answer) / 2.0
    return {
        "sim_query": sim_query,
        "sim_answer": sim_answer,
        "score": score,
    }


def pick_query_text(sample: Dict[str, Any]) -> str:
    q = sample.get("question")
    if isinstance(q, str) and q.strip():
        return q.strip()
    # Fallback if question is unavailable.
    meta = sample.get("meta")
    if isinstance(meta, dict):
        for key in ("query_node_name", "query_node_id", "source_exercise_id"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def score_sample(
    sample: Dict[str, Any],
    ngram_n: int,
) -> Dict[str, Any]:
    query_text = pick_query_text(sample)
    answers = clean_list(sample.get("answer"))
    candidates = clean_list(sample.get("candidates"))

    scored_candidates: List[Dict[str, Any]] = []
    for c in candidates:
        s = score_candidate(
            candidate=c,
            query_text=query_text,
            answers=answers,
            ngram_n=ngram_n,
        )
        scored_candidates.append(
            {
                "candidate": c,
                "sim_query": round(s["sim_query"], 6),
                "sim_answer": round(s["sim_answer"], 6),
                "score": round(s["score"], 6),
            }
        )

    scored_candidates.sort(key=lambda x: (x["score"], x["sim_query"]), reverse=True)

    out = dict(sample)
    out["candidate_scores"] = scored_candidates
    out["scoring_meta"] = {
        "query_text": query_text,
        "ngram_n": ngram_n,
        "formula": "score = (sim_query + sim_answer) / 2",
    }
    return out


def collect_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [p for p in input_path.glob("*.jsonl") if p.is_file()],
            key=lambda p: p.name,
        )
    raise FileNotFoundError(f"input path not found: {input_path}")


def build_preview(rows: List[Dict[str, Any]], sample_count: int, min_candidates: int) -> List[Dict[str, Any]]:
    eligible = [r for r in rows if len(clean_list(r.get("candidates"))) >= min_candidates]
    eligible.sort(key=lambda r: len(clean_list(r.get("candidates"))), reverse=True)
    picked = eligible[:sample_count]
    preview: List[Dict[str, Any]] = []
    for r in picked:
        preview.append(
            {
                "id": r.get("id"),
                "task": r.get("task"),
                "subtask": r.get("subtask"),
                "answer": clean_list(r.get("answer")),
                "candidate_count": len(clean_list(r.get("candidates"))),
                "candidate_scores": r.get("candidate_scores", []),
            }
        )
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score benchmark candidates by text similarity.")
    parser.add_argument("--input", type=Path, required=True, help="Input jsonl file or directory containing jsonl files.")
    parser.add_argument("--sim-score-dir", type=Path, required=True, help="Where scored jsonl and preview json are written.")
    parser.add_argument("--rerank-dir", type=Path, required=True, help="Where reranked jsonl files are written.")
    parser.add_argument("--suspicious-path", type=Path, required=True, help="Where suspicious (sim_answer > threshold) report is written.")
    parser.add_argument("--ngram-n", type=int, default=3, help="Character n-gram size for cosine similarity.")
    parser.add_argument("--suspicious-threshold", type=float, default=0.9, help="Candidates with sim_answer above this are reported.")
    parser.add_argument("--preview-count", type=int, default=5, help="How many high-candidate samples to preview.")
    parser.add_argument("--preview-min-candidates", type=int, default=8, help="Minimum candidates to be eligible for preview.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = collect_input_files(args.input)
    args.sim_score_dir.mkdir(parents=True, exist_ok=True)
    args.rerank_dir.mkdir(parents=True, exist_ok=True)
    args.suspicious_path.parent.mkdir(parents=True, exist_ok=True)

    run_summary: Dict[str, Any] = {
        "input": str(args.input),
        "file_count": len(files),
        "ngram_n": args.ngram_n,
        "score_formula": "score = (sim_query + sim_answer) / 2",
        "suspicious_threshold": args.suspicious_threshold,
        "outputs": [],
    }
    suspicious_rows: List[Dict[str, Any]] = []

    for fp in files:
        rows = read_jsonl(fp)
        scored_rows = [
            score_sample(
                sample=r,
                ngram_n=args.ngram_n,
            )
            for r in rows
        ]

        scored_path = args.sim_score_dir / f"{fp.stem}.scored.jsonl"
        preview_path = args.sim_score_dir / f"{fp.stem}.preview.json"
        rerank_path = args.rerank_dir / fp.name
        write_jsonl(scored_path, scored_rows)

        preview = build_preview(
            scored_rows,
            sample_count=args.preview_count,
            min_candidates=args.preview_min_candidates,
        )
        with preview_path.open("w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)
            f.write("\n")

        reranked_rows: List[Dict[str, Any]] = []
        for scored_row, raw_row in zip(scored_rows, rows):
            # Keep all fields unchanged, only reorder candidates.
            new_row = dict(raw_row)
            new_row["candidates"] = [x["candidate"] for x in scored_row.get("candidate_scores", [])]
            reranked_rows.append(new_row)
            for c in scored_row.get("candidate_scores", []):
                if c.get("sim_answer", 0.0) > args.suspicious_threshold:
                    suspicious_rows.append(
                        {
                            "source_file": fp.name,
                            "id": raw_row.get("id"),
                            "task": raw_row.get("task"),
                            "subtask": raw_row.get("subtask"),
                            "question": raw_row.get("question"),
                            "answer": clean_list(raw_row.get("answer")),
                            "candidate": c.get("candidate"),
                            "sim_query": c.get("sim_query"),
                            "sim_answer": c.get("sim_answer"),
                            "score": c.get("score"),
                        }
                    )
        write_jsonl(rerank_path, reranked_rows)

        run_summary["outputs"].append(
            {
                "input_file": str(fp),
                "rows": len(rows),
                "scored_file": str(scored_path),
                "preview_file": str(preview_path),
                "rerank_file": str(rerank_path),
                "preview_samples": len(preview),
            }
        )

    suspicious_rows.sort(
        key=lambda x: (x.get("sim_answer", 0.0), x.get("score", 0.0)),
        reverse=True,
    )
    with args.suspicious_path.open("w", encoding="utf-8") as f:
        json.dump(suspicious_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary_path = args.sim_score_dir / "similarity_scoring_summary.json"
    run_summary["suspicious_count"] = len(suspicious_rows)
    run_summary["suspicious_path"] = str(args.suspicious_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(run_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
