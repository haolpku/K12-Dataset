#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from openai import AsyncOpenAI


@dataclass
class FilterResult:
    sample_id: str
    status: str
    kept_candidates: List[str]
    raw_output: str
    reason: str


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


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    for v in values:
        if not isinstance(v, str):
            continue
        t = v.strip()
        if t:
            out.append(t)
    return out


def load_done_ids_from_output(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    done: Set[str] = set()
    for row in read_jsonl(path):
        sid = row.get("id")
        if isinstance(sid, str) and sid.strip():
            done.add(sid.strip())
    return done


def build_user_prompt(question: str, answers: List[str], candidates: List[str]) -> str:
    ans_lines = [f"- {a}" for a in answers]
    cand_lines = [f"- {c}" for c in candidates]
    return (
        "你将看到一到题目、正确答案和若干候选干扰选项。"
        "请设想真实的教学场景，返回所有【可以是正确选项、或与正确答案同义/近义】的候选干扰项。\n"
        "注意：\n"
        "1) 只能从给定 candidates 中选择，不能改写、不能新增。\n"
        "2) 输出必须是严格 JSON：{\"drop_candidates\": [\"候选1\", \"候选2\"]}\n"
        "3) 若都可以保留，可返回空数组；若都不应保留，可返回与整个候选干扰项一致的数组。\n\n"
        f"题目：\n{question}\n\n"
        f"正确答案：\n{os.linesep.join(ans_lines)}\n\n"
        f"候选干扰项：\n{os.linesep.join(cand_lines)}\n"
    )


def parse_candidates_from_model(raw_output: str, original_candidates: List[str]) -> tuple[bool, List[str]]:
    try:
        obj = json.loads(raw_output)
    except json.JSONDecodeError:
        return False, []
    if not isinstance(obj, dict):
        return False, []

    original_set = set(original_candidates)
    vals_keep = obj.get("keep_candidates")
    vals_drop = obj.get("drop_candidates")
    if isinstance(vals_keep, list):
        keep_set = {
            x.strip()
            for x in vals_keep
            if isinstance(x, str) and x.strip() and x.strip() in original_set
        }
        # Keep original order to minimize downstream perturbation.
        return True, [c for c in original_candidates if c in keep_set]
    if isinstance(vals_drop, list):
        drop_set = {
            x.strip()
            for x in vals_drop
            if isinstance(x, str) and x.strip() and x.strip() in original_set
        }
        return True, [c for c in original_candidates if c not in drop_set]
    return False, []


async def call_model(
    client: AsyncOpenAI,
    model: str,
    question: str,
    answers: List[str],
    candidates: List[str],
    timeout: int,
) -> str:
    system_prompt = (
        "你是一个严谨的K12评测数据审校助手。"
        "你的任务是排除不合理的干扰选项。"
        "你必须仅输出JSON，且只包含键 keep_candidates 或 drop_candidates。"
    )
    user_prompt = build_user_prompt(question=question, answers=answers, candidates=candidates)
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=512,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


async def process_file(
    *,
    file_path: Path,
    output_path: Path,
    raw_path: Path,
    client: AsyncOpenAI,
    model: str,
    timeout: int,
    concurrency: int,
) -> Dict[str, Any]:
    rows = read_jsonl(file_path)
    done_ids = load_done_ids_from_output(output_path)
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()

    kept_rows: List[Dict[str, Any]] = []
    existing_out: Dict[str, Dict[str, Any]] = {}
    if output_path.exists():
        for r in read_jsonl(output_path):
            sid = str(r.get("id", "")).strip()
            if sid:
                existing_out[sid] = r
        kept_rows = list(existing_out.values())

    stats = {
        "input_rows": len(rows),
        "skipped_before_lt4": 0,
        "skipped_parse_fail": 0,
        "skipped_after_lt4": 0,
        "kept_rows": len(kept_rows),
    }

    async def worker(sample: Dict[str, Any]) -> None:
        sid = str(sample.get("id", "")).strip()
        if not sid:
            return
        if sid in done_ids:
            return
        answers = clean_text_list(sample.get("answer"))
        candidates = clean_text_list(sample.get("candidates"))
        if len(answers) + len(candidates) < 4:
            row = {
                "id": sid,
                "status": "skip_before_lt4",
                "reason": "len(answer)+len(candidates)<4 before filtering",
                "raw_output": "",
                "kept_candidates": [],
            }
            async with lock:
                append_jsonl(raw_path, row)
                stats["skipped_before_lt4"] += 1
            return

        question = str(sample.get("question", "")).strip()
        async with sem:
            try:
                raw_output = await call_model(
                    client=client,
                    model=model,
                    question=question,
                    answers=answers,
                    candidates=candidates,
                    timeout=timeout,
                )
            except Exception as e:  # noqa: BLE001
                row = {
                    "id": sid,
                    "status": "skip_parse_fail",
                    "reason": f"api_error: {type(e).__name__}: {e}",
                    "raw_output": "",
                    "kept_candidates": [],
                }
                async with lock:
                    append_jsonl(raw_path, row)
                    stats["skipped_parse_fail"] += 1
                return

        parsed_ok, kept = parse_candidates_from_model(raw_output=raw_output, original_candidates=candidates)
        if not parsed_ok:
            status = "skip_parse_fail"
            reason = "model output json parse failed or missing keep_candidates/drop_candidates"
            row = {
                "id": sid,
                "status": status,
                "reason": reason,
                "raw_output": raw_output,
                "kept_candidates": kept,
            }
            async with lock:
                append_jsonl(raw_path, row)
                stats["skipped_parse_fail"] += 1
            return

        if len(answers) + len(kept) < 4:
            row = {
                "id": sid,
                "status": "skip_after_lt4",
                "reason": "len(answer)+len(kept_candidates)<4 after filtering",
                "raw_output": raw_output,
                "kept_candidates": kept,
            }
            async with lock:
                append_jsonl(raw_path, row)
                stats["skipped_after_lt4"] += 1
            return

        new_row = dict(sample)
        new_row["candidates"] = kept
        log_row = {
            "id": sid,
            "status": "kept",
            "reason": "",
            "raw_output": raw_output,
            "kept_candidates": kept,
        }
        async with lock:
            append_jsonl(raw_path, log_row)
            existing_out[sid] = new_row
            stats["kept_rows"] = len(existing_out)

    await asyncio.gather(*(worker(s) for s in rows))
    final_rows = list(existing_out.values())
    write_jsonl(output_path, final_rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT-4o to filter distractor candidates in benchmark jsonl files.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing reranked *.jsonl files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for filtered *.jsonl files.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory for per-sample raw model outputs (*.raw.jsonl).")
    parser.add_argument("--glob", type=str, default="*.jsonl", help="Input file glob.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible API base URL. If empty, read from env first.")
    parser.add_argument("--base-url-env", type=str, default="OPENAI_BASE_URL", help="Environment variable for API base URL.")
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API requests per file.")
    parser.add_argument("--max-samples-per-file", type=int, default=0, help="0 means no limit; useful for cheap dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    env_base_url = os.getenv(args.base_url_env, "").strip()
    base_url = args.base_url.strip() or env_base_url or "https://api.openai.com/v1"
    if not api_key:
        raise ValueError(f"environment variable not set: {args.api_key_env}")
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    files = sorted([p for p in args.input_dir.glob(args.glob) if p.is_file() and p.suffix == ".jsonl"])
    summary: Dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "raw_dir": str(args.raw_dir),
        "file_count": len(files),
        "model": args.model,
        "base_url": base_url,
        "files": [],
    }

    for fp in files:
        output_path = args.output_dir / fp.name
        raw_path = args.raw_dir / f"{fp.stem}.raw.jsonl"
        if args.max_samples_per_file > 0:
            rows = read_jsonl(fp)[: args.max_samples_per_file]
            tmp_input = args.raw_dir / f"{fp.stem}.tmp_input.jsonl"
            write_jsonl(tmp_input, rows)
            file_for_run = tmp_input
        else:
            file_for_run = fp
        stats = asyncio.run(
            process_file(
                file_path=file_for_run,
                output_path=output_path,
                raw_path=raw_path,
                client=client,
                model=args.model,
                timeout=args.timeout,
                concurrency=args.concurrency,
            )
        )
        one = {
            "file": fp.name,
            "output_file": str(output_path),
            "raw_file": str(raw_path),
            **stats,
        }
        summary["files"].append(one)
        print(json.dumps(one, ensure_ascii=False))

    summary_path = args.raw_dir / "filter_by_4o_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"summary_file": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
