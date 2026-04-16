#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate multi-select K12 benchmark JSONL via an OpenAI-compatible Chat API.

Loads optional ``eval/configs/.env`` (not repo ``config/.env``) for model
YAML ``${VAR}`` expansion, writes per-input prediction JSONL, and emits ``summary.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import yaml
from openai import AsyncOpenAI

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.envfile import load_env_file

LABELS = ("A", "B", "C", "D")
LABEL_SET = set(LABELS)


@dataclass
class SampleResult:
    sample_id: str
    gold: List[str]
    pred: List[str]
    raw_output: str
    exact_match: float
    precision: float
    recall: float
    f1: float


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be object: {path}")
    return expand_env(data)


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_str(value: str) -> str:
    def _repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")

    return _ENV_VAR_RE.sub(_repl, value)


def expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env_str(obj)
    if isinstance(obj, list):
        return [expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    return obj


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


def clean_label_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        if not isinstance(v, str):
            continue
        t = v.strip().upper()
        if t in LABEL_SET and t not in seen:
            seen.add(t)
            out.append(t)
    return sorted(out)


def build_user_prompt(sample: Dict[str, Any], tmpl: str) -> str:
    options = sample.get("options", {}) if isinstance(sample.get("options"), dict) else {}
    vars_map = {
        "question": str(sample.get("question", "")).strip(),
        "A": str(options.get("A", "")).strip(),
        "B": str(options.get("B", "")).strip(),
        "C": str(options.get("C", "")).strip(),
        "D": str(options.get("D", "")).strip(),
    }
    return tmpl.format(**vars_map)


def parse_prediction(text: str) -> List[str]:
    raw = (text or "").strip()
    if raw.startswith("__ERROR__") or raw == "__EMPTY_RESPONSE__":
        return []

    t = raw.upper()
    t = t.replace("，", ",").replace("、", ",").replace("；", ",")
    if "</THINK>" in t:
        t = t.split("</THINK>")[-1].strip()
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    if lines:
        letters = [ch for ch in lines[0] if ch in LABEL_SET]
        if letters:
            return sorted(set(letters))
    letters = re.findall(r"[ABCD]", t)
    return sorted(set(letters))


def score_prediction(gold: Sequence[str], pred: Sequence[str]) -> Dict[str, float]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact = 1.0 if gold_set == pred_set else 0.0
    return {
        "exact_match": exact,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate(results: Sequence[SampleResult]) -> Dict[str, float]:
    n = len(results)
    if n == 0:
        return {"count": 0, "exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        "count": n,
        "exact_match": sum(x.exact_match for x in results) / n,
        "precision": sum(x.precision for x in results) / n,
        "recall": sum(x.recall for x in results) / n,
        "f1": sum(x.f1 for x in results) / n,
    }


async def call_model(
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
    extra_body: Dict[str, Any] | None = None,
) -> str:
    req: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if isinstance(reasoning_effort, str) and reasoning_effort.strip():
        req["reasoning_effort"] = reasoning_effort.strip()
    if isinstance(extra_body, dict) and extra_body:
        req["extra_body"] = extra_body
    resp = await client.chat.completions.create(**req, timeout=timeout)
    return (resp.choices[0].message.content or "").strip()


def load_done_ids(pred_path: Path) -> Set[str]:
    if not pred_path.exists():
        return set()
    done: Set[str] = set()
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("id")
            if isinstance(sid, str) and sid.strip():
                done.add(sid.strip())
    return done


async def eval_file_async(
    *,
    fp: Path,
    rows: List[Dict[str, Any]],
    pred_path: Path,
    client: AsyncOpenAI,
    model_name: str,
    system_prompt: str,
    user_tmpl: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None,
    extra_body: Dict[str, Any] | None,
    concurrency: int,
) -> List[SampleResult]:
    done_ids = load_done_ids(pred_path)
    pending_rows = [r for r in rows if str(r.get("id", "")).strip() not in done_ids]
    if not pending_rows:
        # Build results from existing file so summary still works.
        existing_results: List[SampleResult] = []
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing_results.append(
                    SampleResult(
                        sample_id=str(obj.get("id", "")),
                        gold=clean_label_list(obj.get("gold")),
                        pred=clean_label_list(obj.get("pred")),
                        raw_output=str(obj.get("raw_output", "")),
                        exact_match=float(obj.get("exact_match", 0.0)),
                        precision=float(obj.get("precision", 0.0)),
                        recall=float(obj.get("recall", 0.0)),
                        f1=float(obj.get("f1", 0.0)),
                    )
                )
        return existing_results

    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    collected: List[SampleResult] = []

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = pred_path.open("a", encoding="utf-8")

    async def worker(sample: Dict[str, Any]) -> None:
        async with sem:
            sid = str(sample.get("id", "")).strip()
            gold = clean_label_list(sample.get("answer"))
            user_prompt = build_user_prompt(sample, user_tmpl)
            try:
                raw_output = await call_model(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body,
                )
                raw_output = raw_output.strip() if raw_output.strip() else "__EMPTY_RESPONSE__"
            except Exception as e:
                raw_output = f"__ERROR__ {type(e).__name__}: {e}"
            pred = parse_prediction(raw_output)
            metrics = score_prediction(gold, pred)
            one = SampleResult(
                sample_id=sid,
                gold=gold,
                pred=pred,
                raw_output=raw_output,
                exact_match=metrics["exact_match"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
            )
            row = {
                "id": sid,
                "gold": gold,
                "pred": pred,
                "raw_output": raw_output,
                "exact_match": one.exact_match,
                "precision": one.precision,
                "recall": one.recall,
                "f1": one.f1,
            }
            async with lock:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                collected.append(one)

    try:
        await asyncio.gather(*(worker(s) for s in pending_rows))
    finally:
        out_f.close()

    # Reload all rows from file to include previous resume history.
    all_results: List[SampleResult] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            all_results.append(
                SampleResult(
                    sample_id=str(obj.get("id", "")),
                    gold=clean_label_list(obj.get("gold")),
                    pred=clean_label_list(obj.get("pred")),
                    raw_output=str(obj.get("raw_output", "")),
                    exact_match=float(obj.get("exact_match", 0.0)),
                    precision=float(obj.get("precision", 0.0)),
                    recall=float(obj.get("recall", 0.0)),
                    f1=float(obj.get("f1", 0.0)),
                )
            )
    return all_results


def parse_args() -> argparse.Namespace:
    bench_dir = Path(__file__).resolve().parent
    repo_root = bench_dir.parent
    parser = argparse.ArgumentParser(description="K12 multi-select benchmark evaluation")
    parser.add_argument(
        "--model",
        required=True,
        help="Stem of ``eval/configs/models/<name>.yaml`` (without ``.yaml``), e.g. gpt4o",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of input ``*.jsonl`` (default: <repo>/data/benchmark/benchmark_qa)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/workspace/benchmark_output/<model>)",
    )
    parser.add_argument("--glob", default="*.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-file", type=int, default=0, help="0 means no per-file cap")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent API requests")
    args = parser.parse_args()
    stem = str(args.model).strip()
    if not stem:
        parser.error("--model must be non-empty")
    args.model_yaml = bench_dir / "configs" / "models" / f"{stem}.yaml"
    args.task_yaml = bench_dir / "configs" / "task_k12_multiselect.yaml"
    if args.input_dir is None:
        args.input_dir = repo_root / "data" / "benchmark" / "benchmark_qa"
    else:
        args.input_dir = Path(args.input_dir).expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = repo_root / "workspace" / "benchmark_output" / stem
    else:
        args.output_dir = Path(args.output_dir).expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # Optional local env for benchmark models only (not repo config/.env).
    configs_dir = args.model_yaml.resolve().parent.parent
    load_env_file(configs_dir / ".env")

    if not args.model_yaml.is_file():
        raise FileNotFoundError(f"Model config not found: {args.model_yaml}")
    if not args.task_yaml.is_file():
        raise FileNotFoundError(f"Task config not found: {args.task_yaml}")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory missing or not a directory: {args.input_dir}")

    model_cfg = read_yaml(args.model_yaml)
    task_cfg = read_yaml(args.task_yaml)

    api_cfg = model_cfg.get("api", {})
    req_cfg = model_cfg.get("request", {})
    if not isinstance(api_cfg, dict) or not isinstance(req_cfg, dict):
        raise ValueError("model config must contain `api` and `request` objects")

    base_url_env = str(api_cfg.get("base_url_env", "")).strip()
    base_url = os.getenv(base_url_env, "").strip() if base_url_env else str(api_cfg.get("base_url", "")).strip()
    model_name = str(api_cfg.get("model", "")).strip()

    api_key_raw = str(api_cfg.get("api_key", "")).strip()
    key_env = str(api_cfg.get("api_key_env", "")).strip()
    if api_key_raw:
        api_key = api_key_raw
    elif key_env:
        api_key = os.getenv(key_env, "")
    else:
        # vLLM / local OpenAI-compatible servers often don't require a real key,
        # but the OpenAI SDK expects a non-empty string.
        api_key = "EMPTY"
    if not base_url or not model_name:
        raise ValueError("model config missing api.base_url or api.model")
    if key_env and not api_key:
        raise ValueError(f"environment variable not set: {key_env}")

    system_prompt = str(task_cfg.get("system_prompt", "")).strip()
    user_tmpl = str(task_cfg.get("user_prompt_template", "")).strip()
    if not system_prompt or not user_tmpl:
        raise ValueError("task config missing system_prompt or user_prompt_template")

    temperature = float(req_cfg.get("temperature", 0.0))
    top_p = float(req_cfg.get("top_p", 1.0))
    max_tokens = int(req_cfg.get("max_tokens", 16))
    timeout = int(req_cfg.get("timeout", 120))
    reasoning_effort = req_cfg.get("reasoning_effort")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise ValueError("request.reasoning_effort must be a string when provided")
    extra_body = req_cfg.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("request.extra_body must be an object when provided")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    files = sorted([p for p in args.input_dir.glob(args.glob) if p.is_file() and p.suffix == ".jsonl"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "model": str(args.model).strip(),
        "model_yaml": str(args.model_yaml),
        "task_yaml": str(args.task_yaml),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "files": [],
    }
    all_results: List[SampleResult] = []

    for fp in files:
        rows = read_jsonl(fp)
        if args.max_samples_per_file > 0:
            rows = rows[: args.max_samples_per_file]

        pred_path = args.output_dir / f"{fp.stem}.predictions.jsonl"
        file_results = asyncio.run(
            eval_file_async(
                fp=fp,
                rows=rows,
                pred_path=pred_path,
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_tmpl=user_tmpl,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
                concurrency=args.concurrency,
            )
        )
        all_results.extend(file_results)
        file_summary = aggregate(file_results)
        file_summary["file"] = fp.name
        file_summary["prediction_file"] = str(pred_path)
        summary["files"].append(file_summary)

    summary["overall"] = aggregate(all_results)

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
