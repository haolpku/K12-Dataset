#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-command SFT QA pipeline: subject-stage KG -> parts -> final.

This script intentionally keeps the logic simple and delegates heavy work to:
- src/sft_qa/generate_qa.py
- src/sft_qa/exercise_to_qa.py (optional, uses data/afterclass_exercises)
- src/sft_qa/merge_qa.py

Outputs are written under workspace/sft_qa/<subject_stage>/ by default.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="一键运行 sft_qa：generate_qa -> (exercise_to_qa) -> merge_qa")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--subject-stage", required=True, help="例如 math_primaryschool / physics_middleschool 等")
    parser.add_argument("--workspace-dir", type=str, default=None, help="覆盖默认 workspace/sft_qa/<subject_stage>")
    parser.add_argument("--input-json", type=str, default=None, help="覆盖 generate_qa 默认输入 subject_stage_kg JSON")
    parser.add_argument("--tasks", type=str, default=None, help="覆盖 generate_qa tasks（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="覆盖 generate_qa limit（每个 task）")
    parser.add_argument("--resume", action="store_true", help="generate_qa resume 模式")
    parser.add_argument("--model", type=str, default=None, help="覆盖 LLM model")
    parser.add_argument("--temperature", type=float, default=None, help="覆盖 temperature")

    parser.add_argument("--with-exercises", action="store_true", help="同时生成课后题 exercise.jsonl（默认不跑）")
    parser.add_argument("--afterclass-dir", type=str, default=None, help="覆盖 afterclass_exercises 目录（默认 data/afterclass_exercises）")
    parser.add_argument("--exercise-mode", type=str, default=None, help="覆盖 exercise_to_qa --mode")
    parser.add_argument("--exercise-allowed-types", type=str, default=None, help="覆盖 exercise_to_qa --allowed-types")

    parser.add_argument("--python", type=str, default=None, help="覆盖 python 解释器路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要运行的命令")
    parser.set_defaults(repo_root=repo_root)
    return parser.parse_args()


def run(cmd: List[str], *, dry_run: bool) -> None:
    print("[cmd]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root: Path = args.repo_root
    py = args.python or sys.executable

    gen_py = repo_root / "src" / "sft_qa" / "generate_qa.py"
    ex_py = repo_root / "src" / "sft_qa" / "exercise_to_qa.py"
    merge_py = repo_root / "src" / "sft_qa" / "merge_qa.py"

    gen_cmd: List[str] = [py, str(gen_py), "--subject-stage", str(args.subject_stage)]
    merge_cmd: List[str] = [py, str(merge_py), "--subject-stage", str(args.subject_stage)]

    if args.config:
        gen_cmd += ["--config", str(args.config)]
        merge_cmd += ["--config", str(args.config)]
    if args.workspace_dir:
        gen_cmd += ["--workspace-dir", str(args.workspace_dir)]
        merge_cmd += ["--workspace-dir", str(args.workspace_dir)]
    if args.input_json:
        gen_cmd += ["--input-json", str(args.input_json)]
    if args.tasks:
        gen_cmd += ["--tasks", str(args.tasks)]
    if args.limit is not None:
        gen_cmd += ["--limit", str(int(args.limit))]
    if args.resume:
        gen_cmd += ["--resume"]
    if args.model:
        gen_cmd += ["--model", str(args.model)]
    if args.temperature is not None:
        gen_cmd += ["--temperature", str(float(args.temperature))]

    run(gen_cmd, dry_run=bool(args.dry_run))

    if args.with_exercises:
        ex_cmd: List[str] = [py, str(ex_py), "--subject-stage", str(args.subject_stage)]
        if args.config:
            ex_cmd += ["--config", str(args.config)]
        if args.workspace_dir:
            ex_cmd += ["--workspace-dir", str(args.workspace_dir)]
        if args.afterclass_dir:
            ex_cmd += ["--afterclass-dir", str(args.afterclass_dir)]
        if args.exercise_mode:
            ex_cmd += ["--mode", str(args.exercise_mode)]
        if args.exercise_allowed_types is not None:
            ex_cmd += ["--allowed-types", str(args.exercise_allowed_types)]
        run(ex_cmd, dry_run=bool(args.dry_run))

    run(merge_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()

