#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single entrypoint for benchmark generation (candidates → four-option QA).

Runs ``generate_benchmark.py`` then ``build_qa.py`` with repository-default paths,
forwarding optional embedding and task overrides.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="一键运行 benchmark：generate_benchmark -> build_qa")
    parser.add_argument("--kg-dir", type=str, default=None, help="默认 data/global_kg")
    parser.add_argument("--subject-kg-dir", type=str, default=None, help="默认 data/subject_kg")
    parser.add_argument("--candidates-dir", type=str, default=None, help="默认 data/benchmark_candidates")
    parser.add_argument("--qa-dir", type=str, default=None, help="默认 data/benchmark_qa")
    parser.add_argument("--embedding-model", type=str, default=None, help="覆盖 embedding model")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="覆盖 embedding batch size")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="逗号分隔任务名（同 generate_benchmark.py --tasks）",
    )
    parser.add_argument("--glob", type=str, default=None, help="覆盖 build_qa.py --glob")
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

    gen_py = repo_root / "src" / "benchmark" / "generate_benchmark.py"
    build_py = repo_root / "src" / "benchmark" / "build_qa.py"

    kg_dir = Path(args.kg_dir).resolve() if args.kg_dir else repo_root / "data" / "global_kg"
    subject_kg_dir = Path(args.subject_kg_dir).resolve() if args.subject_kg_dir else repo_root / "data" / "subject_kg"
    candidates_dir = Path(args.candidates_dir).resolve() if args.candidates_dir else repo_root / "data" / "benchmark_candidates"
    qa_dir = Path(args.qa_dir).resolve() if args.qa_dir else repo_root / "data" / "benchmark_qa"

    gen_cmd: List[str] = [
        py,
        str(gen_py),
        "--kg-dir",
        str(kg_dir),
        "--subject-kg-dir",
        str(subject_kg_dir),
        "--output-dir",
        str(candidates_dir),
    ]
    if args.embedding_model:
        gen_cmd += ["--embedding-model", str(args.embedding_model)]
    if args.embedding_batch_size:
        gen_cmd += ["--embedding-batch-size", str(args.embedding_batch_size)]
    if args.tasks:
        gen_cmd += ["--tasks", *[t.strip() for t in str(args.tasks).split(",") if t.strip()]]

    build_cmd: List[str] = [
        py,
        str(build_py),
        "--input-dir",
        str(candidates_dir),
        "--output-dir",
        str(qa_dir),
        "--subject-kg-dir",
        str(subject_kg_dir),
    ]
    if args.glob:
        build_cmd += ["--glob", str(args.glob)]

    run(gen_cmd, dry_run=bool(args.dry_run))
    run(build_cmd, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()

