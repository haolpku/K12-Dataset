#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: 根据 sections_index.json 将整本书 Markdown 切分为分章节文件。

读取 segment_textbooks.py 生成的 sections_index.json，将完整教材 Markdown
拆分为独立的章节 Markdown 文件，同时清理图片链接等冗余内容。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segment_textbooks import discover_books, build_file_blocks_from_index, cleanup_markdown


def split_one_book(record, output_root: Path) -> tuple[bool, str, dict]:
    out_grade = output_root / record.subject / record.stage / record.version / record.grade
    out_sections = out_grade / "out_sections"

    index_path = out_sections / "sections_index.json"
    if not index_path.exists():
        return False, "missing_index", {"index_path": str(index_path)}

    try:
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, "invalid_index", {"error": str(e), "index_path": str(index_path)}

    raw_items = index_data.get("sections") if isinstance(index_data, dict) else None
    if not isinstance(raw_items, list):
        return False, "invalid_index", {"error": "sections is not a list", "index_path": str(index_path)}

    index_items = []
    for item in raw_items:
        if isinstance(item, dict):
            index_items.append({k: str(v) for k, v in item.items()})

    content = record.source_md.read_text(encoding="utf-8", errors="ignore")
    metadata_text, blocks, info = build_file_blocks_from_index(content, index_items, stage=record.stage)
    if not blocks:
        return False, "no_blocks_from_index", info

    out_sections.mkdir(parents=True, exist_ok=True)
    (out_sections / "metadata.md").write_text(cleanup_markdown(metadata_text), encoding="utf-8")

    # Keep split step deterministic: write exactly what index resolves to.
    for filename, cleaned in blocks:
        (out_sections / filename).write_text(cleaned, encoding="utf-8")

    return True, "ok", {"written": len(blocks), **info}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: split markdown based on existing sections_index.json")
    parser.add_argument("--input-root", required=True, help="Path to ChinaTextbook_after_mineru root")
    parser.add_argument("--output-root", required=True, help="Path to k12_graphbench/output root")
    parser.add_argument("--stage", default=None, help="Filter stage, e.g. 小学/初中/高中")
    parser.add_argument("--subject", default=None, help="Filter subject")
    parser.add_argument("--version", default=None, help="Filter version")
    parser.add_argument("--grade", default=None, help="Filter grade")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N books")

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    summary = {"processed": 0, "success": 0, "failed": 0, "failures": []}

    count = 0
    for rec in discover_books(
        input_root,
        stage=args.stage,
        subject=args.subject,
        version=args.version,
        grade=args.grade,
    ):
        ok, status, info = split_one_book(rec, output_root)
        summary["processed"] += 1

        if ok:
            summary["success"] += 1
            print(f"[OK][split] {rec.subject}/{rec.stage}/{rec.version}/{rec.grade}/{rec.book_name} -> {info.get('written', 0)}")
        else:
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "source_md": str(rec.source_md),
                    "subject": rec.subject,
                    "stage": rec.stage,
                    "version": rec.version,
                    "grade": rec.grade,
                    "book_name": rec.book_name,
                    "status": status,
                    "info": info,
                }
            )
            print(f"[FAIL][split] {rec.subject}/{rec.stage}/{rec.version}/{rec.grade}/{rec.book_name} ({status})")

        count += 1
        if args.limit and count >= args.limit:
            break

    print("\n===== SUMMARY =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
