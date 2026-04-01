#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 tests_concept_without_img.json + tests_concept_with_img.json（以及 skill 同理），
再生成统一 QA JSONL：tests_concept_qa.jsonl、tests_skill_qa.jsonl。

- 概念：question = 「"<source_stem>"这道题考察了什么概念？」, answer = target_name(s)，多个用逗号分隔
- 技能：question = 「"<source_stem>"这道题考察了什么方法？」, answer = target_name(s)，多个用逗号分隔
- id 为 item_1, item_2, ...（按合并后顺序）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "数学" / "小学" / "人教版"


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def build_qa_list(
    items: List[Dict[str, Any]],
    question_template: str,
) -> List[Dict[str, str]]:
    qa_list: List[Dict[str, str]] = []
    for idx, rec in enumerate(items, start=1):
        if not isinstance(rec, dict):
            continue
        source_stem = rec.get("source_stem") or ""
        targets = rec.get("targets") or []
        target_names = [t.get("target_name", "") for t in targets if isinstance(t, dict)]
        answer = "，".join(n for n in target_names if n)

        question = question_template.format(source_stem=source_stem)
        qa_list.append({
            "id": f"item_{idx}",
            "question": question,
            "answer": answer,
        })
    return qa_list


def main() -> None:
    # 概念：without_img + with_img 合并
    concept_without = load_json_list(OUT_DIR / "tests_concept_without_img.json")
    concept_with = load_json_list(OUT_DIR / "tests_concept_with_img.json")
    concept_merged = concept_without + concept_with

    if concept_merged:
        (OUT_DIR / "tests_concept.json").write_text(
            json.dumps(concept_merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        qas = build_qa_list(concept_merged, '"{source_stem}"这道题考察了什么概念？')
        (OUT_DIR / "tests_concept_qa.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in qas) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] 概念: 合并 {len(concept_without)} + {len(concept_with)} = {len(concept_merged)} 条 -> tests_concept.json, tests_concept_qa.jsonl")
    else:
        print("[warn] 概念: 无数据，跳过")

    # 技能：without_img + with_img 合并
    skill_without = load_json_list(OUT_DIR / "tests_skill_without_img.json")
    skill_with = load_json_list(OUT_DIR / "tests_skill_with_img.json")
    skill_merged = skill_without + skill_with

    if skill_merged:
        (OUT_DIR / "tests_skill.json").write_text(
            json.dumps(skill_merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        qas = build_qa_list(skill_merged, '"{source_stem}"这道题考察了什么方法？')
        (OUT_DIR / "tests_skill_qa.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in qas) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] 技能: 合并 {len(skill_without)} + {len(skill_with)} = {len(skill_merged)} 条 -> tests_skill.json, tests_skill_qa.jsonl")
    else:
        print("[warn] 技能: 无数据，跳过")


if __name__ == "__main__":
    main()
