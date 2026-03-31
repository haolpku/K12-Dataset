#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每本教材的 exercises 补全 answer/difficulty/type/analysis，并生成 tests_concept/tests_skill。

输入：
- output/**/out_sections/exercises_from_exercise_md_text_only.json
- output/**/raw_output/<source_md_stem>.json

输出：
- output/**/raw_output/exercise.json
- output/**/raw_output/exercise_llm_raw/*.txt  (每次 LLM 调用完整原文)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
BUILD_GRAPH_DIR = THIS_DIR.parent / "build_graph"
if str(BUILD_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_GRAPH_DIR))

from llm_client import create_llm_client  # type: ignore


ROUND1_KEYS = ["answer", "difficulty", "type", "analysis"]
ROUND2_KEYS = ["tests_concept", "tests_skill"]


@dataclass
class BookPaths:
    book_dir: Path
    input_exercises: Path
    raw_output_dir: Path
    output_exercise_json: Path
    raw_text_dir: Path


class ExerciseEnricher:
    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str], temperature: float) -> None:
        llm_kwargs: Dict[str, Any] = {"model": model}
        if api_key:
            llm_kwargs["api_key"] = api_key
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.client = create_llm_client(**llm_kwargs)
        self.temperature = temperature

    def _safe_print(self, msg: str) -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))

    @staticmethod
    def _extract_json_text(text: str) -> Optional[str]:
        if not text:
            return None

        fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            return text[start_obj:end_obj + 1]

        start_arr = text.find("[")
        end_arr = text.rfind("]")
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            return text[start_arr:end_arr + 1]

        return None

    @classmethod
    def _parse_json_loose(cls, text: str) -> Optional[Any]:
        json_text = cls._extract_json_text(text)
        if not json_text:
            return None
        try:
            return json.loads(json_text)
        except Exception:
            return None

    @staticmethod
    def _normalize_id_list(values: Any, allowed_ids: Optional[set[str]] = None) -> List[str]:
        if not isinstance(values, list):
            return []
        out: List[str] = []
        for v in values:
            if not isinstance(v, str):
                continue
            vv = v.strip()
            if not vv:
                continue
            if allowed_ids is not None and vv not in allowed_ids:
                continue
            if vv not in out:
                out.append(vv)
        return out
    
    @staticmethod
    def _map_ids_to_id_name(items: List[str], candidates: List[Dict[str, str]]) -> List[Dict[str, str]]:
        id_to_name: Dict[str, str] = {}
        for c in candidates:
            cid = c.get("id")
            cname = c.get("name")
            if isinstance(cid, str) and isinstance(cname, str) and cid.strip() and cname.strip():
                id_to_name[cid.strip()] = cname.strip()

        out: List[Dict[str, str]] = []
        for cid in items:
            if cid in id_to_name:
                out.append({"id": cid, "name": id_to_name[cid]})
        return out
    
    @staticmethod
    def _normalize_difficulty(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if 1 <= value <= 5 else None
        if isinstance(value, float) and value.is_integer():
            iv = int(value)
            return iv if 1 <= iv <= 5 else None
        if isinstance(value, str):
            v = value.strip()
            if v.isdigit():
                iv = int(v)
                return iv if 1 <= iv <= 5 else None
        return None

    def _call_llm(self, prompt: str, raw_txt_path: Path, max_tokens: int = 2000) -> str:
        response = self.client.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        raw_txt_path.parent.mkdir(parents=True, exist_ok=True)
        raw_txt_path.write_text(response if isinstance(response, str) else str(response), encoding="utf-8")
        return response

    @staticmethod
    def _build_round1_prompt(stem: str) -> str:
        return (
            "你是一位K12学科老师。\n"
            "请只根据下面这道题的题干，输出一个JSON对象，不要输出任何其他文字。\n"
            "JSON字段要求：\n"
            "- answer: 字符串，题目参考答案\n"
            "- difficulty: 数字1~5，题目的难度，数字越大代表难度越高\n"
            "- type: 字符串，题目的题型，取值为选择题、填空题、判断题、简答题、证明题、应用题等\n"
            "- analysis: 字符串，题目的解题过程分析，为可选字段\n\n"
            "题干如下：\n"
            f"{stem}\n"
        )

    @staticmethod
    def _build_round2_prompt(stem: str, concepts: List[Dict[str, str]], skills: List[Dict[str, str]]) -> str:
        candidates = {
            "concepts": concepts,
            "skills": skills,
        }
        return (
            "你是一位K12知识图谱标注专家。\n"
            "给定题干和候选概念/方法集合，请只输出一个JSON对象，不要输出任何其他文字。\n"
            "你只能从候选集合里**选择id**。\n"
            "JSON字段要求：\n"
            "- tests_concept: 字符串数组（可空）\n"
            "- tests_skill: 字符串数组（可空）\n\n"
            "题干如下：\n"
            f"{stem}\n\n"
            "候选集合如下（仅id和name）：\n"
            f"{json.dumps(candidates, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _extract_candidates_from_chapter_json(chapter_json_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if not chapter_json_path.exists():
            return [], []

        try:
            data = json.loads(chapter_json_path.read_text(encoding="utf-8"))
        except Exception:
            return [], []

        nodes = data.get("nodes", [])
        if not isinstance(nodes, list):
            return [], []

        concepts: List[Dict[str, str]] = []
        skills: List[Dict[str, str]] = []
        seen_c: set[str] = set()
        seen_s: set[str] = set()

        for node in nodes:
            if not isinstance(node, dict):
                continue
            label = node.get("label")
            node_id = node.get("id")
            props = node.get("properties", {}) if isinstance(node.get("properties"), dict) else {}
            name = props.get("name") if isinstance(props.get("name"), str) else None
            if not isinstance(node_id, str) or not node_id.strip() or not name or not name.strip():
                continue

            entry = {"id": node_id.strip(), "name": name.strip()}
            if label == "Concept":
                if entry["id"] not in seen_c:
                    seen_c.add(entry["id"])
                    concepts.append(entry)
            elif label == "Skill":
                if entry["id"] not in seen_s:
                    seen_s.add(entry["id"])
                    skills.append(entry)

        return concepts, skills

    @staticmethod
    def _to_chapter_json_path(raw_output_dir: Path, source_md: str) -> Path:
        source_name = Path(source_md).name
        stem = Path(source_name).stem
        return raw_output_dir / f"{stem}.json"

    
    @staticmethod
    def _is_completed_exercise(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        props = item.get("properties")
        if not isinstance(props, dict):
            return False

        answer_ok = isinstance(props.get("answer"), str) and bool(props.get("answer").strip())
        difficulty_ok = isinstance(props.get("difficulty"), int) and 1 <= props.get("difficulty") <= 5
        type_ok = isinstance(props.get("type"), str) and bool(props.get("type").strip())
        tc_ok = isinstance(props.get("tests_concept"), list)
        ts_ok = isinstance(props.get("tests_skill"), list)
        return answer_ok and difficulty_ok and type_ok and tc_ok and ts_ok

    
    @staticmethod
    def _build_existing_map(output_json_path: Path) -> Dict[str, Dict[str, Any]]:
        if not output_json_path.exists():
            return {}
        try:
            data = json.loads(output_json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, list):
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for i, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            iid = item.get("id") if isinstance(item.get("id"), str) else f"idx_{i}"
            result[iid] = item
        return result

    def process_book(self, book: BookPaths, resume: bool = True) -> Dict[str, int]:
        self._safe_print(f"\n==> Processing book: {book.book_dir}")
        raw_data = json.loads(book.input_exercises.read_text(encoding="utf-8"))
        if not isinstance(raw_data, list):
            raise ValueError(f"Input exercises must be a list: {book.input_exercises}")

        existing_map = self._build_existing_map(book.output_exercise_json) if resume else {}

        chapter_cache: Dict[str, Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = {}
        out_items: List[Dict[str, Any]] = []

        stats = {
            "total": 0,
            "resumed": 0,
            "round1_parse_ok": 0,
            "round2_parse_ok": 0,
            "missing_source_md": 0,
            "missing_chapter_json": 0,
        }

        for idx, item in enumerate(raw_data, start=1):
            stats["total"] += 1
            if not isinstance(item, dict):
                out_items.append(item)
                continue

            ex_id = item.get("id") if isinstance(item.get("id"), str) else f"idx_{idx}"
            properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            stem = properties.get("stem") if isinstance(properties.get("stem"), str) else ""

            existing_item = existing_map.get(ex_id)
            if existing_item is not None and self._is_completed_exercise(existing_item):
                out_items.append(existing_item)
                stats["resumed"] += 1
                continue

            if not stem.strip():
                new_item = dict(item)
                new_props = dict(properties)
                new_props["tests_concept"] = []
                new_props["tests_skill"] = []
                new_item["properties"] = new_props
                out_items.append(new_item)
                continue

            source_md = meta.get("source_md") if isinstance(meta.get("source_md"), str) else ""
            if not source_md:
                stats["missing_source_md"] += 1

            chapter_json_path = self._to_chapter_json_path(book.raw_output_dir, source_md) if source_md else Path("__missing__.json")
            chapter_key = str(chapter_json_path)
            if chapter_key not in chapter_cache:
                if chapter_json_path.exists():
                    chapter_cache[chapter_key] = self._extract_candidates_from_chapter_json(chapter_json_path)
                else:
                    stats["missing_chapter_json"] += 1
                    chapter_cache[chapter_key] = ([], [])
            concepts, skills = chapter_cache[chapter_key]

            # 第一轮：只给 stem
            round1_prompt = self._build_round1_prompt(stem)
            round1_raw_path = book.raw_text_dir / f"{ex_id}_round1.txt"
            round1_resp = self._call_llm(round1_prompt, round1_raw_path, max_tokens=2400)
            round1_obj = self._parse_json_loose(round1_resp)

            answer: str = ""
            difficulty: Optional[int] = None
            q_type: str = ""
            analysis: Optional[str] = None

            if isinstance(round1_obj, dict):
                stats["round1_parse_ok"] += 1
                ans_raw = round1_obj.get("answer")
                typ_raw = round1_obj.get("type")
                ana_raw = round1_obj.get("analysis")
                diff_raw = round1_obj.get("difficulty")

                if isinstance(ans_raw, str):
                    answer = ans_raw.strip()
                if isinstance(typ_raw, str):
                    q_type = typ_raw.strip()
                if isinstance(ana_raw, str) and ana_raw.strip():
                    analysis = ana_raw.strip()
                difficulty = self._normalize_difficulty(diff_raw)

            # 第二轮：只给 stem + 候选concept/skill(id+name)
            round2_prompt = self._build_round2_prompt(stem, concepts, skills)
            round2_raw_path = book.raw_text_dir / f"{ex_id}_round2.txt"
            round2_resp = self._call_llm(round2_prompt, round2_raw_path, max_tokens=1800)
            round2_obj = self._parse_json_loose(round2_resp)

            concept_ids: List[str] = []
            skill_ids: List[str] = []
            if isinstance(round2_obj, dict):
                stats["round2_parse_ok"] += 1
                allowed_c = {x["id"] for x in concepts}
                allowed_s = {x["id"] for x in skills}
                concept_ids = self._normalize_id_list(round2_obj.get("tests_concept"), allowed_c)
                skill_ids = self._normalize_id_list(round2_obj.get("tests_skill"), allowed_s)

            tests_concept = self._map_ids_to_id_name(concept_ids, concepts)
            tests_skill = self._map_ids_to_id_name(skill_ids, skills)

            new_item = dict(item)
            new_props = dict(properties)

            # 第一轮产物
            if answer:
                new_props["answer"] = answer
            elif isinstance(new_props.get("answer"), str):
                pass
            else:
                new_props["answer"] = ""

            if difficulty is not None:
                new_props["difficulty"] = difficulty
            elif "difficulty" not in new_props:
                new_props["difficulty"] = None

            # 按用户要求：type 不做集合校验，直接使用模型输出
            if q_type:
                new_props["type"] = q_type
            elif isinstance(new_props.get("type"), str):
                pass
            else:
                new_props["type"] = ""

            if analysis:
                new_props["analysis"] = analysis

            # 第二轮产物：tests_* 列表写在 properties 下
            new_props["tests_concept"] = tests_concept
            new_props["tests_skill"] = tests_skill

            new_item["properties"] = new_props
            out_items.append(new_item)

            if idx % 20 == 0:
                self._safe_print(f"  processed {idx}/{len(raw_data)} exercises")

        book.raw_output_dir.mkdir(parents=True, exist_ok=True)
        book.output_exercise_json.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
        self._safe_print(f"  wrote: {book.output_exercise_json}")
        return stats


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.startswith(("\"", "'")) and v.endswith(("\"", "'")) and len(v) >= 2:
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v


def find_books(root_output: Path) -> List[BookPaths]:
    books: List[BookPaths] = []
    for input_json in root_output.rglob("exercises_from_exercise_md_text_only.json"):
        if "/out_sections/" not in input_json.as_posix():
            continue
        out_sections_dir = input_json.parent
        if out_sections_dir.name != "out_sections":
            continue

        book_dir = out_sections_dir.parent
        raw_output_dir = book_dir / "raw_output"
        books.append(
            BookPaths(
                book_dir=book_dir,
                input_exercises=input_json,
                raw_output_dir=raw_output_dir,
                output_exercise_json=raw_output_dir / "exercise.json",
                raw_text_dir=raw_output_dir / "exercise_llm_raw",
            )
        )

    books.sort(key=lambda x: x.book_dir.as_posix())
    return books


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exercise.json for all books")
    parser.add_argument("--output-root", default="output", help="Root output directory")
    parser.add_argument("--model", default=None, help="LLM model")
    parser.add_argument("--api-key", default=None, help="OPENAI api key")
    parser.add_argument("--base-url", default=None, help="OPENAI base url")
    parser.add_argument("--env-file", default=None, help="Env file path (default: src/exercise/.openai.env)")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    parser.add_argument("--max-books", type=int, default=None, help="Process at most N books for test")
    parser.add_argument("--book-contains", default=None, help="Only process books whose path contains this text")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume; rerun all questions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = THIS_DIR.parent.parent

    env_file = Path(args.env_file) if args.env_file else (THIS_DIR / ".openai.env")
    load_env_file(env_file)

    model = args.model or os.getenv("OPENAI_MODEL") or "gpt-4o"
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required (via --api-key or src/exercise/.openai.env)")

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    books = find_books(output_root)
    if args.book_contains:
        books = [b for b in books if args.book_contains in b.book_dir.as_posix()]
    if args.max_books is not None:
        books = books[: max(args.max_books, 0)]

    if not books:
        print("No books found to process.")
        return

    print(f"Found books: {len(books)}")
    enricher = ExerciseEnricher(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=args.temperature,
    )

    total_stats = {
        "books": 0,
        "total": 0,
        "resumed": 0,
        "round1_parse_ok": 0,
        "round2_parse_ok": 0,
        "missing_source_md": 0,
        "missing_chapter_json": 0,
    }

    for i, book in enumerate(books, start=1):
        print(f"\n[{i}/{len(books)}] {book.book_dir}")
        stats = enricher.process_book(book, resume=not args.no_resume)
        total_stats["books"] += 1
        for k in ["total", "resumed", "round1_parse_ok", "round2_parse_ok", "missing_source_md", "missing_chapter_json"]:
            total_stats[k] += stats.get(k, 0)

    print("\nDone.")
    print(json.dumps(total_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
