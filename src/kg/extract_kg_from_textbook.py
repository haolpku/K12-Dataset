#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: read segmented sections and extract chapter-level KG JSON files."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import load_config
from utils.io import read_json, write_json
from utils.llm_client import create_llm_client

NODE_PREFIX_MAP = {
    "Concept": "cpt",
    "Skill": "skl",
    "Experiment": "exp",
    "Exercise": "exe",
}

NODE_ALLOWED_PROPS = {
    "Concept": {"definition", "importance", "examples", "aliases", "formula", "unit", "pages"},
    "Skill": {"description", "importance"},
    "Experiment": {"description", "importance", "instrument", "is_student", "conclusion", "process", "phenomena"},
    "Exercise": {"stem", "answer", "difficulty", "type", "analysis"},
}

EDGE_ALLOWED_PROPS = {
    "evidence",
    "original_text",
    "relations",
}

AGGREGATED_EDGE_TYPES = {"tests_concept", "tests_skill"}


@dataclass
class Section:
    chapter_num: str
    chapter_title: str
    section_num: str
    section_title: str
    file_name: str
    content: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract chapter KG files from segmented textbook sections")
    parser.add_argument("--config", default=None, help="Pipeline config path")
    parser.add_argument("--prompt", default=None, help="Prompt template path")
    parser.add_argument("--filter-prefix", action="append", default=None, help="Only process specific book_prefix values")
    parser.add_argument("--chapters", default=None, help="Comma-separated section file stems, e.g. ch1_s1,ch2_s3")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N books")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max completion tokens")
    return parser.parse_args()


def load_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def build_prompt(template: str, section: Section) -> str:
    return template.replace("{{Section_Markdown_Content}}", section.content)


def normalize_node_label(raw_node: Dict[str, Any]) -> str:
    label = raw_node.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    raw_id = str(raw_node.get("id", ""))
    if raw_id.startswith("cpt_"):
        return "Concept"
    if raw_id.startswith("skl_"):
        return "Skill"
    if raw_id.startswith("exp_"):
        return "Experiment"
    if raw_id.startswith("exe_"):
        return "Exercise"
    return "Concept"


def normalize_node_index(raw_node: Dict[str, Any], counters: Dict[str, int]) -> int:
    raw_id = str(raw_node.get("id", "")).strip()
    match = re.search(r"(\d+)$", raw_id)
    label = normalize_node_label(raw_node)
    if match:
        return int(match.group(1))
    counters[label] = counters.get(label, 0) + 1
    return counters[label]


def normalize_ref(ref: Any) -> List[str]:
    if ref is None:
        return []
    if isinstance(ref, str):
        return [ref]
    if isinstance(ref, dict):
        for key in ("id", "name", "stem", "label"):
            value = ref.get(key)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
        return []
    if isinstance(ref, list):
        out: List[str] = []
        for item in ref:
            out.extend(normalize_ref(item))
        return out
    return [str(ref)]


def clean_node_properties(label: str, raw_node: Dict[str, Any], name: str) -> Dict[str, Any]:
    props = raw_node.get("properties")
    merged: Dict[str, Any] = dict(props) if isinstance(props, dict) else {}
    for key, value in raw_node.items():
        if key in {"id", "label", "name", "properties"}:
            continue
        merged[key] = value
    allowed = NODE_ALLOWED_PROPS.get(label, set())
    cleaned = {key: value for key, value in merged.items() if key in allowed}
    if label == "Exercise" and "stem" not in cleaned and name:
        cleaned["stem"] = name
    return cleaned


def clean_edge_properties(raw_edge: Dict[str, Any]) -> Dict[str, Any]:
    props = raw_edge.get("properties")
    merged: Dict[str, Any] = dict(props) if isinstance(props, dict) else {}
    for key, value in raw_edge.items():
        if key in {"source", "target", "type", "properties", "target_name_to_ids"}:
            continue
        merged[key] = value
    return {key: value for key, value in merged.items() if key in EDGE_ALLOWED_PROPS}


def materialize_section(
    book_prefix: str,
    section_record: Dict[str, Any],
    content: str,
) -> Section:
    return Section(
        chapter_num=str(section_record.get("chapter_num", "")),
        chapter_title=str(section_record.get("chapter_title", "")),
        section_num=str(section_record.get("section_num", "")),
        section_title=str(section_record.get("section_title", "")),
        file_name=str(section_record.get("file", "")),
        content=content,
    )


def build_nodes(book_prefix: str, raw_nodes: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    processed: List[Dict[str, Any]] = []
    raw_to_new: Dict[str, str] = {}
    info_by_new: Dict[str, Dict[str, Any]] = {}
    counters: Dict[str, int] = {}

    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue
        label = normalize_node_label(raw_node)
        idx = normalize_node_index(raw_node, counters)
        prefix = NODE_PREFIX_MAP.get(label, label.lower())
        node_id = f"{book_prefix}_{prefix}{idx}"
        name = str(raw_node.get("name") or raw_node.get("stem") or raw_node.get("id") or node_id).strip()
        raw_key = str(raw_node.get("id") or name)
        properties = clean_node_properties(label, raw_node, name)
        node = {
            "id": node_id,
            "label": label,
            "name": name,
        }
        if properties:
            node["properties"] = properties
        processed.append(node)
        raw_to_new[raw_key] = node_id
        info_by_new[node_id] = node

    return processed, raw_to_new, info_by_new


def build_standard_edge(
    source_id: str,
    target_id: str,
    edge_type: str,
    source_node: Dict[str, Any],
    target_node: Dict[str, Any],
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    edge = {
        "source": source_id,
        "target": target_id,
        "type": edge_type,
    }
    if source_node["label"] == "Exercise":
        edge["source_stem"] = source_node.get("properties", {}).get("stem", source_node["name"])
    else:
        edge["source_name"] = source_node["name"]
    edge["target_name"] = target_node["name"]
    if properties:
        edge["properties"] = properties
    return edge


def build_edges(
    raw_edges: Sequence[Dict[str, Any]],
    raw_to_new: Dict[str, str],
    info_by_new: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    standard_edges: List[Dict[str, Any]] = []
    aggregated: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            continue
        edge_type = str(raw_edge.get("type", "")).strip()
        if not edge_type:
            continue

        source_refs = normalize_ref(raw_edge.get("source"))
        if not source_refs:
            continue
        source_id = raw_to_new.get(source_refs[0])
        if not source_id or source_id not in info_by_new:
            continue
        source_node = info_by_new[source_id]
        properties = clean_edge_properties(raw_edge)

        if edge_type in AGGREGATED_EDGE_TYPES:
            target_pairs = raw_edge.get("target_name_to_ids")
            target_refs: List[str] = []
            if isinstance(target_pairs, list):
                for item in target_pairs:
                    if isinstance(item, dict) and isinstance(item.get("target"), str):
                        target_refs.append(item["target"])
            else:
                target_refs.extend(normalize_ref(raw_edge.get("target")))

            key = (source_id, edge_type)
            bucket = aggregated.setdefault(
                key,
                {
                    "source": source_id,
                    "type": edge_type,
                    "source_stem": source_node.get("properties", {}).get("stem", source_node["name"]),
                    "target_name_to_ids": [],
                },
            )
            if properties and "properties" not in bucket:
                bucket["properties"] = properties
            seen_targets = {(item["target"], item["target_name"]) for item in bucket["target_name_to_ids"]}
            for ref in target_refs:
                target_id = raw_to_new.get(ref)
                if not target_id or target_id not in info_by_new:
                    continue
                target_node = info_by_new[target_id]
                pair = (target_id, target_node["name"])
                if pair in seen_targets:
                    continue
                seen_targets.add(pair)
                bucket["target_name_to_ids"].append({"target": target_id, "target_name": target_node["name"]})
            continue

        target_refs = normalize_ref(raw_edge.get("target"))
        for ref in target_refs:
            target_id = raw_to_new.get(ref)
            if not target_id or target_id not in info_by_new:
                continue
            target_node = info_by_new[target_id]
            standard_edges.append(
                build_standard_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    source_node=source_node,
                    target_node=target_node,
                    properties=properties,
                )
            )

    standard_edges.extend(aggregated.values())
    return standard_edges


def selected_sections(sections: Sequence[Dict[str, Any]], chapters_arg: Optional[str]) -> Iterable[Dict[str, Any]]:
    if not chapters_arg:
        yield from sections
        return
    wanted = {item.strip() for item in chapters_arg.split(",") if item.strip()}
    for item in sections:
        stem = Path(str(item.get("file", ""))).stem
        if stem in wanted:
            yield item


def process_book(
    book: Dict[str, Any],
    prompt_template: str,
    chapters_arg: Optional[str],
    max_tokens: int,
    config_path: Optional[str],
) -> Dict[str, Any]:
    config = load_config(config_path)
    book_prefix = str(book["book_prefix"])
    sections_index = read_json(config.sections_index_for(book))
    sections = sections_index.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError(f"invalid sections_index for {book_prefix}")

    llm_cfg = config.llm
    client = create_llm_client(
        provider=str(llm_cfg.get("provider", "openai")),
        model=str(llm_cfg.get("model", "gpt-4.1-mini")),
        api_key=str(llm_cfg.get("api_key", "")),
        base_url=str(llm_cfg.get("base_url", "")) or None,
    )

    raw_dir = config.build_graph_book_dir(book_prefix)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir = config.chapter_kg_dir / book_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for section_record in selected_sections(sections, chapters_arg):
        file_name = str(section_record.get("file", "")).strip()
        if not file_name:
            continue
        section_path = config.sections_dir_for(book_prefix) / file_name
        content = section_path.read_text(encoding="utf-8")
        section = materialize_section(book_prefix, section_record, content)
        prompt = build_prompt(prompt_template, section)
        response = client.generate(
            prompt,
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=max_tokens,
        )

        parsed = client.parse_response(response)
        nodes, raw_to_new, info_by_new = build_nodes(book_prefix, parsed.get("nodes", []))
        edges = build_edges(parsed.get("edges", []), raw_to_new, info_by_new)
        payload = {
            "section_info": {
                "chapter": section.chapter_num,
                "chapter_title": section.chapter_title,
                "section": section.section_num,
                "section_title": section.section_title,
            },
            "nodes": nodes,
            "edges": edges,
        }
        stem = Path(file_name).stem
        write_json(output_dir / f"{stem}.json", payload)
        (raw_dir / f"{stem}_llm_raw.txt").write_text(response, encoding="utf-8")
        written += 1

    return {"book_prefix": book_prefix, "written": written}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompt_path = Path(args.prompt) if args.prompt else THIS_DIR / "prompt.txt"
    prompt_template = load_prompt(prompt_path)
    filter_prefixes = {item.strip() for item in (args.filter_prefix or []) if item.strip()}

    count = 0
    for book in config.load_books():
        if filter_prefixes and book["book_prefix"] not in filter_prefixes:
            continue
        result = process_book(book, prompt_template, args.chapters, args.max_tokens, args.config)
        print(f"[OK] {result['book_prefix']} -> {result['written']} sections")
        count += 1
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
