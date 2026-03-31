#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按“每本书”合并 raw_output 下分章节 KG 为 merged_kg.json。

实现要点：
1. 全局重编号，避免章节内局部编号冲突。
2. 仅对 Concept / Skill 做跨章节去重（轻量归一化匹配）。
3. 边按 (source, target, type) 去重并合并 properties。
4. 记录 (source, target) 下 type 冲突，用于人工复核。
5. tests_* 边补充 source_stem（来源 Exercise 节点 stem）和 target_name。
6. 统一关系名 prerequisites for -> prerequisites_for。
7. 发现已有 merged_kg.json 时先备份，再写新文件，并输出报告。
8. 默认跳过数学小学与数学小学 copy。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REL_TESTS = {"tests_concept", "tests_skill"}
REL_NAME_NORMALIZE = {"prerequisites for": "prerequisites_for"}
NODE_PREFIX_MAP = {
    "Concept": "cpt",
    "Skill": "skl",
    "Experiment": "exp",
    "Exercise": "t",
}
STRIP_NODE_PROPS = {
    "id",
    "label",
    "source_chapter",
    "source_chapter_title",
    "source_section",
    "source_section_title",
    "page_nums",
    "pages",
}
STRIP_EDGE_PROPS = {
    "source_chapter",
    "source_chapter_title",
    "source_section",
    "source_section_title",
    "page_nums",
    "pages",
}


def normalize_text_light(text: str) -> str:
    """轻量归一化：NFKC + 去首尾空白 + 压缩空白 + 小写（仅影响拉丁字母）。"""
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def remove_pages_from_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in (props or {}).items():
        if k in STRIP_NODE_PROPS:
            continue
        out[k] = v
    return out


def infer_book_prefix(book_dir: Path) -> Optional[str]:
    """从已有 merged_kg 或 raw_output/ch*.json 节点 ID 推断 book_prefix。"""
    candidates: List[Path] = []
    merged = book_dir / "merged_kg.json"
    if merged.exists():
        candidates.append(merged)
    candidates.extend(sorted((book_dir / "raw_output").glob("*.json")))

    pat = re.compile(r"^(.+?)_(?:cpt|skl|exp|ch\d+)")
    for fp in candidates:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        for node in nodes:
            node_id = str((node or {}).get("id", ""))
            m = pat.match(node_id)
            if m:
                return m.group(1)
    return None


def chapter_section_from_filename(name: str) -> Tuple[Optional[str], Optional[str]]:
    # 兼容两种文件名：
    # 1) ch12.json / ch12_s3.json
    # 2) u5_ch1.json / u5_ch1_s2.json（初中生物常见）
    m = re.match(r"^ch(\d+)(?:_s(.+?))?\.json$", name)
    if m:
        return m.group(1), m.group(2)

    m = re.match(r"^u(\d+)_ch(\d+)(?:_s(.+?))?\.json$", name)
    if m:
        unit = m.group(1)
        chapter = m.group(2)
        section = m.group(3)
        # 以 unit+chapter 作为章节键，避免不同单元 ch1/ch2 冲突。
        return f"u{unit}_ch{chapter}", section

    return None, None

def normalize_relation_type(edge_type: str) -> str:
    return REL_NAME_NORMALIZE.get(edge_type, edge_type)


def generate_new_node_id(
    label: str,
    book_prefix: str,
    chapter: Optional[str],
    section: Optional[str],
    counter: Dict[str, int],
) -> str:
    counter[label] += 1
    if label == "Exercise":
        ch = chapter or "0"
        sec = f"_s{section}" if section else ""
        return f"{book_prefix}_ch{ch}{sec}_t{counter[label]}"
    suffix = NODE_PREFIX_MAP.get(label, label.lower())
    return f"{book_prefix}_{suffix}{counter[label]}"


def merge_node_properties(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """节点属性合并规则：
    - pages 已外层清理。
    - examples 列表并集。
    - definition/description 取更长文本。
    - 其余字段保留 base，base 缺失时补 other。
    """
    out = dict(base)
    for k, v in other.items():
        if k not in out or out[k] in (None, "", [], {}):
            out[k] = v
            continue
        if k == "examples" and isinstance(out[k], list) and isinstance(v, list):
            seen = set()
            merged = []
            for item in out[k] + v:
                token = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if token not in seen:
                    seen.add(token)
                    merged.append(item)
            out[k] = merged
            continue
        if k in {"definition", "description"} and isinstance(out[k], str) and isinstance(v, str):
            if len(v) > len(out[k]):
                out[k] = v
            continue
    return out


def merge_edge_properties(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """边属性合并规则：并集优先，冲突保留 base。"""
    out = dict(base)
    for k, v in other.items():
        if k not in out:
            out[k] = v
            continue
        cur = out[k]
        if cur == v:
            continue
        if isinstance(cur, list) and isinstance(v, list):
            seen = set()
            merged = []
            for item in cur + v:
                token = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if token not in seen:
                    seen.add(token)
                    merged.append(item)
            out[k] = merged
            continue
        # 字段冲突默认保留 base
    return out


def collect_book_dirs(base_dir: Path, subjects: List[str], skip_math_primary: bool) -> List[Path]:
    books: List[Path] = []
    for subject in subjects:
        subject_dir = base_dir / subject
        if not subject_dir.exists():
            continue
        for raw in subject_dir.rglob("raw_output"):
            book_dir = raw.parent
            p = str(book_dir)
            if skip_math_primary and subject == "数学":
                if "/小学/" in p or "/小学 copy/" in p:
                    continue
            books.append(book_dir)
    return sorted(set(books))


def parse_book_context(book_dir: Path) -> Dict[str, str]:
    parts = book_dir.parts
    # .../output/学科/学段/出版社/书名
    try:
        output_idx = parts.index("output")
    except ValueError:
        return {"subject": "", "grade": book_dir.name, "publisher": ""}
    subject = parts[output_idx + 1] if len(parts) > output_idx + 1 else ""
    publisher = parts[output_idx + 3] if len(parts) > output_idx + 3 else ""
    grade = parts[output_idx + 4] if len(parts) > output_idx + 4 else book_dir.name
    return {"subject": subject, "grade": grade, "publisher": publisher}


def merge_single_book(book_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    raw_dir = book_dir / "raw_output"
    chapter_files = []
    for fp in sorted(raw_dir.glob("*.json")):
        if chapter_section_from_filename(fp.name) == (None, None):
            continue
        chapter_files.append(fp)

    if not chapter_files:
        return {
            "book_dir": str(book_dir),
            "status": "skipped",
            "reason": "no_chapter_json",
        }

    book_prefix = infer_book_prefix(book_dir)
    if not book_prefix:
        return {
            "book_dir": str(book_dir),
            "status": "failed",
            "reason": "cannot_infer_book_prefix",
        }

    context = parse_book_context(book_dir)

    node_counter: Dict[str, int] = defaultdict(int)
    id_mapping: Dict[Tuple[str, str, str], str] = {}
    all_nodes: List[Dict[str, Any]] = []
    raw_edges: List[Tuple[Dict[str, Any], str, str]] = []
    raw_nodes_total = 0
    raw_edges_total = 0
    chapters_info: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"title": "", "sections": {}})
    node_sources: Dict[str, Set[Tuple[str, Optional[str]]]] = defaultdict(set)

    for fp in chapter_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        section_info = data.get("section_info", {}) if isinstance(data, dict) else {}
        ch_num = str(section_info.get("chapter") or chapter_section_from_filename(fp.name)[0] or "")
        sec_num = section_info.get("section")
        sec_num = str(sec_num) if sec_num not in (None, "") else None

        if ch_num:
            chapters_info[ch_num]["title"] = str(section_info.get("chapter_title") or chapters_info[ch_num]["title"])
            if sec_num:
                chapters_info[ch_num]["sections"][sec_num] = str(section_info.get("section_title") or "")

        for node in data.get("nodes", []):
            raw_nodes_total += 1
            old_id = str(node.get("id", ""))
            label = str(node.get("label", "Concept"))
            if not old_id:
                continue

            key = (old_id, ch_num or "", sec_num or "")
            if key not in id_mapping:
                id_mapping[key] = generate_new_node_id(label, book_prefix, ch_num, sec_num, node_counter)
            new_id = id_mapping[key]

            props = remove_pages_from_properties(dict(node.get("properties", {}) or {}))
            all_nodes.append({"id": new_id, "label": label, "properties": props})
            if ch_num:
                node_sources[new_id].add((ch_num, sec_num))

        for edge in data.get("edges", []):
            raw_edges_total += 1
            raw_edges.append((dict(edge), ch_num or "", sec_num or ""))

    # 结构节点
    book_id = book_prefix
    struct_nodes: List[Dict[str, Any]] = [
        {
            "id": book_id,
            "label": "Book",
            "properties": {
                "subject": context.get("subject", ""),
                "grade": context.get("grade", ""),
                "publisher": context.get("publisher", ""),
            },
        }
    ]

    struct_edges: List[Dict[str, Any]] = []
    for ch in sorted(chapters_info.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        ch_id = f"{book_prefix}_ch{ch}"
        struct_nodes.append({"id": ch_id, "label": "Chapter", "properties": {"title": chapters_info[ch]["title"]}})
        struct_edges.append({"source": ch_id, "target": book_id, "type": "is_part_of"})

        for sec in sorted(chapters_info[ch]["sections"].keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            sec_id = f"{book_prefix}_ch{ch}_s{sec}"
            struct_nodes.append(
                {
                    "id": sec_id,
                    "label": "Section",
                    "properties": {"title": chapters_info[ch]["sections"][sec]},
                }
            )
            struct_edges.append({"source": sec_id, "target": ch_id, "type": "is_part_of"})

    # Concept/Skill 去重
    dedup_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    non_dedup_nodes: List[Dict[str, Any]] = []
    for node in all_nodes:
        label = node.get("label")
        name = str(node.get("properties", {}).get("name", ""))
        if label in {"Concept", "Skill"} and name.strip():
            dedup_groups[(label, normalize_text_light(name))].append(node)
        else:
            non_dedup_nodes.append(node)

    node_merge_map: Dict[str, str] = {}
    merged_nodes: List[Dict[str, Any]] = []
    merged_sources: Dict[str, Set[Tuple[str, Optional[str]]]] = defaultdict(set)

    for key, group in dedup_groups.items():
        main = dict(group[0])
        main_id = str(main["id"])
        for g in group:
            gid = str(g["id"])
            if gid != main_id:
                node_merge_map[gid] = main_id
                main["properties"] = merge_node_properties(main.get("properties", {}), g.get("properties", {}))
            for src in node_sources.get(gid, set()):
                merged_sources[main_id].add(src)
        merged_nodes.append(main)

    dedup_nodes_before = sum(len(group) for group in dedup_groups.values())
    dedup_nodes_after = len(dedup_groups)
    merged_nodes_count = max(0, dedup_nodes_before - dedup_nodes_after)

    for node in non_dedup_nodes:
        nid = str(node["id"])
        merged_nodes.append(node)
        for src in node_sources.get(nid, set()):
            merged_sources[nid].add(src)

    # 结构节点加入
    merged_nodes.extend(struct_nodes)

    # 建立最终 node 映射
    node_by_id = {}
    for n in merged_nodes:
        node_by_id[str(n["id"])] = n

    def resolved_node_id(old_id: str, ch: str, sec: str) -> Optional[str]:
        mapped = id_mapping.get((old_id, ch, sec))
        if not mapped:
            return None
        return node_merge_map.get(mapped, mapped)

    # appears_in 边
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    type_conflicts: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    edge_inserted_count = 0
    edge_merged_duplicate_count = 0

    def upsert_edge(edge: Dict[str, Any]) -> None:
        nonlocal edge_inserted_count, edge_merged_duplicate_count
        rel_type = normalize_relation_type(str(edge.get("type", "")))
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not source or not target or not rel_type:
            return

        edge["type"] = rel_type
        key = (source, target, rel_type)
        st_key = (source, target)

        existing_types = type_conflicts[st_key]
        existing_types.add(rel_type)

        if key not in edge_map:
            edge_map[key] = edge
            edge_inserted_count += 1
            return

        edge_merged_duplicate_count += 1
        old_props = edge_map[key].get("properties", {}) or {}
        new_props = edge.get("properties", {}) or {}
        merged_props = merge_edge_properties(old_props, new_props)
        if merged_props:
            edge_map[key]["properties"] = merged_props

    for node_id, srcs in merged_sources.items():
        node = node_by_id.get(node_id, {})
        if node.get("label") in {"Book", "Chapter", "Section"}:
            continue
        for ch, sec in sorted(srcs):
            target = f"{book_prefix}_ch{ch}_s{sec}" if sec else f"{book_prefix}_ch{ch}"
            upsert_edge({"source": node_id, "target": target, "type": "appears_in"})

    # 结构边
    for e in struct_edges:
        upsert_edge(dict(e))

    # 原始边
    for edge, ch, sec in raw_edges:
        src_old = str(edge.get("source", ""))
        tgt_old = str(edge.get("target", ""))
        src = resolved_node_id(src_old, ch, sec)
        tgt = resolved_node_id(tgt_old, ch, sec)
        if not src or not tgt:
            continue

        rel_type = normalize_relation_type(str(edge.get("type", "")))
        props = {k: v for k, v in dict(edge.get("properties", {}) or {}).items() if k not in STRIP_EDGE_PROPS}

        src_node = node_by_id.get(src, {})
        tgt_node = node_by_id.get(tgt, {})
        src_name = str(src_node.get("properties", {}).get("name", ""))
        tgt_name = str(tgt_node.get("properties", {}).get("name", ""))

        # 对齐小学数学字段顺序：
        # source -> (source_name/source_stem) -> target -> target_name -> type -> properties
        out_edge: Dict[str, Any] = {"source": src}
        if rel_type in REL_TESTS:
            source_stem = str(src_node.get("properties", {}).get("stem", ""))
            if source_stem:
                out_edge["source_stem"] = source_stem
        else:
            if src_name:
                out_edge["source_name"] = src_name

        out_edge["target"] = tgt
        if tgt_name:
            out_edge["target_name"] = tgt_name

        out_edge["type"] = rel_type
        if props:
            out_edge["properties"] = props
        upsert_edge(out_edge)

    # 边类型冲突列表
    type_conflict_items = []
    for (source, target), types in sorted(type_conflicts.items()):
        if len(types) > 1:
            type_conflict_items.append(
                {
                    "source": source,
                    "target": target,
                    "types": sorted(types),
                }
            )

    result = {"nodes": list(node_by_id.values()), "edges": list(edge_map.values())}

    out_file = book_dir / "merged_kg.json"
    backup_file = None
    preexisting = out_file.exists()
    if preexisting:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = book_dir / f"merged_kg.json.backup_{stamp}"

    report = {
        "book_dir": str(book_dir),
        "book_prefix": book_prefix,
        "status": "ok",
        "preexisting_merged_kg": preexisting,
        "backup_file": str(backup_file) if backup_file else None,
        "chapter_files": len(chapter_files),
        "input_nodes_total": raw_nodes_total,
        "input_edges_total": raw_edges_total,
        "merged_nodes_count": merged_nodes_count,
        "merged_edges_count": edge_merged_duplicate_count,
        "output_nodes_total": len(result["nodes"]),
        "output_edges_total": len(result["edges"]),
        "struct_nodes_added": len(struct_nodes),
        "edge_unique_triples": edge_inserted_count,
        "nodes": len(result["nodes"]),
        "edges": len(result["edges"]),
        "type_conflicts": type_conflict_items,
        "type_conflict_count": len(type_conflict_items),
    }

    if not dry_run:
        if preexisting and backup_file is not None:
            shutil.copy2(out_file, backup_file)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        report_file = book_dir / "merge_report.json"
        with report_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def run_batch(base_dir: Path, subjects: List[str], skip_math_primary: bool, dry_run: bool) -> Dict[str, Any]:
    books = collect_book_dirs(base_dir, subjects, skip_math_primary)
    reports = []
    for book in books:
        reports.append(merge_single_book(book, dry_run=dry_run))

    summary = {
        "total_books": len(books),
        "ok": sum(1 for r in reports if r.get("status") == "ok"),
        "failed": sum(1 for r in reports if r.get("status") == "failed"),
        "skipped": sum(1 for r in reports if r.get("status") == "skipped"),
        "preexisting_merged_kg_books": [r.get("book_dir") for r in reports if r.get("preexisting_merged_kg")],
        "reports": reports,
    }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge per-chapter KG JSONs into per-book merged_kg.json")
    p.add_argument("--base-dir", default="output", help="输出根目录")
    p.add_argument("--subjects", default="数学,物理,化学", help="学科列表，逗号分隔")
    p.add_argument("--book-dir", help="只处理单本书目录")
    p.add_argument("--skip-math-primary", action="store_true", default=True, help="跳过数学小学/小学 copy")
    p.add_argument("--no-skip-math-primary", action="store_false", dest="skip_math_primary")
    p.add_argument("--dry-run", action="store_true", help="仅预演，不写文件")
    p.add_argument("--summary-out", help="批量模式摘要输出路径")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    subjects = [x.strip() for x in str(args.subjects).split(",") if x.strip()]

    if args.book_dir:
        report = merge_single_book(Path(args.book_dir), dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    summary = run_batch(
        base_dir=base_dir,
        subjects=subjects,
        skip_math_primary=bool(args.skip_math_primary),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
