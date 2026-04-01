#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SUBJECT_MAP = {
    "化学": "chemistry",
    "数学": "math",
    "物理": "physics",
    "生物学": "biology",
}

STAGE_MAP = {
    "小学": "primaryschool",
    "初中": "middleschool",
    "高中": "highschool",
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"invalid graph json object: {path}")
    return obj


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_merged_kg_files(output_root: Path) -> Iterable[Path]:
    return sorted(output_root.rglob("merged_kg.json"))


def longest_common_prefix(values: List[str]) -> str:
    if not values:
        return ""
    p = values[0]
    for v in values[1:]:
        i = 0
        m = min(len(p), len(v))
        while i < m and p[i] == v[i]:
            i += 1
        p = p[:i]
        if not p:
            break
    return p


def infer_book_prefix(nodes: List[Dict[str, Any]], fallback: str) -> str:
    ids = [str(n.get("id", "")).strip() for n in nodes if isinstance(n, dict)]
    ids = [x for x in ids if x]
    if not ids:
        return fallback
    # Prefer canonical knowledge-node IDs such as:
    # chemistry_highschool_rjb_bx1_cpt1 -> chemistry_highschool_rjb_bx1
    # math_9a_rjb_skl12 -> math_9a_rjb
    stem_hits: Dict[str, int] = defaultdict(int)
    for nid in ids:
        m = re.match(r"^(.*)_(?:cpt|skl|exp)\d+$", nid)
        if m:
            stem = m.group(1).strip("_")
            if stem:
                stem_hits[stem] += 1
    if stem_hits:
        # highest frequency first, then shortest (more canonical), then lexicographic
        best = sorted(stem_hits.items(), key=lambda x: (-x[1], len(x[0]), x[0]))[0][0]
        return best
    p = longest_common_prefix(ids)
    if "_" in p:
        p = p[: p.rfind("_")]
    p = p.strip("_")
    return p or fallback


def normalize_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    nodes = raw.get("nodes", [])
    edges = raw.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("graph must contain list fields: nodes/edges")
    return {"nodes": deepcopy(nodes), "edges": deepcopy(edges)}


def merge_nodes_by_name(graph: Dict[str, Any]) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = graph.get("nodes", [])
    edges: List[Dict[str, Any]] = graph.get("edges", [])

    id_to_node: Dict[str, Dict[str, Any]] = {}
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("id", "")).strip()
        if not nid:
            continue
        id_to_node[nid] = node
        label = str(node.get("label", "")).strip()
        props = node.get("properties", {})
        name = ""
        if isinstance(props, dict):
            name = str(props.get("name", "")).strip()
        groups[(label, name)].append(nid)

    remap: Dict[str, str] = {}
    kept_ids = set(id_to_node.keys())
    for ids in groups.values():
        if len(ids) <= 1:
            continue
        canonical = min(ids)
        for nid in ids:
            remap[nid] = canonical
            if nid != canonical and nid in kept_ids:
                kept_ids.remove(nid)

    new_nodes = [deepcopy(id_to_node[nid]) for nid in sorted(kept_ids)]

    new_edges: List[Dict[str, Any]] = []
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        e = deepcopy(edge)
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if src in remap:
            e["source"] = remap[src]
        if tgt in remap:
            e["target"] = remap[tgt]
        key = json.dumps(e, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(e)

    return {"nodes": new_nodes, "edges": new_edges}


def merge_many_graphs(graphs: List[Dict[str, Any]], do_name_merge: bool) -> Dict[str, Any]:
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []
    for g in graphs:
        all_nodes.extend(deepcopy(g.get("nodes", [])))
        all_edges.extend(deepcopy(g.get("edges", [])))
    merged = {"nodes": all_nodes, "edges": all_edges}
    if do_name_merge:
        return merge_nodes_by_name(merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hierarchy merged graphs (book, subject_stage, subject).")
    parser.add_argument("--output-root", type=Path, required=True, help="Root folder that contains per-book merged_kg.json files.")
    parser.add_argument("--target-root", type=Path, required=True, help="Target root for book_graph/subject_stage_graph/subject_graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    target_root = args.target_root
    book_dir = target_root / "book_graph"
    ss_dir = target_root / "subject_stage_graph"
    subject_dir = target_root / "subject_graph"
    book_dir.mkdir(parents=True, exist_ok=True)
    ss_dir.mkdir(parents=True, exist_ok=True)
    subject_dir.mkdir(parents=True, exist_ok=True)

    for p in list(book_dir.glob("*.json")) + list(ss_dir.glob("*.json")) + list(subject_dir.glob("*.json")):
        p.unlink()

    by_subject_stage: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary: Dict[str, Any] = {
        "book_graph": [],
        "subject_stage_graph": [],
        "subject_graph": [],
    }

    for kg_path in iter_merged_kg_files(output_root):
        rel = kg_path.relative_to(output_root)
        if len(rel.parts) < 5:
            continue
        subject_cn, stage_cn = rel.parts[0], rel.parts[1]
        subject = SUBJECT_MAP.get(subject_cn)
        stage = STAGE_MAP.get(stage_cn)
        if not subject or not stage:
            continue

        raw = normalize_graph(read_json(kg_path))
        fallback = f"{subject}_{stage}_{kg_path.parent.name}".replace(" ", "_")
        book_prefix = infer_book_prefix(raw.get("nodes", []), fallback=fallback)
        book_path = book_dir / f"{book_prefix}.json"
        write_json(book_path, raw)
        summary["book_graph"].append(
            {
                "source": str(kg_path),
                "book_file": str(book_path),
                "nodes": len(raw.get("nodes", [])),
                "edges": len(raw.get("edges", [])),
            }
        )

        by_subject_stage[(subject, stage)].append(raw)
        by_subject[subject].append(raw)

    for (subject, stage), graphs in sorted(by_subject_stage.items()):
        merged = merge_many_graphs(graphs, do_name_merge=True)
        out_path = ss_dir / f"{subject}_{stage}.json"
        write_json(out_path, merged)
        summary["subject_stage_graph"].append(
            {
                "file": str(out_path),
                "book_count": len(graphs),
                "nodes": len(merged.get("nodes", [])),
                "edges": len(merged.get("edges", [])),
            }
        )

    for subject, graphs in sorted(by_subject.items()):
        merged = merge_many_graphs(graphs, do_name_merge=True)
        out_path = subject_dir / f"{subject}.json"
        write_json(out_path, merged)
        summary["subject_graph"].append(
            {
                "file": str(out_path),
                "book_count": len(graphs),
                "nodes": len(merged.get("nodes", [])),
                "edges": len(merged.get("edges", [])),
            }
        )

    write_json(target_root / "merge_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
