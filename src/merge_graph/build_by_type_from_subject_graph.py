#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Set, Tuple


NODE_LABEL_TO_FILE = {
    "Concept": "concepts.json",
    "Skill": "skills.json",
    "Experiment": "experiments.json",
    "Exercise": "exercises.json",
    "Book": "books.json",
    "Chapter": "chapters.json",
    "Section": "sections.json",
}

EDGE_TYPE_TO_FILE = {
    "is_a": "is_a.json",
    "prerequisites_for": "prerequisites_for.json",
    "relates_to": "relates_to.json",
    "tests_concept": "tests_concept.json",
    "tests_skill": "tests_skill.json",
    "verifies": "verifies.json",
    "appears_in": "appears_in.json",
    "is_part_of": "is_part_of.json",
}

TEST_RELATIONS = ("tests_concept", "tests_skill")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_nodes(nodes: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    by_file: Dict[str, List[Dict[str, Any]]] = {v: [] for v in NODE_LABEL_TO_FILE.values()}
    other: List[Dict[str, Any]] = []
    for n in nodes:
        f = NODE_LABEL_TO_FILE.get(n.get("label"))
        if f:
            by_file[f].append(n)
        else:
            other.append(n)
    return by_file, other


def split_edges(edges: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    by_file: Dict[str, List[Dict[str, Any]]] = {v: [] for v in EDGE_TYPE_TO_FILE.values()}
    other: List[Dict[str, Any]] = []
    for e in edges:
        f = EDGE_TYPE_TO_FILE.get(e.get("type"))
        if f:
            by_file[f].append(e)
        else:
            other.append(e)
    return by_file, other


def build_node_index(nodes: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    id_to_label: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    id_to_stem: Dict[str, str] = {}
    for n in nodes:
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        label = str(n.get("label", "")).strip()
        props = n.get("properties", {}) if isinstance(n.get("properties", {}), dict) else {}
        name = str(props.get("name", "")).strip()
        stem = str(props.get("stem", "")).strip()
        id_to_label[nid] = label
        id_to_name[nid] = name
        id_to_stem[nid] = stem
    return id_to_label, id_to_name, id_to_stem


def merge_tests_edges(
    relation_type: str,
    edges: Iterable[Dict[str, Any]],
    id_to_label: Dict[str, str],
    id_to_name: Dict[str, str],
    id_to_stem: Dict[str, str],
) -> List[Dict[str, Any]]:
    grouped: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in edges:
        if str(e.get("type", "")).strip() != relation_type:
            continue
        src = str(e.get("source", "")).strip()
        tgt = str(e.get("target", "")).strip()
        if not src or not tgt:
            continue
        if id_to_label.get(src) != "Exercise":
            continue
        tgt_name = str(e.get("target_name", "")).strip() or id_to_name.get(tgt, "")
        if not tgt_name:
            continue
        grouped[src][tgt_name].add(tgt)

    merged_records: List[Dict[str, Any]] = []
    for src in sorted(grouped.keys()):
        name_to_ids = grouped[src]
        target_names = sorted(name_to_ids.keys())
        merged_records.append(
            {
                "source": src,
                "source_name": id_to_name.get(src, ""),
                "source_stem": id_to_stem.get(src, ""),
                "type": relation_type,
                "target_names": target_names,
                "target_name_to_ids": [
                    {"target_name": n, "target_ids": sorted(name_to_ids[n])} for n in target_names
                ],
            }
        )
    return merged_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split subject_graph into merged_data-like by-type files.")
    parser.add_argument("--subject-graph-dir", type=Path, required=True, help="Directory containing subject_graph/*.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output base directory, e.g. merged_graph/by_type")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"subject_count": 0, "subjects": [], "output_mode": "single_merged_version"}

    subject_files = sorted([p for p in args.subject_graph_dir.glob("*.json") if p.is_file()])
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []
    for sf in subject_files:
        subject = sf.stem
        data = read_json(sf)
        nodes = [x for x in data.get("nodes", []) if isinstance(x, dict)]
        edges = [x for x in data.get("edges", []) if isinstance(x, dict)]
        summary["subjects"].append(
            {
                "subject": subject,
                "source_file": str(sf),
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    nodes_all = sorted(all_nodes, key=lambda x: str(x.get("id", "")))
    edges_all = sorted(all_edges, key=lambda x: (str(x.get("type", "")), str(x.get("source", "")), str(x.get("target", ""))))
    node_file_map, other_nodes = split_nodes(nodes_all)
    edge_file_map, other_edges = split_edges(edges_all)

    merged_data_dir = args.output_dir
    if merged_data_dir.exists():
        for old in merged_data_dir.glob("*.json"):
            old.unlink()
    merged_data_dir.mkdir(parents=True, exist_ok=True)

    write_json(merged_data_dir / "nodes_all.json", nodes_all)
    write_json(merged_data_dir / "edges_all.json", edges_all)
    for fn, payload in node_file_map.items():
        write_json(merged_data_dir / fn, payload)
    for fn, payload in edge_file_map.items():
        write_json(merged_data_dir / fn, payload)

    id_to_label, id_to_name, id_to_stem = build_node_index(nodes_all)
    tests_concept_merged = merge_tests_edges("tests_concept", edge_file_map["tests_concept.json"], id_to_label, id_to_name, id_to_stem)
    tests_skill_merged = merge_tests_edges("tests_skill", edge_file_map["tests_skill.json"], id_to_label, id_to_name, id_to_stem)
    write_json(merged_data_dir / "tests_concept_merged.json", tests_concept_merged)
    write_json(merged_data_dir / "tests_skill_merged.json", tests_skill_merged)

    stats = {
        "subject": "ALL",
        "stage": None,
        "version": "subject_graph_merged",
        "source_merged_kg_files": None,
        "source_files": [str(x) for x in subject_files],
        "node_count": len(nodes_all),
        "edge_count": len(edges_all),
        "node_counts_by_label": {
            "Concept": len(node_file_map["concepts.json"]),
            "Skill": len(node_file_map["skills.json"]),
            "Experiment": len(node_file_map["experiments.json"]),
            "Exercise": len(node_file_map["exercises.json"]),
            "Book": len(node_file_map["books.json"]),
            "Chapter": len(node_file_map["chapters.json"]),
            "Section": len(node_file_map["sections.json"]),
            "Other": len(other_nodes),
        },
        "edge_counts_by_type": {
            "is_a": len(edge_file_map["is_a.json"]),
            "prerequisites_for": len(edge_file_map["prerequisites_for.json"]),
            "relates_to": len(edge_file_map["relates_to.json"]),
            "tests_concept": len(edge_file_map["tests_concept.json"]),
            "tests_skill": len(edge_file_map["tests_skill.json"]),
            "verifies": len(edge_file_map["verifies.json"]),
            "appears_in": len(edge_file_map["appears_in.json"]),
            "is_part_of": len(edge_file_map["is_part_of.json"]),
            "Other": len(other_edges),
        },
        "name_collision_count": 0,
        "duplicate_node_id_conflict_count": 0,
        "collision_report": None,
        "merged_data_dir": str(merged_data_dir),
    }
    write_json(merged_data_dir / "stats.json", stats)

    summary["subject_count"] = len(summary["subjects"])
    summary["merged_data_dir"] = str(merged_data_dir)
    summary["node_count"] = len(nodes_all)
    summary["edge_count"] = len(edges_all)
    summary["tests_concept_merged_count"] = len(tests_concept_merged)
    summary["tests_skill_merged_count"] = len(tests_skill_merged)
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
