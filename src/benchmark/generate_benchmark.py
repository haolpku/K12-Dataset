#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


TASKS = ("task1", "task2", "task3", "task4", "task5")
ALLOWED_TASK2_LABELS = {"Concept", "Skill", "Experiment"}


@dataclass
class NodeInfo:
    node_id: str
    label: str
    name: str
    stem: str


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def clean_name_list(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for x in values:
        if not isinstance(x, str):
            continue
        y = x.strip()
        if not y or y in seen:
            continue
        seen.add(y)
        out.append(y)
    return out


def parse_pairs(pair_args: List[str]) -> Optional[Set[Tuple[str, str]]]:
    if not pair_args:
        return None
    out: Set[Tuple[str, str]] = set()
    for raw in pair_args:
        if ":" not in raw:
            raise ValueError(f"invalid --pair format: {raw}; expected <subject>:<stage>")
        subject, stage = raw.split(":", 1)
        subject = subject.strip()
        stage = stage.strip()
        if not subject or not stage:
            raise ValueError(f"invalid --pair format: {raw}; empty subject/stage")
        out.add((subject, stage))
    return out


def discover_merged_data_dirs(base_dir: Path, selected_pairs: Optional[Set[Tuple[str, str]]]) -> List[Tuple[str, str, str, Path]]:
    found: List[Tuple[str, str, str, Path]] = []
    # Compatibility mode: base_dir itself is a merged_data-like directory
    # (contains nodes_all.json + edges_all.json), e.g. merged_graph/by_type.
    direct_nodes = base_dir / "nodes_all.json"
    direct_edges = base_dir / "edges_all.json"
    if direct_nodes.exists() and direct_edges.exists():
        stats_path = base_dir / "stats.json"
        subject, stage, version = "ALL", "ALL", "by_type"
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                if isinstance(stats, dict):
                    subject = str(stats.get("subject", subject))
                    stage = str(stats.get("stage", stage))
                    version = str(stats.get("version", version))
            except Exception:
                pass
        pair = (subject, stage)
        if (selected_pairs is None) or (pair in selected_pairs):
            found.append((subject, stage, version, base_dir))
        return found

    for p in sorted(base_dir.rglob("merged_data")):
        if not p.is_dir():
            continue
        try:
            rel = p.relative_to(base_dir)
        except ValueError:
            continue
        if len(rel.parts) != 4 or rel.parts[3] != "merged_data":
            continue
        subject, stage, version = rel.parts[0], rel.parts[1], rel.parts[2]
        if selected_pairs and (subject, stage) not in selected_pairs:
            continue
        found.append((subject, stage, version, p))
    return found


def build_node_map(nodes: List[Dict[str, Any]]) -> Dict[str, NodeInfo]:
    out: Dict[str, NodeInfo] = {}
    for n in nodes:
        node_id = n.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue

        label = n.get("label")
        label = label if isinstance(label, str) else ""
        props = n.get("properties") if isinstance(n.get("properties"), dict) else {}

        name = props.get("name")
        if not isinstance(name, str) or not name.strip():
            title = props.get("title")
            name = title if isinstance(title, str) else ""

        stem = props.get("stem")
        out[node_id] = NodeInfo(
            node_id=node_id,
            label=label,
            name=name if isinstance(name, str) else "",
            stem=stem if isinstance(stem, str) else "",
        )
    return out


def pick_question_template(prefix: str, key: str, templates: List[str], x: str) -> str:
    h = hashlib.sha256(f"{prefix}|{key}".encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(templates)
    return templates[idx].replace("【X】", x)


def display_exercise(node: NodeInfo) -> str:
    text = node.stem.strip() or node.name.strip()
    return text if text else node.node_id


def collect_name_pool_by_label(nodes: Dict[str, NodeInfo], label: str) -> List[str]:
    names = {n.name.strip() for n in nodes.values() if n.label == label and n.name.strip()}
    return sorted(names)


def ids_to_names(nodes: Dict[str, NodeInfo], ids: Iterable[str], label_filter: Optional[Set[str]] = None) -> List[str]:
    names: List[str] = []
    for x in ids:
        n = nodes.get(x)
        if not n:
            continue
        if label_filter and n.label not in label_filter:
            continue
        if n.label == "Exercise":
            names.append(display_exercise(n))
        else:
            names.append(n.name.strip())
    return clean_name_list(names)


def make_candidates_from_ids(nodes: Dict[str, NodeInfo], candidate_ids: Iterable[str], answer_names: List[str], sample_key: str) -> List[str]:
    answer_set = set(clean_name_list(answer_names))
    cand_names = ids_to_names(nodes, candidate_ids)
    cand_names = [x for x in cand_names if x not in answer_set]
    if not cand_names:
        return []
    cand_names = sorted(
        set(cand_names),
        key=lambda x: hashlib.sha256(f"{sample_key}|{x}".encode("utf-8")).hexdigest(),
    )
    return cand_names


def build_edge_indexes(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx: Dict[str, Any] = {}

    rel_adj: DefaultDict[str, Set[str]] = defaultdict(set)
    isa_adj: DefaultDict[str, Set[str]] = defaultdict(set)
    prereq_in: DefaultDict[str, Set[str]] = defaultdict(set)
    prereq_out: DefaultDict[str, Set[str]] = defaultdict(set)

    parent_to_children_is_a: DefaultDict[str, Set[str]] = defaultdict(set)
    child_to_parents_is_a: DefaultDict[str, Set[str]] = defaultdict(set)
    parent_to_children_prereq: DefaultDict[str, Set[str]] = defaultdict(set)

    tests_concept_by_ex: DefaultDict[str, Set[str]] = defaultdict(set)
    tests_skill_by_ex: DefaultDict[str, Set[str]] = defaultdict(set)
    ex_by_concept: DefaultDict[str, Set[str]] = defaultdict(set)
    ex_by_skill: DefaultDict[str, Set[str]] = defaultdict(set)

    verifies_exp_to_concept: DefaultDict[str, Set[str]] = defaultdict(set)
    verifies_concept_to_exp: DefaultDict[str, Set[str]] = defaultdict(set)

    appears_src_to_sec: DefaultDict[str, Set[str]] = defaultdict(set)
    part_of_parent: DefaultDict[str, Set[str]] = defaultdict(set)

    for e in edges:
        s = e.get("source")
        t = e.get("target")
        r = e.get("type")
        if not isinstance(s, str) or not s or not isinstance(t, str) or not t or not isinstance(r, str):
            continue

        if r == "relates_to":
            rel_adj[s].add(t)
            rel_adj[t].add(s)
        elif r == "is_a":
            isa_adj[s].add(t)
            isa_adj[t].add(s)
            parent_to_children_is_a[t].add(s)
            child_to_parents_is_a[s].add(t)
        elif r == "prerequisites_for":
            prereq_out[s].add(t)
            prereq_in[t].add(s)
            parent_to_children_prereq[t].add(s)
        elif r == "tests_concept":
            tests_concept_by_ex[s].add(t)
            ex_by_concept[t].add(s)
        elif r == "tests_skill":
            tests_skill_by_ex[s].add(t)
            ex_by_skill[t].add(s)
        elif r == "verifies":
            verifies_exp_to_concept[s].add(t)
            verifies_concept_to_exp[t].add(s)
        elif r == "appears_in":
            appears_src_to_sec[s].add(t)
        elif r == "is_part_of":
            part_of_parent[s].add(t)

    idx["rel_adj"] = rel_adj
    idx["isa_adj"] = isa_adj
    idx["prereq_in"] = prereq_in
    idx["prereq_out"] = prereq_out
    idx["parent_to_children_is_a"] = parent_to_children_is_a
    idx["child_to_parents_is_a"] = child_to_parents_is_a
    idx["parent_to_children_prereq"] = parent_to_children_prereq
    idx["tests_concept_by_ex"] = tests_concept_by_ex
    idx["tests_skill_by_ex"] = tests_skill_by_ex
    idx["ex_by_concept"] = ex_by_concept
    idx["ex_by_skill"] = ex_by_skill
    idx["verifies_exp_to_concept"] = verifies_exp_to_concept
    idx["verifies_concept_to_exp"] = verifies_concept_to_exp
    idx["appears_src_to_sec"] = appears_src_to_sec
    idx["part_of_parent"] = part_of_parent
    return idx


def undirected_three_rel_neighbors(idx: Dict[str, Any], node_id: str) -> Set[str]:
    return (
        idx["rel_adj"].get(node_id, set())
        | idx["isa_adj"].get(node_id, set())
        | idx["prereq_in"].get(node_id, set())
        | idx["prereq_out"].get(node_id, set())
    )


def two_hop_nodes(idx: Dict[str, Any], node_id: str) -> Set[str]:
    one_hop = undirected_three_rel_neighbors(idx, node_id)
    out: Set[str] = set()
    for mid in one_hop:
        out |= undirected_three_rel_neighbors(idx, mid)
    return out - {node_id}


def is_a_parents(idx: Dict[str, Any], node_id: str) -> Set[str]:
    return set(idx["child_to_parents_is_a"].get(node_id, set()))


def is_a_children(idx: Dict[str, Any], node_id: str) -> Set[str]:
    return set(idx["parent_to_children_is_a"].get(node_id, set()))


def collect_is_a_parents_children(idx: Dict[str, Any], node_ids: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for nid in node_ids:
        out |= is_a_parents(idx, nid)
        out |= is_a_children(idx, nid)
    return out


def collect_siblings_same_type(nodes: Dict[str, NodeInfo], idx: Dict[str, Any], node_ids: Set[str], expected_label: str) -> Set[str]:
    out: Set[str] = set()
    for aid in node_ids:
        for parent, children in idx["parent_to_children_is_a"].items():
            if aid in children:
                for c in children:
                    if c != aid and c in nodes and nodes[c].label == expected_label:
                        out.add(c)
        for parent, children in idx["parent_to_children_prereq"].items():
            if aid in children:
                for c in children:
                    if c != aid and c in nodes and nodes[c].label == expected_label:
                        out.add(c)
    return out


def task1_subtask1_distractors(
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
    answer_ids: Set[str],
    expected_label: str,
) -> Set[str]:
    rel_adj = idx["rel_adj"]
    prereq_in = idx["prereq_in"]
    prereq_out = idx["prereq_out"]
    parent_to_children_is_a = idx["parent_to_children_is_a"]
    parent_to_children_prereq = idx["parent_to_children_prereq"]

    out: Set[str] = set()
    answer_parent_child = collect_is_a_parents_children(idx, answer_ids)

    for aid in answer_ids:
        for nb in rel_adj.get(aid, set()):
            if nodes.get(nb) and nodes[nb].label == expected_label:
                out.add(nb)

        for nb in prereq_in.get(aid, set()) | prereq_out.get(aid, set()):
            if nodes.get(nb) and nodes[nb].label == expected_label:
                out.add(nb)

        # siblings by same is_a / prerequisites_for parent (shared target)
        for parent, children in parent_to_children_is_a.items():
            if aid in children:
                for c in children:
                    if c != aid and nodes.get(c) and nodes[c].label == expected_label:
                        out.add(c)
        for parent, children in parent_to_children_prereq.items():
            if aid in children:
                for c in children:
                    if c != aid and nodes.get(c) and nodes[c].label == expected_label:
                        out.add(c)

    out -= answer_ids
    out -= answer_parent_child
    return out


def gen_task1(
    subject: str,
    stage: str,
    version: str,
    merged_data_dir: Path,
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t1_s1: List[Dict[str, Any]] = []
    t1_s2: List[Dict[str, Any]] = []

    tests_concept_by_ex = idx["tests_concept_by_ex"]
    tests_skill_by_ex = idx["tests_skill_by_ex"]
    ex_by_concept = idx["ex_by_concept"]
    ex_by_skill = idx["ex_by_skill"]
    rel_adj = idx["rel_adj"]
    prereq_in = idx["prereq_in"]
    prereq_out = idx["prereq_out"]
    isa_adj = idx["isa_adj"]

    q_templates_s1_concept = [
        "【X】这道题考察了什么核心概念？",
        "要解决【X】这道题，需要用到哪些知识？",
    ]
    q_templates_s1_skill = [
        "【X】这道题考察了什么核心方法？",
        "要解决【X】这道题，需要用到哪些知识？",
    ]
    q_templates_s2 = [
        "哪些题目考察了【X】这个知识点？",
        "以下哪些题目会用到【X】？",
        "围绕【X】这一知识点，教材里有哪些例题？",
    ]

    # subtask1: Exercise -> Concept/Skill
    for ex_id, concept_ids in sorted(tests_concept_by_ex.items()):
        ex = nodes.get(ex_id)
        if not ex or ex.label != "Exercise":
            continue
        ex_text = display_exercise(ex)
        answer_ids = {x for x in concept_ids if x in nodes and nodes[x].label == "Concept"}
        answer_names = ids_to_names(nodes, answer_ids, {"Concept"})
        if not answer_names:
            continue

        sample_key = f"task1::subtask1::{ex_id}::concept"
        distractor_ids = task1_subtask1_distractors(nodes, idx, answer_ids, "Concept")
        t1_s1.append(
            {
                "id": sample_key,
                "task": "task1",
                "subtask": "subtask1",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": "Concept",
                "question": pick_question_template("task1_subtask1", sample_key, q_templates_s1_concept, f"【{ex_text}】"),
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                "meta": {"source_exercise_id": ex_id, "relation": "tests_concept", "merged_data_dir": str(merged_data_dir)},
            }
        )

    for ex_id, skill_ids in sorted(tests_skill_by_ex.items()):
        ex = nodes.get(ex_id)
        if not ex or ex.label != "Exercise":
            continue
        ex_text = display_exercise(ex)
        answer_ids = {x for x in skill_ids if x in nodes and nodes[x].label == "Skill"}
        answer_names = ids_to_names(nodes, answer_ids, {"Skill"})
        if not answer_names:
            continue

        sample_key = f"task1::subtask1::{ex_id}::skill"
        distractor_ids = task1_subtask1_distractors(nodes, idx, answer_ids, "Skill")
        t1_s1.append(
            {
                "id": sample_key,
                "task": "task1",
                "subtask": "subtask1",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": "Skill",
                "question": pick_question_template("task1_subtask1", sample_key, q_templates_s1_skill, f"【{ex_text}】"),
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                "meta": {"source_exercise_id": ex_id, "relation": "tests_skill", "merged_data_dir": str(merged_data_dir)},
            }
        )

    # subtask2: Concept/Skill -> Exercises
    for node_id, node in sorted(nodes.items(), key=lambda x: x[0]):
        if node.label not in {"Concept", "Skill"}:
            continue
        node_name = node.name.strip()
        if not node_name:
            continue

        if node.label == "Concept":
            answer_ex_ids = set(ex_by_concept.get(node_id, set()))
            one_hop_nodes = {x for x in (rel_adj.get(node_id, set()) | prereq_in.get(node_id, set()) | prereq_out.get(node_id, set())) if x in nodes and nodes[x].label == "Concept" and x != node_id}
            two_hop_all = {x for x in two_hop_nodes(idx, node_id) if x in nodes and nodes[x].label == "Concept" and x != node_id}
            distractor_source_nodes = one_hop_nodes | two_hop_all
            distractor_ex_ids = set()
            for nb in distractor_source_nodes:
                distractor_ex_ids |= set(ex_by_concept.get(nb, set()))
        else:
            answer_ex_ids = set(ex_by_skill.get(node_id, set()))
            one_hop_nodes = {x for x in (rel_adj.get(node_id, set()) | prereq_in.get(node_id, set()) | prereq_out.get(node_id, set())) if x in nodes and nodes[x].label == "Skill" and x != node_id}
            two_hop_all = {x for x in two_hop_nodes(idx, node_id) if x in nodes and nodes[x].label == "Skill" and x != node_id}
            distractor_source_nodes = one_hop_nodes | two_hop_all
            distractor_ex_ids = set()
            for nb in distractor_source_nodes:
                distractor_ex_ids |= set(ex_by_skill.get(nb, set()))

        answer_ex_ids = {x for x in answer_ex_ids if x in nodes and nodes[x].label == "Exercise"}
        answer_names = ids_to_names(nodes, answer_ex_ids, {"Exercise"})
        if not answer_names:
            continue

        distractor_ex_ids = {x for x in distractor_ex_ids if x in nodes and nodes[x].label == "Exercise" and x not in answer_ex_ids}

        sample_key = f"task1::subtask2::{node_id}"
        t1_s2.append(
            {
                "id": sample_key,
                "task": "task1",
                "subtask": "subtask2",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": "Exercise",
                "question": pick_question_template("task1_subtask2", sample_key, q_templates_s2, f"【{node_name}】"),
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, distractor_ex_ids, answer_names, sample_key),
                "meta": {"query_node_id": node_id, "query_label": node.label, "merged_data_dir": str(merged_data_dir)},
            }
        )

    return t1_s1, t1_s2


def gen_task2(
    subject: str,
    stage: str,
    version: str,
    merged_data_dir: Path,
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t2_s1: List[Dict[str, Any]] = []
    t2_s2: List[Dict[str, Any]] = []

    prereq_in = idx["prereq_in"]
    prereq_out = idx["prereq_out"]
    rel_adj = idx["rel_adj"]
    isa_adj = idx["isa_adj"]

    q_templates_s1 = [
        "在学习【X】之前，需要学习哪些知识？",
        "以下哪些知识是【X】的前置知识？",
        "要掌握【X】，应先具备哪些基础？",
    ]
    q_templates_s2 = [
        "在学习了【X】之后，下一步**最适合**学习什么知识？",
        "以下哪些知识是【X】的**最直接后置**知识？",
        "掌握【X】后，通常会**马上**继续学习哪些内容？",
    ]

    allowed_ids = {nid for nid, n in nodes.items() if n.label in ALLOWED_TASK2_LABELS and n.name.strip()}

    def collect_all_predecessors(node_id: str) -> Set[str]:
        visited: Set[str] = set()
        stack: List[str] = list(prereq_in.get(node_id, set()))
        while stack:
            cur = stack.pop()
            if cur in visited or cur not in allowed_ids:
                continue
            visited.add(cur)
            stack.extend(prereq_in.get(cur, set()))
        return visited

    for node_id in sorted(allowed_ids):
        node = nodes[node_id]
        node_name = node.name.strip()

        # Correct answers are the full prerequisite closure along directed prerequisites_for.
        all_pre_ids = collect_all_predecessors(node_id)
        pre_ids = {x for x in prereq_in.get(node_id, set()) if x in allowed_ids}
        post_ids = {x for x in prereq_out.get(node_id, set()) if x in allowed_ids}

        related_ids = {x for x in rel_adj.get(node_id, set()) if x in allowed_ids}
        parent_child_ids = {x for x in isa_adj.get(node_id, set()) if x in allowed_ids}

        if all_pre_ids:
            answer_names = ids_to_names(nodes, sorted(all_pre_ids))
            all_pre_names = ids_to_names(nodes, sorted(all_pre_ids))
            distractor_ids = (parent_child_ids | related_ids | post_ids) - all_pre_ids - {node_id}

            sample_key = f"task2::subtask1::{node_id}"
            t2_s1.append(
                {
                    "id": sample_key,
                    "task": "task2",
                    "subtask": "subtask1",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept|Skill|Experiment",
                    "question": pick_question_template("task2_subtask1", sample_key, q_templates_s1, f"【{node_name}】"),
                    "answer": answer_names,
                    "full_prerequisites": all_pre_names,
                    "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                    "meta": {
                        "query_node_id": node_id,
                        "query_label": node.label,
                        "direct_prerequisite_ids": ids_to_names(nodes, sorted(pre_ids)),
                        "full_prerequisite_ids": all_pre_names,
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

        if post_ids:
            answer_names = ids_to_names(nodes, post_ids)
            post_of_post_ids: Set[str] = set()
            for pid in post_ids:
                post_of_post_ids |= {x for x in prereq_out.get(pid, set()) if x in allowed_ids}
            distractor_ids = (parent_child_ids | related_ids | pre_ids | post_of_post_ids) - post_ids - {node_id}
            sample_key = f"task2::subtask2::{node_id}"
            t2_s2.append(
                {
                    "id": sample_key,
                    "task": "task2",
                    "subtask": "subtask2",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept|Skill|Experiment",
                    "question": pick_question_template("task2_subtask2", sample_key, q_templates_s2, f"【{node_name}】"),
                    "answer": answer_names,
                    "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                    "meta": {
                        "query_node_id": node_id,
                        "query_label": node.label,
                        "direct_successor_ids": sorted(post_ids),
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

    return t2_s1, t2_s2




def gen_task3(
    subject: str,
    stage: str,
    version: str,
    merged_data_dir: Path,
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    t3_s1: List[Dict[str, Any]] = []
    t3_s21: List[Dict[str, Any]] = []
    t3_s22: List[Dict[str, Any]] = []
    t3_s3: List[Dict[str, Any]] = []

    rel_adj = idx["rel_adj"]
    isa_adj = idx["isa_adj"]
    prereq_in = idx["prereq_in"]
    prereq_out = idx["prereq_out"]
    parent_to_children_is_a = idx["parent_to_children_is_a"]
    child_to_parents_is_a = idx["child_to_parents_is_a"]

    q_templates_s1 = [
        "以下哪些概念与【X】密切相关？",
        "以下哪些概念与【X】可以相对照着学？",
    ]
    q_templates_s21 = [
        "以下哪些知识点属于【X】这一类？",
    ]
    q_templates_s22 = [
        "【X】属于以下哪一类？",
    ]
    q_templates_s3 = [
        "以下哪些概念与【X】**直接相关**（包括分类关系或紧密关联）？",
    ]

    concept_ids = {nid for nid, n in nodes.items() if n.label == "Concept" and n.name.strip()}

    def concept_neighbors(node_id: str) -> Set[str]:
        return {
            x
            for x in (rel_adj.get(node_id, set()) | isa_adj.get(node_id, set()) | prereq_in.get(node_id, set()) | prereq_out.get(node_id, set()))
            if x in concept_ids and x != node_id
        }

    def linked_from_answers(answer_ids: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for aid in answer_ids:
            out |= concept_neighbors(aid)
        return out

    for query_id in sorted(concept_ids):
        query = nodes[query_id]
        query_name = query.name.strip()

        # subtask1: query concept -> relates_to concepts
        ans_s1 = {x for x in rel_adj.get(query_id, set()) if x in concept_ids and x != query_id}
        if ans_s1:
            d_s1 = linked_from_answers(ans_s1)
            d_s1 |= {x for x in (isa_adj.get(query_id, set()) | prereq_in.get(query_id, set()) | prereq_out.get(query_id, set())) if x in concept_ids}
            d_s1 -= ans_s1
            d_s1.discard(query_id)

            sample_key = f"task3::subtask1::{query_id}"
            answer_names = ids_to_names(nodes, ans_s1, {"Concept"})
            t3_s1.append(
                {
                    "id": sample_key,
                    "task": "task3",
                    "subtask": "subtask1",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept",
                    "question": pick_question_template("task3_subtask1", sample_key, q_templates_s1, f"【{query_name}】"),
                    "answer": answer_names,
                    "candidates": make_candidates_from_ids(nodes, d_s1, answer_names, sample_key),
                    "meta": {
                        "query_node_id": query_id,
                        "query_label": query.label,
                        "relation": "relates_to",
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

        # subtask2_1: query concept -> is_a incoming (children)
        ans_s21 = {x for x in parent_to_children_is_a.get(query_id, set()) if x in concept_ids and x != query_id}
        if ans_s21:
            d_s21 = linked_from_answers(ans_s21)
            d_s21 |= {x for x in (rel_adj.get(query_id, set()) | prereq_in.get(query_id, set()) | prereq_out.get(query_id, set())) if x in concept_ids}
            d_s21 |= {x for x in child_to_parents_is_a.get(query_id, set()) if x in concept_ids}
            d_s21 -= ans_s21
            d_s21.discard(query_id)

            sample_key = f"task3::subtask2_1::{query_id}"
            answer_names = ids_to_names(nodes, ans_s21, {"Concept"})
            t3_s21.append(
                {
                    "id": sample_key,
                    "task": "task3",
                    "subtask": "subtask2_1",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept",
                    "question": pick_question_template("task3_subtask2_1", sample_key, q_templates_s21, f"【{query_name}】"),
                    "answer": answer_names,
                    "candidates": make_candidates_from_ids(nodes, d_s21, answer_names, sample_key),
                    "meta": {
                        "query_node_id": query_id,
                        "query_label": query.label,
                        "relation": "is_a_incoming",
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

        # subtask2_2: query concept -> is_a outgoing (parents)
        ans_s22 = {x for x in child_to_parents_is_a.get(query_id, set()) if x in concept_ids and x != query_id}
        if ans_s22:
            d_s22 = linked_from_answers(ans_s22)
            d_s22 |= {x for x in (rel_adj.get(query_id, set()) | prereq_in.get(query_id, set()) | prereq_out.get(query_id, set())) if x in concept_ids}
            d_s22 |= {x for x in parent_to_children_is_a.get(query_id, set()) if x in concept_ids}
            d_s22 -= ans_s22
            d_s22.discard(query_id)

            sample_key = f"task3::subtask2_2::{query_id}"
            answer_names = ids_to_names(nodes, ans_s22, {"Concept"})
            t3_s22.append(
                {
                    "id": sample_key,
                    "task": "task3",
                    "subtask": "subtask2_2",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept",
                    "question": pick_question_template("task3_subtask2_2", sample_key, q_templates_s22, f"【{query_name}】"),
                    "answer": answer_names,
                    "candidates": make_candidates_from_ids(nodes, d_s22, answer_names, sample_key),
                    "meta": {
                        "query_node_id": query_id,
                        "query_label": query.label,
                        "relation": "is_a_outgoing",
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

        # subtask3: direct neighbors by is_a/relates_to
        ans_s3 = {x for x in (isa_adj.get(query_id, set()) | rel_adj.get(query_id, set())) if x in concept_ids and x != query_id}
        if ans_s3:
            one_hop_all = {x for x in undirected_three_rel_neighbors(idx, query_id) if x in concept_ids and x != query_id}
            two_hop_all = {x for x in two_hop_nodes(idx, query_id) if x in concept_ids and x != query_id}

            # Exact 2-hop only (distance == 2)
            d_s3 = two_hop_all - one_hop_all
            d_s3 -= ans_s3
            d_s3.discard(query_id)
            # Remove nodes that are 2-hop away from any correct option via relates_to-relates_to.
            rr2_of_answers: Set[str] = set()
            for aid in ans_s3:
                first_hop = {x for x in rel_adj.get(aid, set()) if x in concept_ids and x != aid}
                for mid in first_hop:
                    rr2_of_answers |= {x for x in rel_adj.get(mid, set()) if x in concept_ids and x != aid}
            d_s3 -= rr2_of_answers

            sample_key = f"task3::subtask3::{query_id}"
            answer_names = ids_to_names(nodes, ans_s3, {"Concept"})
            t3_s3.append(
                {
                    "id": sample_key,
                    "task": "task3",
                    "subtask": "subtask3",
                    "subject": subject,
                    "stage": stage,
                    "version": version,
                    "answer_type": "Concept",
                    "question": pick_question_template("task3_subtask3", sample_key, q_templates_s3, f"【{query_name}】"),
                    "answer": answer_names,
                    "candidates": make_candidates_from_ids(nodes, d_s3, answer_names, sample_key),
                    "meta": {
                        "query_node_id": query_id,
                        "query_label": query.label,
                        "relation": "is_a|relates_to",
                        "merged_data_dir": str(merged_data_dir),
                    },
                }
            )

    return t3_s1, t3_s21, t3_s22, t3_s3


def gen_task4(
    subject: str,
    stage: str,
    version: str,
    merged_data_dir: Path,
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    t4_s1: List[Dict[str, Any]] = []
    t4_s2: List[Dict[str, Any]] = []

    v_e2c = idx["verifies_exp_to_concept"]
    v_c2e = idx["verifies_concept_to_exp"]
    rel_adj = idx["rel_adj"]
    isa_adj = idx["isa_adj"]
    prereq_in = idx["prereq_in"]
    prereq_out = idx["prereq_out"]

    q_templates_s1 = [
        "教材中哪个实验验证了【X】原理？",
        "以下哪些实验可以验证【X】？",
        "围绕【X】，教材安排了哪些验证实验？",
    ]
    q_templates_s2 = [
        "【X】实验验证了什么原理？",
        "以下哪些概念可由【X】实验验证？",
        "通过【X】实验，可以支持哪些核心概念？",
    ]

    all_exp_ids = {nid for nid, n in nodes.items() if n.label == "Experiment"}

    # subtask1: Concept -> Experiment
    for concept_id, exp_ids in sorted(v_c2e.items()):
        concept = nodes.get(concept_id)
        if not concept or concept.label != "Concept" or not concept.name.strip():
            continue

        ans_ids = {x for x in exp_ids if x in all_exp_ids}
        answer_names = ids_to_names(nodes, ans_ids, {"Experiment"})
        if not answer_names:
            continue

        neighbor_concepts = {
            x
            for x in (
                rel_adj.get(concept_id, set())
                | isa_adj.get(concept_id, set())
                | prereq_in.get(concept_id, set())
                | prereq_out.get(concept_id, set())
            )
            if x in nodes and nodes[x].label == "Concept" and x != concept_id
        }
        distractor_ids: Set[str] = set()
        for nb in neighbor_concepts:
            distractor_ids |= set(v_c2e.get(nb, set()))
        distractor_ids = {x for x in distractor_ids if x in all_exp_ids}
        distractor_ids -= ans_ids
        sample_key = f"task4::subtask1::{concept_id}"
        t4_s1.append(
            {
                "id": sample_key,
                "task": "task4",
                "subtask": "subtask1",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": "Experiment",
                "question": pick_question_template("task4_subtask1", sample_key, q_templates_s1, f"【{concept.name.strip()}】"),
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                "meta": {"query_concept_id": concept_id, "answer_experiment_ids": sorted(ans_ids), "merged_data_dir": str(merged_data_dir)},
            }
        )

    # subtask2: Experiment -> Concept
    for exp_id, concept_ids in sorted(v_e2c.items()):
        exp = nodes.get(exp_id)
        if not exp or exp.label != "Experiment" or not exp.name.strip():
            continue

        ans_ids = {x for x in concept_ids if x in nodes and nodes[x].label == "Concept"}
        answer_names = ids_to_names(nodes, ans_ids, {"Concept"})
        if not answer_names:
            continue

        distractor_ids: Set[str] = set()
        answer_parent_child = collect_is_a_parents_children(idx, ans_ids)
        sibling_ids = collect_siblings_same_type(nodes, idx, ans_ids, "Concept")
        distractor_ids |= sibling_ids
        for aid in ans_ids:
            for nb in rel_adj.get(aid, set()) | prereq_in.get(aid, set()) | prereq_out.get(aid, set()):
                if nb in nodes and nodes[nb].label == "Concept":
                    distractor_ids.add(nb)
        distractor_ids -= ans_ids
        distractor_ids -= answer_parent_child

        sample_key = f"task4::subtask2::{exp_id}"
        t4_s2.append(
            {
                "id": sample_key,
                "task": "task4",
                "subtask": "subtask2",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": "Concept",
                "question": pick_question_template("task4_subtask2", sample_key, q_templates_s2, f"【{exp.name.strip()}】"),
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, distractor_ids, answer_names, sample_key),
                "meta": {"query_experiment_id": exp_id, "merged_data_dir": str(merged_data_dir)},
            }
        )

    return t4_s1, t4_s2


def climb_to_chapters(section_id: str, part_of_parent: DefaultDict[str, Set[str]], nodes: Dict[str, NodeInfo]) -> Set[str]:
    out: Set[str] = set()
    q = [section_id]
    visited: Set[str] = set()
    while q:
        cur = q.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for p in part_of_parent.get(cur, set()):
            if p in nodes and nodes[p].label == "Chapter":
                out.add(p)
            q.append(p)
    return out


def subject_granularity(subject: str) -> str:
    if subject in {"数学", "物理", "化学"}:
        return "Section"
    if "生物" in subject:
        return "Chapter"
    return "Section"


def gen_task5(
    subject: str,
    stage: str,
    version: str,
    merged_data_dir: Path,
    nodes: Dict[str, NodeInfo],
    idx: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    appears = idx["appears_src_to_sec"]
    part_of_parent = idx["part_of_parent"]

    gran = subject_granularity(subject)
    all_sections = {nid for nid, n in nodes.items() if n.label == "Section"}
    all_chapters = {nid for nid, n in nodes.items() if n.label == "Chapter"}

    for src_id, sec_ids in sorted(appears.items()):
        src = nodes.get(src_id)
        if not src or src.label not in {"Concept", "Skill", "Experiment"}:
            continue
        src_name = src.name.strip()
        if not src_name:
            continue

        sec_ids = {x for x in sec_ids if x in all_sections}
        if not sec_ids:
            continue

        if gran == "Section":
            ans_ids = sec_ids
            candidate_ids = all_sections - ans_ids
        else:
            chapter_ids: Set[str] = set()
            for sid in sec_ids:
                chapter_ids |= climb_to_chapters(sid, part_of_parent, nodes)
            ans_ids = {x for x in chapter_ids if x in all_chapters}
            if not ans_ids:
                continue
            candidate_ids = all_chapters - ans_ids

        answer_names = ids_to_names(nodes, ans_ids, {gran})
        if not answer_names:
            continue

        sample_key = f"task5::{src_id}"
        out.append(
            {
                "id": sample_key,
                "task": "task5",
                "subject": subject,
                "stage": stage,
                "version": version,
                "answer_type": gran,
                "question": f"【{src_name}】这一知识点最早出现在教材的哪个{('章节' if gran == 'Section' else '章')}？",
                "answer": answer_names,
                "candidates": make_candidates_from_ids(nodes, candidate_ids, answer_names, sample_key),
                "meta": {"source_id": src_id, "source_label": src.label, "granularity": gran, "merged_data_dir": str(merged_data_dir)},
            }
        )

    return out


def extract_answer_names(sample: Dict[str, Any]) -> List[str]:
    ans = sample.get("answer")
    if isinstance(ans, list):
        return clean_name_list(ans)
    return []


def validate_records(records: List[Dict[str, Any]], file_tag: str) -> None:
    seen: Set[str] = set()
    for i, rec in enumerate(records, 1):
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            raise ValueError(f"{file_tag}: invalid id at line {i}")
        if rid in seen:
            raise ValueError(f"{file_tag}: duplicate id: {rid}")
        seen.add(rid)

        answers = set(extract_answer_names(rec))
        if not answers:
            raise ValueError(f"{file_tag}: empty answer at id={rid}")

        cands = rec.get("candidates")
        if not isinstance(cands, list):
            raise ValueError(f"{file_tag}: candidates must be list at id={rid}")
        overlap = answers & set(clean_name_list(cands))
        if overlap:
            raise ValueError(f"{file_tag}: candidates include answer at id={rid}, overlap={sorted(overlap)[:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark JSONL from merged_data")
    parser.add_argument("--base-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_pairs = parse_pairs(args.pair)
    tasks = list(dict.fromkeys(args.tasks))

    merged_dirs = discover_merged_data_dirs(args.base_dir, selected_pairs)

    outputs: Dict[str, List[Dict[str, Any]]] = {
        "task1_subtask1": [],
        "task1_subtask2": [],
        "task2_subtask1": [],
        "task2_subtask2": [],
        "task3_subtask3": [],
        "task4_subtask1": [],
        "task4_subtask2": [],
        "task5": [],
    }

    for subject, stage, version, merged_data_dir in merged_dirs:
        nodes = build_node_map(load_json_list(merged_data_dir / "nodes_all.json"))
        edges = load_json_list(merged_data_dir / "edges_all.json")
        idx = build_edge_indexes(edges)

        if "task1" in tasks:
            s1, s2 = gen_task1(subject, stage, version, merged_data_dir, nodes, idx)
            outputs["task1_subtask1"].extend(s1)
            outputs["task1_subtask2"].extend(s2)

        if "task2" in tasks:
            s1, s2 = gen_task2(subject, stage, version, merged_data_dir, nodes, idx)
            outputs["task2_subtask1"].extend(s1)
            outputs["task2_subtask2"].extend(s2)

        if "task3" in tasks:
            _, _, _, s3 = gen_task3(subject, stage, version, merged_data_dir, nodes, idx)
            outputs["task3_subtask3"].extend(s3)

        if "task4" in tasks:
            s1, s2 = gen_task4(subject, stage, version, merged_data_dir, nodes, idx)
            outputs["task4_subtask1"].extend(s1)
            outputs["task4_subtask2"].extend(s2)

        if "task5" in tasks:
            outputs["task5"].extend(gen_task5(subject, stage, version, merged_data_dir, nodes, idx))

    selected_keys = [
        k
        for k in outputs
        if (k.startswith("task1") and "task1" in tasks)
        or (k.startswith("task2") and "task2" in tasks)
        or (k.startswith("task3") and "task3" in tasks)
        or (k.startswith("task4") and "task4" in tasks)
        or (k == "task5" and "task5" in tasks)
    ]
    counts: Dict[str, int] = {}
    for k in selected_keys:
        validate_records(outputs[k], k)
        counts[k] = write_jsonl(args.output_dir / f"{k}.jsonl", outputs[k])

    empty_candidates_counts: Dict[str, int] = {
        k: sum(1 for rec in outputs[k] if isinstance(rec.get("candidates"), list) and len(rec["candidates"]) == 0)
        for k in selected_keys
    }
    task2_subtask1_full_not_equal_direct_count = sum(
        1
        for rec in outputs.get("task2_subtask1", [])
        if set(clean_name_list(rec.get("answer") if isinstance(rec.get("answer"), list) else []))
        != set(
            clean_name_list(
                rec.get("full_prerequisites") if isinstance(rec.get("full_prerequisites"), list) else []
            )
        )
    )

    summary = {
        "base_dir": str(args.base_dir),
        "output_dir": str(args.output_dir),
        "selected_pairs": [f"{a}:{b}" for a, b in sorted(selected_pairs)] if selected_pairs else "ALL",
        "merged_data_dirs": len(merged_dirs),
        "tasks": tasks,
        "outputs": selected_keys,
        "counts": counts,
        "empty_candidates_counts": empty_candidates_counts,
        "task2_subtask1_full_not_equal_direct_count": task2_subtask1_full_not_equal_direct_count,
    }
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
