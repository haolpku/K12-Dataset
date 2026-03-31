#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 output 下每本书 merged_kg.json 中 is_a / prerequisites_for 是否存在有向环。

python3 src/check_graph/check_cycles.py --output-root output
python3 src/check_graph/check_cycles.py --output-root output --save-report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

TARGET_TYPES = ("is_a", "prerequisites_for")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check cycles for is_a and prerequisites_for in merged_kg.json files")
    parser.add_argument("--output-root", default="output", help="Root output directory")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report to src/check_graph/cycle_report.json")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root is not object: {path}")
    return data


def build_graph(edges: List[Any], edge_type: str) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        if not isinstance(e, dict):
            continue
        if e.get("type") != edge_type:
            continue
        s = e.get("source")
        t = e.get("target")
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        s = s.strip()
        t = t.strip()
        if not s or not t:
            continue
        graph[s].add(t)
        if t not in graph:
            graph[t] = graph[t]
    return graph


def find_cycles(graph: Dict[str, Set[str]], max_cycles: int = 200) -> List[List[str]]:
    """基于 DFS 的简单环枚举（去重），最多返回 max_cycles 个环。"""
    cycles: List[List[str]] = []
    seen_cycle_keys: Set[Tuple[str, ...]] = set()

    nodes = sorted(graph.keys())

    def canonicalize(cycle_nodes: List[str]) -> Tuple[str, ...]:
        # cycle_nodes 形如 [a,b,c,a]，去掉尾部重复点后做最小旋转规范化
        core = cycle_nodes[:-1]
        if not core:
            return tuple()
        n = len(core)
        rotations = [tuple(core[i:] + core[:i]) for i in range(n)]
        return min(rotations)

    def dfs(start: str, current: str, stack: List[str], in_stack: Set[str]) -> None:
        if len(cycles) >= max_cycles:
            return

        for nxt in graph.get(current, set()):
            if nxt == start and len(stack) >= 2:
                cycle = stack + [start]
                key = canonicalize(cycle)
                if key and key not in seen_cycle_keys:
                    seen_cycle_keys.add(key)
                    cycles.append(cycle)
                continue
            if nxt in in_stack:
                continue
            # 为减少重复搜索，只向字典序不小于起点的节点扩展
            if nxt < start:
                continue
            in_stack.add(nxt)
            stack.append(nxt)
            dfs(start, nxt, stack, in_stack)
            stack.pop()
            in_stack.remove(nxt)

    for start in nodes:
        dfs(start, start, [start], {start})
        if len(cycles) >= max_cycles:
            break

    return cycles


def check_file(path: Path) -> Dict[str, Any]:
    data = load_json(path)
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        edges = []

    result: Dict[str, Any] = {
        "file": str(path),
        "is_a": {"edge_count": 0, "has_cycle": False, "cycle_count": 0, "cycles": []},
        "prerequisites_for": {"edge_count": 0, "has_cycle": False, "cycle_count": 0, "cycles": []},
    }

    for t in TARGET_TYPES:
        g = build_graph(edges, t)
        edge_count = sum(len(v) for v in g.values())
        cycles = find_cycles(g)
        result[t] = {
            "edge_count": edge_count,
            "has_cycle": len(cycles) > 0,
            "cycle_count": len(cycles),
            "cycles": cycles,
        }

    return result


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    if not root.exists():
        raise FileNotFoundError(f"Output root not found: {root}")

    files = sorted(root.glob("**/merged_kg.json"))
    if not files:
        print("No merged_kg.json files found.")
        return

    reports = [check_file(p) for p in files]

    total_files = len(reports)
    is_a_cycle_files = sum(1 for r in reports if r["is_a"]["has_cycle"])
    pre_cycle_files = sum(1 for r in reports if r["prerequisites_for"]["has_cycle"])

    print(f"FILES: {total_files}")
    print(f"is_a cycle files: {is_a_cycle_files}")
    print(f"prerequisites_for cycle files: {pre_cycle_files}")

    print("\nFiles with cycles:")
    found_any = False
    for r in reports:
        isa = r["is_a"]
        pre = r["prerequisites_for"]
        if not isa["has_cycle"] and not pre["has_cycle"]:
            continue
        found_any = True
        print(f"- {r['file']}")
        if isa["has_cycle"]:
            print(f"  is_a cycles: {isa['cycle_count']}")
        if pre["has_cycle"]:
            print(f"  prerequisites_for cycles: {pre['cycle_count']}")

    if not found_any:
        print("- None")

    if args.save_report:
        report_path = Path("src/check_graph/cycle_report.json")
        report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
