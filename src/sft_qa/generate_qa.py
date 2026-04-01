#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从概念 / 技能 / 习题 / 关系类 JSON 中读取条目，结合 prompt 模板，
调用 GPT 系列模型生成用于 SFT 的 QA 数据（JSONL）。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import requests


def load_items(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"输入 JSON 不是数组: {input_path}")
    return data


def load_prompt_template(prompt_path: Path) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt_fields(
    item: Dict[str, Any],
    mode: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    根据节点 / 关系的结构构造 prompt 占位符字典，以及用于输出 JSONL 的 name：
    
    - 节点模式（mode == "node"）：
        可用占位符：
          {{label}}      节点类型（Concept / Skill / Exercise）
          {{name}}       节点名称（Concept/Skill 使用）
          {{stem}}       题目题干（Exercise 使用）
          {{answer}}     题目答案（Exercise 使用）
          {{attributes}} 经过裁剪后的属性 dict（会被序列化为 JSON）
    
      裁剪规则：
        - Concept: 从 properties 删除 name、importance、examples
        - Skill:   从 properties 删除 examples
        - Exercise:从 properties 删除 difficulty（stem 和 answer 单独提取）
    
    - 关系模式（mode == "edge"）：
        可用占位符：
          {{type}}         边类型（is_a / prerequisites_for / relates_to / verifies 等）
          {{source_name}}  源节点名称（source_name 优先，其次 source id）
          {{target_name}}  目标节点名称（target_name 优先，其次 target id）
          {{attributes}}   关系属性（删除 page / evidence_page 等与页码相关字段）
    """
    properties = item.get("properties", {}) or {}
    
    if mode == "node":
        label = item.get("label")
        props = dict(properties)
        
        # 依据 label 裁剪属性
        if label == "Concept":
            for k in ("name", "importance", "examples"):
                props.pop(k, None)
            name = properties.get("name") or item.get("id", "")
            fields: Dict[str, Any] = {
                "label": label or "",
                "name": name,
                "attributes": props,
            }
        elif label == "Skill":
            props.pop("examples", None)
            name = properties.get("name") or item.get("id", "")
            fields: Dict[str, Any] = {
                "label": label or "",
                "name": name,
                "attributes": props,
            }
        elif label == "Exercise":
            # Exercise 节点特殊处理：提取 stem 和 answer，删除 difficulty
            props.pop("difficulty", None)
            stem = properties.get("stem", "")
            answer = properties.get("answer", "")
            # 用于 JSONL 输出的 name 字段，使用 stem 的前50个字符或 id
            name = stem[:50] if stem else item.get("id", "")
            fields: Dict[str, Any] = {
                "label": label or "",
                "stem": stem,
                "answer": answer,
                "attributes": props,
            }
        else:
            # 其他类型节点，使用默认处理
            name = properties.get("name") or item.get("id", "")
            fields: Dict[str, Any] = {
                "label": label or "",
                "name": name,
                "attributes": props,
            }
        
        return name, fields
    
    # 关系模式
    edge_type = item.get("type", "Relation")
    source_name = item.get("source_name") or item.get("source") or ""
    target_name = item.get("target_name") or item.get("target") or ""
    
    attrs = dict(properties)
    # 删除与页码相关的字段
    attrs.pop("page", None)
    attrs.pop("evidence_page", None)
    
    # 用于 JSONL 中的可读 relationship 字段，便于调试
    arrow = "<->" if edge_type == "relates_to" else "->"
    rel_str = ""
    if source_name and target_name:
        rel_str = f"{source_name} {arrow} {target_name}"
    
    name = f"{edge_type} | {rel_str}" if rel_str else edge_type
    
    fields = {
        "type": edge_type,
        "source_name": source_name,
        "target_name": target_name,
        "attributes": attrs,
    }
    return name, fields


def render_prompt(
    template: str,
    n: int,
    fields: Dict[str, Any],
) -> str:
    """
    将占位符字典填充进 prompt 模板。
    
    - {n}        用命令行参数替换
    - 其它占位符形如 {{key}}，从 fields[key] 取值
    - attributes 字段会以 JSON 格式嵌入，便于 LLM 精确读取
    """
    prompt = template.replace("{n}", str(n))
    
    for key, value in fields.items():
        if key == "attributes":
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            value_str = str(value)
        placeholder = "{{" + key + "}}"
        prompt = prompt.replace(placeholder, value_str)
    
    return prompt


def call_openai_chat_raw(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """
    调用 /v1/chat/completions 接口，返回模型的原始文本输出（不做 JSON 解析）。
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=600)
    except Exception as e:
        # 网络级错误（连不上 / 超时等），也返回错误文本，方便写入 raw 日志排查
        err_text = f"[request_error] {type(e).__name__}: {e}"
        print(f"[warn] OpenAI API 请求异常: {err_text}", file=sys.stderr)
        return err_text

    # 无论状态码如何，都尽量拿到文本内容返回，方便写入 raw 日志排查
    if resp.status_code != 200:
        # 打印一条简短错误信息到 stderr，但仍然返回 resp.text
        print(
            f"[warn] OpenAI API 调用失败: status={resp.status_code}, body={resp.text}",
            file=sys.stderr,
        )
        return resp.text

    # 200 时按正常 chat.completions 结构解析
    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"无法解析 API 返回结果: {e}; 原始数据: {data}") from e

    return content


def parse_model_output_text(
    content: str,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    将模型返回的原始文本解析为 JSON（对象或数组）。

    策略：
      1. 先整体 json.loads；
      2. 失败则在文本中截取首个 '{' 或 '[' 到最后一个 '}' 或 ']' 之间的片段再解析。
    """
    text = content.strip()
    try:
        return json.loads(text)
    except Exception:
        # 在返回内容中寻找首个 JSON 数组或对象
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [p for p in (start_obj, start_arr) if p != -1]
        if not candidates:
            raise RuntimeError(
                "模型返回的内容不是合法 JSON，且找不到 '{' 或 '[' 起始: "
                f"{content}"
            )
        start = min(candidates)

        # 粗暴从起始位置截到最后一个 '}' 或 ']'
        end_obj = text.rfind("}")
        end_arr = text.rfind("]")
        end_candidates = [p for p in (end_obj, end_arr) if p != -1]
        if not end_candidates:
            raise RuntimeError(
                "模型返回的内容不是合法 JSON，且找不到 '}' 或 ']' 结束: "
                f"{content}"
            )
        end = max(end_candidates) + 1

        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except Exception as e:
            raise RuntimeError(
                f"模型返回的内容不是合法 JSON，截取片段解析仍失败: {e}; snippet={snippet}"
            ) from e


def extract_qa_list(
    model_output: Union[Dict[str, Any], List[Any]]
) -> List[Dict[str, str]]:
    """
    将模型返回结构统一转成 [{'question': ..., 'answer': ...}, ...]。
    """
    if isinstance(model_output, dict):
        # 允许模型直接返回单个对象或包在某个 key 下
        if "question" in model_output and "answer" in model_output:
            return [
                {
                    "question": str(model_output["question"]),
                    "answer": str(model_output["answer"]),
                }
            ]
        # 如果是 {"items": [...]} 之类的结构
        for v in model_output.values():
            if isinstance(v, list):
                model_output = v
                break

    if isinstance(model_output, list):
        qa_list: List[Dict[str, str]] = []
        for item in model_output:
            if not isinstance(item, dict):
                continue
            q = item.get("question")
            a = item.get("answer")
            if q is None or a is None:
                continue
            qa_list.append({"question": str(q), "answer": str(a)})
        return qa_list

    raise ValueError(f"无法从模型输出中提取 QA 列表: {model_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据图谱 JSON 与 prompt 模板生成 SFT QA 数据（JSONL）"
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="输入 JSON 文件路径，例如 concepts.json / skills.json / exercises.json / relates_to.json 等",
    )
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="prompt 模板文件路径，例如 src/sft_qa/prompt_cpt.txt",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help=(
            "输出 JSONL 文件路径："
            "node 模式概念/技能为 {id,name,question,answer}；"
            "node 模式习题为 {id,stem,question,answer}；"
            "edge 模式为 {id,relationship,question,answer}"
        ),
    )
    parser.add_argument(
        "-n",
        type=int,
        default=3,
        help="每个输入条目希望模型生成的问题数量（传给 prompt 中的 {n}）",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="要调用的 GPT 模型名称（默认: gpt-4.1-mini，可自行修改）",
    )
    parser.add_argument(
        "--mode",
        choices=["node", "edge"],
        default="node",
        help="生成模式：node 表示概念/技能/习题等节点类，edge 表示关系类（is_a、prerequisites_for 等）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="开启断点续跑：如果输出 JSONL 已存在，则跳过其中已经出现过的 id，并在文件尾部追加新样本",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="仅处理前 max-items 条（调试用）",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("错误: 请先在环境变量中设置 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")

    input_path = Path(args.input_json)
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output_jsonl)

    items = load_items(input_path)
    if args.max_items is not None:
        items = items[: args.max_items]
    
    template = load_prompt_template(prompt_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 断点续跑：收集已经写入过的 id，后续跳过
    processed_ids = set()
    if args.resume and output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    rec_id = rec.get("id")
                    if rec_id:
                        processed_ids.add(rec_id)
        except Exception as e:
            print(f"[warn] 读取已有输出文件失败，将不进行断点续跑: {e}", file=sys.stderr)
            processed_ids = set()
            args.resume = False

    # 写出模式：续跑时追加，否则重写
    out_mode = "a" if args.resume and output_path.exists() else "w"
    out_f = output_path.open(out_mode, encoding="utf-8")
    
    total_qa = 0
    raw_log_path = output_path.with_suffix(output_path.suffix + ".raw.txt")
    
    try:
        for idx, item in enumerate(items, start=1):
            item_id = item.get("id", f"item_{idx}")
    
            if item_id in processed_ids:
                print(f"[{idx}/{len(items)}] 跳过已处理 id={item_id}")
                continue
    
            # 根据模式构造 prompt 占位符与可读 name
            name, fields = build_prompt_fields(item, args.mode)
            prompt = render_prompt(template, args.n, fields)
    
            print(f"[{idx}/{len(items)}] 调用模型生成 QA: id={item_id}, name={name}")
            try:
                content = call_openai_chat_raw(
                    prompt=prompt,
                    model=args.model,
                    api_key=api_key,
                    base_url=base_url,
                )
            except Exception as e:
                # API 级别错误，直接跳过，并在 stderr 打印
                print(f"  [错误] 调用模型接口失败: {e}", file=sys.stderr)
                continue

            # 先把原始输出保存下来，方便之后排查
            try:
                with raw_log_path.open("a", encoding="utf-8") as rf:
                    rf.write(f"===== id={item_id} name={name} =====\n")
                    rf.write(content)
                    rf.write("\n\n")
            except Exception as e:
                print(f"  [警告] 写入原始输出日志失败: {e}", file=sys.stderr)

            # 然后再尝试解析为结构化 JSON
            try:
                model_output = parse_model_output_text(content)
                qa_list = extract_qa_list(model_output)
            except Exception as e:
                print(
                    f"  [警告] 解析模型输出失败，将跳过本条目（原始输出已写入 {raw_log_path.name}）: {e}",
                    file=sys.stderr,
                )
                continue

            for qa in qa_list:
                # 基础字段：id + QA
                # 注意：这里显式控制键的插入顺序，以保证 JSONL 中字段顺序稳定：
                # - 节点（非 Exercise）：id, name, question, answer
                # - 节点（Exercise）：   id, stem, question, answer
                # - 关系 edge：         id, relationship, question, answer
                record: Dict[str, Any] = {"id": item_id}

                if args.mode == "edge":
                    # 关系模式：先写 relationship，再写 QA
                    record["relationship"] = name
                    record["question"] = qa["question"]
                    record["answer"] = qa["answer"]
                else:
                    label = item.get("label")
                    if label == "Exercise":
                        # 习题：按要求输出 {id, stem, question, answer}
                        stem = (item.get("properties") or {}).get("stem", "")
                        record["stem"] = stem
                        record["question"] = qa["question"]
                        record["answer"] = qa["answer"]
                    else:
                        # 概念 / 技能等：使用 name 字段
                        record["name"] = name
                        record["question"] = qa["question"]
                        record["answer"] = qa["answer"]

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_qa += 1

    finally:
        out_f.close()

    print(f"\n完成！共写入 {total_qa} 条 QA 到 {output_path}")


if __name__ == "__main__":
    main()
