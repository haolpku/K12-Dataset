# K12-GraphBench Pipeline

基于 K12 教材构建知识图谱（KG），并在此基础上生成 Benchmark 评测题和 SFT 训练数据的完整流水线。

## Pipeline 总览

```
教材 Markdown
    │
    ▼
┌─────────────┐
│     kg      │  ── segmentation → build_graph → merge_graph → check_graph
└─────┬───────┘
      ▼
┌──────────┐     ┌───────────┐     ┌──────────┐
│ exercise │     │ benchmark │     │  sft_qa  │
│ 习题补全  │     │ 评测题生成 │     │ SFT 数据 │
└──────────┘     └─────┬─────┘     └──────────┘
                       ▼
                ┌────────────┐
                │ test_bench │  ── 模型评测
                └────────────┘
```

## 各模块说明

### 1. kg — 图谱主线

图谱主线已经统一收拢到 `src/kg/`，内部按四个顺序入口脚本组织。

| 文件 | 功能 |
|------|------|
| `segment_textbooks.py` | 读取 `books.yaml` 和教材 Markdown，解析 TOC，生成 `sections_index.json` 并切分章节文件 |
| `extract_kg_from_textbook.py` | 读取切分后的章节 Markdown，调用 LLM 抽取 chapter 级 KG，输出到 `data/chapter_kg/` |
| `prompt.txt` | `extract_kg_from_textbook.py` 默认使用的 KG 抽取 prompt 模板 |
| `merge_kg.py` | 将 `chapter_kg` 逐层合并为 `book_kg`、`subject_stage_kg`、`subject_kg`、`global_kg` |
| `check_cycles.py` | 对合并图谱执行环检测，当前检查 `is_a` 和 `prerequisites_for` 两类边 |

### 2. exercise — 习题补全

调用 LLM 补全习题的答案、难度、题型、解析等字段。

| 文件 | 功能 |
|------|------|
| `generate_exercise_json.py` | 根据习题文本和章节 KG，用 LLM 补全习题答案 |
| `run.sh` | 加载环境变量并调用 `generate_exercise_json.py` |

### 3. benchmark — Benchmark 评测题生成

基于知识图谱结构生成多种类型的评测题。

| 文件 | 功能 |
|------|------|
| `generate_benchmark.py` | 基于图谱索引与任务配置生成 benchmark 原始题目 |
| `build_qa.py` | 将 benchmark 原始题目转换为评测使用的多选 QA JSONL |
| `bench.md` | benchmark 模块专属说明文档 |

### 4. test_bench — 模型评测

测试模型在bench上的表现（目前只测了qwen3-32b）。

| 文件 | 功能 |
|------|------|
| `eval_multiselect.py` | 多选题评测 |
| `configs/task_k12_multiselect.yaml` | 评测任务配置（选项标签、prompt 模板等） |
| `configs/models/*.yaml` | 各模型的 API 端点和超参配置 |
| `vllm_scripts/*.sh` | vLLM 服务启动脚本 |

### 5. sft_qa — SFT 训练数据生成

调用 LLM 基于图谱节点和关系生成问答对，用于模型微调。

| 文件 | 功能 |
|------|------|
| `generate_qa.py` | 按节点或关系类型调用模板与 API，生成分片 QA 结果 |
| `tests_to_qa.py` | 将 `tests_concept` / `tests_skill` 关系转换为 QA 数据 |
| `merge_qa.py` | 汇总分片结果并生成合并后的 QA 产物 |
| `prompts/*.txt` | 各类型节点与关系的 prompt 模板 |
| `sft.md` / `sft_v2.md` | SFT 模块说明与细化设计文档 |

