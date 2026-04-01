# K12-GraphBench Pipeline

基于 K12 教材构建知识图谱（KG），并在此基础上生成 Benchmark 评测题和 SFT 训练数据的完整流水线。

## Pipeline 总览

```
教材 Markdown
    │
    ▼
┌─────────────┐
│ segmentation│  ── TOC 解析 → 章节切分
└─────┬───────┘
      ▼
┌─────────────┐
│ build_graph │  ── LLM 提取知识图谱
└─────┬───────┘
      ▼
┌─────────────┐
│ merge_graph │  ── 书级合并 → 学段级 → 学科级 → 按类型拆分
└─────┬───────┘
      ▼
┌─────────────┐
│ check_graph │  ── 环检测
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

### 1. segmentation — 教材切分

分两步将教材 Markdown 切分为独立章节文件。

| 文件 | 功能 |
|------|------|
| `segment_textbooks.py` | Step 1：通过 TOC 识别课本章节结构，生成 `sections_index.json` 和 `metadata.md` |
| `split_by_sections_index.py` | Step 2：根据 `sections_index.json` 将整本书 MD 切分为分章节 MD，清理图片链接等 |

### 2. build_graph — 知识图谱提取

调用 LLM 从章节文本中提取结构化知识图谱（节点 + 边）。

| 文件 | 功能 |
|------|------|
| `extract_kg_from_textbook.py` | 主 pipeline：遍历章节、调用 LLM、解析响应、生成唯一节点 ID |
| `llm_client.py` | LLM 客户端抽象基类及 OpenAI 实现，处理 API 调用和 JSON 响应解析 |
| `prompt_builder.py` | 读取 prompt 模板，填充占位符（如 `{{Section_Markdown_Content}}`） |
| `run.sh` | 批量执行脚本，支持学科/年级/章节等参数，自动跳过已完成文件 |
| `prompt.txt` | 理科的 KG 提取 prompt 模板 |

### 3. merge_graph — 图谱合并

将各章节/书级 KG 逐层合并为学段级、学科级图谱，并按类型拆分输出。

| 文件 | 功能 |
|------|------|
| `merge_book_kg.py` | 书级合并：全局 ID 重编号、概念/技能去重、边去重与冲突记录 |
| `build_merged_graph_hierarchy.py` | 层级合并：从书级 → 学科+学段级 → 学科级，同名节点去重、边重映射 |
| `build_by_type_from_subject_graph.py` | 将学科级图谱按节点/边类型拆分为独立 JSON，并合并 `tests_*` 边 |

### 4. check_graph — 图谱校验

| 文件 | 功能 |
|------|------|
| `check_cycles.py` | DFS 检测 `is_a` / `prerequisites_for` 边中的有向环 |

### 5. exercise — 习题补全

调用 LLM 补全习题的答案、难度、题型、解析等字段。

| 文件 | 功能 |
|------|------|
| `generate_exercise_json.py` | 根据习题文本和章节 KG，用 LLM 补全习题答案 |
| `run.sh` | 加载环境变量并调用 `generate_exercise_json.py` |

### 6. benchmark — Benchmark 评测题生成

基于知识图谱结构生成多种类型的评测题。

| 文件 | 功能 |
|------|------|
| `generate_benchmark.py` | 从 `merged_data` 生成多任务 benchmark JSONL |
| `score_candidate_similarity.py` | 用文本相似度为候选干扰项打分 |
| `filter_candidates.py` | 过滤干扰项候选 |
| `build_qa_tmp_for_eval.py` | 构造四选一评测格式的 QA JSONL（临时测试版） |

### 7. test_bench — 模型评测

测试模型在bench上的表现（目前只测了qwen3-32b）。

| 文件 | 功能 |
|------|------|
| `eval_multiselect.py` | 多选题评测 |
| `configs/task_k12_multiselect.yaml` | 评测任务配置（选项标签、prompt 模板等） |
| `configs/models/*.yaml` | 各模型的 API 端点和超参配置 |
| `vllm_scripts/*.sh` | vLLM 服务启动脚本 |

### 8. sft_qa — SFT 训练数据生成（还是之前测小学数学的版本）

调用 LLM 基于图谱节点和关系生成问答对，用于模型微调。

| 文件 | 功能 |
|------|------|
| `generate_qa.py` | 主生成脚本：读取节点/关系 JSON + prompt 模板，调用 API 生成 QA JSONL |
| `tests_to_qa_jsonl.py` | 将 tests 关系转换为 QA JSONL 格式 |
| `prompt_*.txt` | 各类型（概念/技能/习题/关系）的 QA 生成 prompt 模板 |

