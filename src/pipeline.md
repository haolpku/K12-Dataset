# K12-GraphBench Pipeline

基于 K12 教材构建知识图谱（KG），并在此基础上生成 Benchmark 评测题和 SFT 训练数据的完整流水线。

当前图谱主线统一放在 `src/kg/`，详细技术文档见 `docs.md`。

## 快速开始

```bash
# 1. 配置
cp config/.env.example config/.env
# 编辑 .env 填入 API 密钥

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行主线中的合并阶段
python src/kg/merge_kg.py --config config/default.yaml
```

## Pipeline 流程

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

## 配置

所有路径和 LLM 参数统一在 `config/default.yaml` 中配置：

```yaml
paths:
  books_yaml: ../books.yaml
  input_dir: ../input
  workspace_dir: ../workspace
  output_dir: ../data
```

## 各模块说明

| 模块 | 功能 |
|------|------|
| `kg/` | 图谱主线模块，包含 `segment_textbooks.py`、`extract_kg_from_textbook.py`、`merge_kg.py`、`check_cycles.py` 四个入口脚本 |
| `exercise/` | LLM 习题补全 |
| `benchmark/` | Benchmark 评测题生成 |
| `test_bench/` | 模型评测 |
| `sft_qa/` | SFT QA 数据生成 |
| `utils/` | 共享工具（IO、配置、LLM 客户端） |

详细技术文档见 [`docs.md`](docs.md)。
