# K12-Dataset 总体与图谱文档

> 版本：2026-04

## 文档范围

这份文档只负责三件事：

1. 说明仓库的总体架构与分发方式
2. 说明知识图谱主线 pipeline
3. 说明图谱产物的数据层级与核心 schema

下面这些内容不再在本文件里展开，避免重复：

- Benchmark 模块设计：[`src/benchmark/bench.md`](./benchmark/bench.md)
- SFT QA 模块现状说明：[`src/sft_qa/sft.md`](./sft_qa/sft.md)
- SFT QA v2 细化设计：[`src/sft_qa/sft_v2.md`](./sft_qa/sft_v2.md)

如果后文和代码冲突，以当前代码为准；如果和 benchmark / sft 的专属文档冲突，则对应模块以内部分文档为准。

## 1. 总体架构

K12-Dataset 是一套面向 K12 教材的知识图谱构建与评测 pipeline。核心主线是：

1. 从教材 Markdown 切分章节
2. 对章节调用 LLM 抽取 chapter 级图谱
3. 将 chapter 图谱逐层合并为 book / subject_stage / subject / global
4. 对合并图谱做基本结构校验
5. 在图谱之上生成 benchmark、SFT QA 等下游数据

当前确认采用“两层分发”设计：

1. GitHub 仓库只放 `pipeline + 文档 + 极小 demo`
2. Hugging Face dataset repo 只放 `全量产物 + 版本快照 + 统计清单`

这意味着：

- `demo/` 是 GitHub 展示层
- `data/` 是本地完整产物目录，不是 GitHub 自带完整数据
- 完整数据集通过 Hugging Face 的 `v1/`、`v2/` 等 snapshot 发布

## 2. 当前设计决议

### 2.1 主线模块边界

本仓库目前把图谱主线统一收拢到 `src/kg/`。

`src/kg/` 下当前包含四个顺序入口脚本：

- `segment_textbooks.py`
- `extract_kg_from_textbook.py`
- `merge_kg.py`
- `check_cycles.py`

它们共同构成“教材 -> 图谱”的主链路，也是本文件重点覆盖的部分。

下游模块：

- `src/exercise/`
- `src/benchmark/`
- `src/test_bench/`
- `src/sft_qa/`

这些模块依赖图谱产物，但具体任务设计不在本文件内展开。

### 2.2 统一配置语义

代码层已经统一使用 `src/utils/config.py` 解析配置，并约定三个核心路径字段：

- `input_dir`：原始教材输入目录
- `workspace_dir`：中间产物目录
- `output_dir`：本地完整产物目录

其中：

- `workspace/` 用于运行时中间文件
- `data/` 用于本地完整输出
- `demo/` 用于仓库附带样例

### 2.3 统一书目清单

`books.yaml` 是唯一书目注册表。目标字段包括：

- `book_prefix`
- `subject`
- `stage`
- `grade`
- `publisher`
- `source`

其中 `source` 表示教材 Markdown 相对于 `input_dir` 的路径。

## 3. 分发策略

### 3.1 GitHub 仓库

GitHub 仓库负责：

- 维护 pipeline 源码
- 维护总体文档与模块文档
- 附带最小 demo，帮助读者理解格式
- 提供面向 Hugging Face 的导出、下载、校验脚本

目标结构如下：

```text
K12-Dataset/
├── README.md
├── books.yaml
├── config/
│   ├── default.yaml
│   └── demo.yaml
├── src/
├── scripts/
│   ├── download_hf_data.py
│   ├── export_demo.py
│   └── validate_demo.py
├── demo/
│   ├── README.md
│   ├── manifest.json
│   ├── kg/
│   ├── benchmark/
│   └── sft_qa/
├── manifests/
│   ├── dataset_card_assets/
│   ├── snapshots.json
│   └── stats.json
├── input/          # gitignored
├── workspace/      # gitignored
└── data/           # gitignored
```

### 3.2 Hugging Face dataset repo

Hugging Face dataset repo 负责：

- 维护完整知识图谱
- 维护 benchmark 全量产物
- 维护 SFT QA 全量产物
- 维护版本 snapshot、统计和 checksums

目标结构如下：

```text
hf://tu-naaa/K12-Dataset-Full/
├── README.md
├── books.yaml
├── manifests/
│   ├── snapshot-v1.json
│   ├── stats.json
│   └── checksums.json
└── v1/
    ├── kg/
    │   ├── chapter_kg/
    │   ├── book_kg/
    │   ├── subject_stage_kg/
    │   ├── subject_kg/
    │   └── global_kg/
    ├── benchmark/
    │   ├── raw/
    │   ├── eval/
    │   └── metadata.json
    └── sft_qa/
        ├── train.jsonl
        ├── dev.jsonl
        ├── test.jsonl
        └── stats.json
```

## 4. 仓库与代码现状

这一节专门记录“当前代码已经这样做了”和“目标设计已经定了但仓库还没补齐”的区别。

### 4.1 当前代码已经落地的部分

- `src/utils/config.py` 已经实现统一配置加载逻辑
- `src/kg/` 已经收拢图谱主线的四个入口脚本
- `src/kg/merge_kg.py` 已经按 `book -> subject_stage -> subject -> global` 输出图谱
- `demo/` 已经在仓库中作为 GitHub 展示层落地

### 4.2 当前代码与目标设计仍未对齐的部分

下面这些是当前最重要的未对齐项：

1. 代码默认会去读 `config/default.yaml`，但仓库当前还没有提交 `config/` 目录。
2. 设计中约定了 `config/demo.yaml`、`scripts/download_hf_data.py`、`scripts/export_demo.py`、`scripts/validate_demo.py`、`manifests/`，但这些仍未落地。
3. `segmentation` 和 `build_graph` 的当前代码都依赖 `books.yaml` 中可用的 `source` 字段；但仓库里的 `books.yaml` 目前多数 `source` 还是空字符串，因而还不能直接端到端运行。
4. `pipeline.md` 里仍有“`config/` 与 `data/` 已完整落地”的旧表述，后续需要和本文件继续统一。

这些点不是设计冲突，而是“目标设计已确定，但实现尚未补齐”。

## 5. 图谱主线 Pipeline

### 5.1 主线流程

```text
input/*.md or input/**/*.md
          │
          ▼
1. segmentation
   -> workspace/segmentation/{book_prefix}/
          │
          ▼
2. build_graph
   -> data/chapter_kg/{book_prefix}/
          │
          ▼
3. merge_graph
   -> data/book_kg/
   -> data/subject_stage_kg/
   -> data/subject_kg/
   -> data/global_kg/
          │
          ▼
4. check_graph
   -> workspace/check_graph/
```

然后，下游模块再读取 `data/` 中的图谱：

- `exercise/`
- `benchmark/`
- `test_bench/`
- `sft_qa/`

### 5.2 数据流向

| 阶段 | 读取 | 写入 |
|------|------|------|
| segmentation | `books.yaml` + `input_dir` | `workspace/segmentation/{book_prefix}/` |
| build_graph | `workspace/segmentation/{book_prefix}/sections/*.md` | `data/chapter_kg/{book_prefix}/` |
| merge_graph | `data/chapter_kg/` + `sections_index.json` + `books.yaml` | `data/{book_kg,subject_stage_kg,subject_kg,global_kg}/` |
| check_graph | 合并后的图谱文件 | `workspace/check_graph/` |

## 6. `src/kg/` 模块说明

### 6.1 segmentation

入口脚本：[`src/kg/segment_textbooks.py`](./kg/segment_textbooks.py)

当前职责：

- 从 `books.yaml` 读取书目信息
- 通过 `source` 找到教材 Markdown
- 识别目录（TOC）或使用回退启发式
- 生成 `sections_index.json`
- 将教材切分为章节级 Markdown
- 将结果写入 `workspace/segmentation/{book_prefix}/`

核心输出：

- `workspace/segmentation/{book_prefix}/sections_index.json`
- `workspace/segmentation/{book_prefix}/sections/*.md`

当前代码特点：

- 已经内联了切分逻辑，不再依赖旧版“双脚本分步切分”说明
- `BookRecord` 使用的是 `publisher` 字段，不再是旧文档中的 `version`
- 端到端运行依赖 `books.yaml` 中有效的 `source`

### 6.2 build_graph

入口脚本：[`src/kg/extract_kg_from_textbook.py`](./kg/extract_kg_from_textbook.py)

当前职责：

- 读取切分后的 section Markdown
- 将 section 内容填入 prompt 模板
- 调用 `src/utils/llm_client.py`
- 将 LLM 输出规范化为 chapter 级 JSON 图谱
- 将原始 LLM 响应写入 `workspace/build_graph/{book_prefix}/`

核心输出：

- `data/chapter_kg/{book_prefix}/*.json`

当前代码特点：

- 当前使用 `prompt.txt`，Prompt 构造逻辑内联在脚本里
- 不存在旧文档里提到的 `prompt_builder.py`
- 节点类型前缀固定为：`cpt`、`skl`、`exp`、`exe`
- `tests_concept` / `tests_skill` 在 chapter 级图谱里以聚合边形式保存

### 6.3 merge_graph

入口脚本：[`src/kg/merge_kg.py`](./kg/merge_kg.py)

当前职责：

- 读取 `chapter_kg`
- 生成 `book_kg`
- 继续生成 `subject_stage_kg`
- 继续生成 `subject_kg`
- 最后生成 `global_kg`

合并顺序：

1. chapter -> book
2. book -> subject_stage
3. subject_stage -> subject
4. subject -> global

当前代码特点：

- 对 `Concept`、`Skill`、`Experiment`、`Exercise` 做去重
- `book_kg` 会补出结构节点：`Book`、`Chapter`、`Section`
- `book_kg` 会补出结构边：`is_part_of`、`appears_in`
- `global_kg` 是精简格式：
  - `nodes.json` 只保留 `id`、`label`、`name`
  - `edges.json` 只保留 `source`、`target`、`type`
- `tests_concept` / `tests_skill` 在写入 `global_kg` 时会从聚合边展开为普通二元边

### 6.4 check_graph

入口脚本：[`src/kg/check_cycles.py`](./kg/check_cycles.py)

当前职责：

- 对合并后的图谱做环检测
- 当前只检查两类边：
  - `is_a`
  - `prerequisites_for`

支持层级：

- `book`
- `subject_stage`
- `subject`
- `global`

输出位置：

- `workspace/check_graph/{level}_cycle_report.json`

## 7. 图谱数据层

### 7.1 产物层级

本仓库的图谱数据按四个层级组织：

#### chapter_kg

- 一个 section 或 chapter 对应一个 JSON 文件
- 保留最细粒度的 LLM 抽取结果
- 位于 `data/chapter_kg/{book_prefix}/`

#### book_kg

- 一本教材一个 JSON 文件
- 在 chapter 级结果上做去重、重编号和结构边补全
- 位于 `data/book_kg/{book_prefix}.json`

#### subject_stage_kg

- 同一学科、同一学段合并为一个 JSON 文件
- 位于 `data/subject_stage_kg/{subject}_{stage}.json`

#### subject_kg

- 同一学科跨学段合并为一个 JSON 文件
- 位于 `data/subject_kg/{subject}.json`

#### global_kg

- 全局精简图谱
- 使用两文件格式：
  - `data/global_kg/nodes.json`
  - `data/global_kg/edges.json`

### 7.2 节点类型

当前代码与样例中实际出现或支持的节点类型包括：

- `Concept`
- `Skill`
- `Experiment`
- `Exercise`
- `Book`
- `Chapter`
- `Section`

其中：

- 前四类是知识图谱主体节点
- 后三类是 `merge_kg.py` 在 `book_kg` 中补出的结构节点

### 7.3 关系类型

当前主线图谱中核心边类型包括：

- `relates_to`
- `prerequisites_for`
- `is_a`
- `verifies`
- `tests_concept`
- `tests_skill`
- `appears_in`
- `is_part_of`

其中：

- `relates_to`、`prerequisites_for`、`is_a`、`verifies` 是知识边
- `tests_concept`、`tests_skill` 是题目到知识点的聚合边
- `appears_in`、`is_part_of` 是 `merge_kg.py` 补出的结构边

### 7.4 chapter / book 级 JSON 格式

`chapter_kg` 与 `book_kg` 都使用统一容器格式：

```json
{
  "nodes": [...],
  "edges": [...]
}
```

其中：

- `nodes` 是节点数组
- `edges` 是边数组

节点的最小公共字段：

```json
{
  "id": "math_7a_rjb_cpt1",
  "label": "Concept",
  "name": "正数"
}
```

边的最小公共字段：

```json
{
  "source": "math_7a_rjb_cpt1",
  "target": "math_7a_rjb_cpt2",
  "type": "is_a"
}
```

不同边会按需要附带：

- `source_name`
- `source_stem`
- `target_name`
- `target_name_to_ids`
- `properties`

### 7.5 global_kg 精简格式

`global_kg` 与其它层级不同，不再包一层 `{nodes, edges}`，而是拆成两个平铺文件：

- `nodes.json`
- `edges.json`

`nodes.json` 中每项只有：

```json
{
  "id": "math_7a_rjb_cpt1",
  "label": "Concept",
  "name": "正数"
}
```

`edges.json` 中每项只有：

```json
{
  "source": "math_7a_rjb_cpt1",
  "target": "math_7a_rjb_cpt2",
  "type": "relates_to"
}
```

这个层级的目标是：

- 统一全局索引
- 降低下游任务读取成本
- 方便 benchmark / sft / demo 导出使用

## 8. 当前推荐阅读顺序

如果要快速理解当前项目，建议按下面顺序看：

1. [`README.md`](../README.md)
2. 本文档
3. [`src/kg/merge_kg.py`](./kg/merge_kg.py)
4. [`src/benchmark/bench.md`](./benchmark/bench.md)
5. [`src/sft_qa/sft.md`](./sft_qa/sft.md)

如果要直接顺着代码跑主线，建议按下面顺序看入口脚本：

1. [`src/kg/segment_textbooks.py`](./kg/segment_textbooks.py)
2. [`src/kg/extract_kg_from_textbook.py`](./kg/extract_kg_from_textbook.py)
3. [`src/kg/merge_kg.py`](./kg/merge_kg.py)
4. [`src/kg/check_cycles.py`](./kg/check_cycles.py)
