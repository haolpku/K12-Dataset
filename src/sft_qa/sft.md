# sft_qa 模块设计说明

## 1. 模块定位

`sft_qa/` 负责从学科学段级知识图谱中生成可用于 SFT 训练的问答数据。

该模块的目标是：

1. 直接调用 LLM 生成最终风格的 QA。
2. 输出格式和当前样例保持一致。
3. 将中间产物统一写入 `workspace/sft_qa/`。
4. 将 `tests_concept` / `tests_skill` / `Exercise` 这类模板型任务独立为规则生成，不走 LLM。

## 2. 当前目标

第一版先收敛为下面这个最小闭环。

### 2.1 输入

主输入：

- `data/subject_stage_kg/<subject_stage_key>.json`
  - 每个 JSON 都包含 `nodes` 和 `edges`
  - `nodes` 中的 `Concept` 和 `Skill` 用于生成 `node.jsonl`
  - `edges` 中的 `is_a`、`prerequisites_for`、`relates_to`、`verifies`、`tests_concept`、`tests_skill` 用于生成 `edge.jsonl`

附加输入：

- 习题数据文件
  - 当前暂未确定具体路径和格式
  - 后续接入后用于生成 `exercise.jsonl`

### 2.2 输出

#### `node.jsonl`

```json
{"id":"node_item_1","name":"concept | 秒","question":"秒是什么时间单位？","answer":"\\box{...} ..."}
```

字段含义：

- `id`：最终重排后的顺序编号
- `name`：概念名或技能名，格式为 `concept | {concept name}` 或 `skill | {skill name}`
- `question`：问题，LLM 生成
- `answer`：答案，LLM 生成

#### `edge.jsonl`

```json
{"id":"edge_item_1","relationship":"is_a | 毫米 -> 厘米","question":"为什么说毫米是厘米的更小单位？","answer":"因为 ... 所以 ..."}
```

字段含义：

- `id`：最终重排后的顺序编号
- `relationship`：关系类型和可读描述
- `question`：问题，LLM 生成；`tests_*` 类问题直接从数据转化
- `answer`：答案，LLM 生成；`tests_*` 类答案直接从数据转化

关系字段格式：

- `is_a | {source name} -> {target name}`
- `prerequisites_for | {source name} -> {target name}`
- `relates_to | {source name} <-> {target name}`
- `verifies | {source name} -> {target name}`
- `tests_concept`
- `tests_skill`

#### `exercise.jsonl`

```json
{"id":"exercise_item_1","stem":"...","question":"...？","answer":"..."}
```

字段含义：

- `id`：最终重排后的顺序编号
- `stem`：题目题干
- `question`：问题，默认可直接取 `stem`
- `answer`：答案，直接从数据转化

## 3. 总体流程

`sft_qa` 的主流程建议固定为三步。

### 3.1 抽取任务

从 `subject_stage_kg/<subject_stage_key>.json` 读取节点和边，并按任务类型拆开：

- `Concept` -> node 生成任务
- `Skill` -> node 生成任务
- `is_a` -> edge 生成任务
- `prerequisites_for` -> edge 生成任务
- `relates_to` -> edge 生成任务
- `verifies` -> edge 生成任务
- `tests_concept` -> 规则生成任务
- `tests_skill` -> 规则生成任务

### 3.2 生成 QA

生成逻辑分两类：

- LLM 生成
  - `Concept`
  - `Skill`
  - `is_a`
  - `prerequisites_for`
  - `relates_to`
  - `verifies`
- 规则生成
  - `tests_concept`
  - `tests_skill`
  - `exercise`

其中，LLM prompt 直接约束最终风格，例如：

- node 答案中的 `\\box{}`
- `is_a` 的“因为……所以……”结构
- `prerequisites_for` 的先修关系解释方式
- `verifies` 的验证关系解释方式

### 3.3 合并产物

将各 task 的局部结果合并为最终文件：

- `node.jsonl`
- `edge.jsonl`
- `exercise.jsonl`

合并阶段只负责：

- 拼接
- 重排 `id`
- 保证字段顺序稳定

不再调用模型。

## 4. 代码设计

`sft_qa` 是小模块，代码结构应尽量收敛，不要拆成太多文件。建议采用下面这套结构：

```text
src/sft_qa/
├── sft.md
├── generate_qa.py
├── tests_to_qa.py
├── merge_qa.py
└── prompts/
    ├── node.txt
    ├── is_a.txt
    ├── prerequisites_for.txt
    ├── relates_to.txt
    └── verifies.txt
```

### 4.1 `generate_qa.py`

主入口，负责所有 LLM 类任务。

职责：

- 读取 `subject_stage_kg/<subject_stage_key>.json`
- 选择需要处理的节点和边
- 构造 prompt
- 调用 `src/utils/llm_client.py`
- 解析模型输出
- 写出各 task 的中间 JSONL

建议这个脚本内部只保留少量辅助函数，不再继续拆出过多模块。优先内联处理：

- task 到 prompt 的映射
- 节点 / 边到 prompt 字段的转换
- 模型输出 JSON 的稳健解析

### 4.2 `tests_to_qa.py`

负责确定性任务：

- `tests_concept`
- `tests_skill`

职责：

- 从边记录中读取 `source_stem`
- 合并同题的多个 target
- 直接生成固定问句 QA

示例：

- `"这道题考察了什么概念？"`
- `"这道题考察了什么方法？"`

### 4.3 `merge_qa.py`

负责合并最终产物。

职责：

- 合并 node 类 JSONL 为 `node.jsonl`
- 合并 edge 类 JSONL 为 `edge.jsonl`
- 合并 exercise 类 JSONL 为 `exercise.jsonl`
- 统一重排 `id`

这一步不负责业务生成，只做结果整理。

### 4.4 `prompts/`

只保留当前确实需要的 prompt 文件：

- `node.txt`
- `is_a.txt`
- `prerequisites_for.txt`
- `relates_to.txt`
- `verifies.txt`

如果后面新增任务，再补对应 prompt 文件，不必一开始把所有可能类型都铺满。

## 5. Workspace 目录约定

建议所有中间产物统一写到：

- `workspace/sft_qa/<subject_stage_key>/`

建议结构：

```text
workspace/sft_qa/<subject_stage_key>/
├── raw/
│   ├── node_raw.txt
│   ├── is_a_raw.txt
│   ├── prerequisites_for_raw.txt
│   ├── relates_to_raw.txt
│   └── verifies_raw.txt
├── parts/
│   ├── node_concept.jsonl
│   ├── node_skill.jsonl
│   ├── edge_is_a.jsonl
│   ├── edge_prerequisites_for.jsonl
│   ├── edge_relates_to.jsonl
│   ├── edge_verifies.jsonl
│   ├── edge_tests_concept.jsonl
│   ├── edge_tests_skill.jsonl
│   └── exercise.jsonl
└── final/
    ├── node.jsonl
    ├── edge.jsonl
    └── exercise.jsonl
```

## 6. 实现原则

### 6.1 直接对齐最终样例

第一版优先对齐当前样例，不额外引入新的统一 schema。

### 6.2 小模块优先收敛

`sft_qa` 不需要拆成很多 Python 文件。只要 `generate_qa.py`、`tests_to_qa.py`、`merge_qa.py` 三个脚本能把职责分清楚，就足够了。

### 6.3 风格控制放在 prompt 里

风格要求直接写进生成 prompt，不单独设计默认的二次改写链路。

### 6.4 tests 类任务规则化

`tests_concept` / `tests_skill` 不走 LLM，直接规则生成。

## 7. 第一版实现顺序

建议按下面顺序落代码：

1. `generate_qa.py`
2. `tests_to_qa.py`
3. `merge_qa.py`
4. `prompts/` 下的必要模板

建议先跑通这几类任务：

- node: `Concept`、`Skill`
- edge: `is_a`、`prerequisites_for`、`relates_to`、`verifies`
- rules: `tests_concept`、`tests_skill`

`exercise` 可以作为下一步补充。

---

这份文档的定位是 `sft_qa` 的直接实现说明。后续写代码时，优先保证流程简单、输出稳定、格式和样例一致。
