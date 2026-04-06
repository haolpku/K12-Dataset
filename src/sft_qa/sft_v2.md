# sft_qa v2 细化设计

## 1. 文档定位

这份文档承接此前写在 `src/docs.md` 中、但不再适合放在总览文档里的 `sft_qa` 细节设计。

它的定位是：

- 补充 [`sft.md`](./sft.md) 中偏简略的模块说明
- 记录 `sft_qa` 的目标结构、样本 schema、任务拆分与 CLI 约定
- 在需要时明确指出“当前代码已经这样做了”和“目标设计还没完全落地”的差别

如果本文件与当前代码冲突，以当前代码为准。

## 2. 模块定位

`sft_qa/` 负责从知识图谱中生成可用于 SFT 训练的问答数据。

从当前代码看，它已经采用“LLM 生成 + 规则生成 + 合并”的三段式结构：

- [`generate_qa.py`](./generate_qa.py)
  - 负责 `Concept` / `Skill` / `is_a` / `prerequisites_for` / `relates_to` / `verifies`
- [`tests_to_qa.py`](./tests_to_qa.py)
  - 负责 `tests_concept` / `tests_skill`
- [`merge_qa.py`](./merge_qa.py)
  - 负责把各类 `parts/*.jsonl` 合并为最终产物

从设计目标上，`sft_qa` 的职责是：

1. 直接从图谱节点和关系生成最终风格的 QA
2. 对不同关系类型使用清晰分工的 prompt
3. 对模板型任务使用确定性规则生成，避免无谓调用 LLM
4. 把中间产物统一写入 `workspace/sft_qa/`

## 3. 输入与输出

### 3.1 当前主输入

当前代码默认读取：

- `data/subject_stage_kg/{subject_stage}.json`

也就是一个“学科 × 学段”级别的合并图谱。

从当前实现看：

- `generate_qa.py` 默认从 `subject_stage_kg/<subject_stage>.json` 读取节点和边
- `tests_to_qa.py` 也读取同一个输入文件

这意味着当前 `sft_qa` 的主输入粒度是 `subject_stage_kg`，而不是 `global_kg`。

### 3.2 当前中间产物目录

当前代码约定的 workspace 目录是：

```text
workspace/sft_qa/<subject_stage>/
├── raw/
│   ├── node_concept_raw.txt
│   ├── node_skill_raw.txt
│   ├── edge_is_a_raw.txt
│   ├── edge_prerequisites_for_raw.txt
│   ├── edge_relates_to_raw.txt
│   └── edge_verifies_raw.txt
├── parts/
│   ├── node_concept.jsonl
│   ├── node_skill.jsonl
│   ├── edge_is_a.jsonl
│   ├── edge_prerequisites_for.jsonl
│   ├── edge_relates_to.jsonl
│   ├── edge_verifies.jsonl
│   ├── edge_tests_concept.jsonl
│   ├── edge_tests_skill.jsonl
│   └── exercise.jsonl            # 当前可选，默认可能不存在
└── final/
    ├── node.jsonl
    ├── edge.jsonl
    └── exercise.jsonl
```

其中：

- `raw/` 保存 LLM 原始输出或错误信息
- `parts/` 保存 task 级中间结果
- `final/` 保存最终合并结果

### 3.3 最终产物

当前 `merge_qa.py` 最终写出三个文件：

- `final/node.jsonl`
- `final/edge.jsonl`
- `final/exercise.jsonl`

注意：

- `exercise.jsonl` 目前在合并逻辑里是预留的
- 仓库当前并没有一个独立的 exercise QA 生成脚本真正把它填满

## 4. 任务拆分

### 4.1 当前任务注册表

当前 [`generate_qa.py`](./generate_qa.py) 中的 `TASK_SPECS` 已经把 LLM 类任务固定为：

- `node_concept`
- `node_skill`
- `edge_is_a`
- `edge_prerequisites_for`
- `edge_relates_to`
- `edge_verifies`

当前 [`tests_to_qa.py`](./tests_to_qa.py) 负责的规则类任务是：

- `tests_concept`
- `tests_skill`

### 4.2 推荐任务分层

从职责上看，`sft_qa` 最清晰的分层应该是：

#### 节点任务

- `Concept`
- `Skill`

#### 关系任务

- `is_a`
- `prerequisites_for`
- `relates_to`
- `verifies`

#### 模板型任务

- `tests_concept`
- `tests_skill`

#### 可选扩展任务

- `Exercise`

也就是说，当前 v2 设计里，“Exercise QA” 仍然属于可以接入、但尚未完全落地的部分。

## 5. 统一样本 schema

### 5.1 parts 层：node 类

当前 `generate_qa.py` 生成的 node 类 `parts` 记录格式为：

```json
{
  "task": "node_concept",
  "source_id": "math_primaryschool_cpt123",
  "name": "concept | 秒",
  "question": "秒是什么时间单位？",
  "answer": "\\box{...} ..."
}
```

字段说明：

- `task`：任务名，如 `node_concept`
- `source_id`：原图谱节点 ID
- `name`：带任务前缀的名称，当前代码会写成 `concept | X` 或 `skill | X`
- `question`：模型生成的问题
- `answer`：模型生成的答案

### 5.2 parts 层：edge 类

当前 `generate_qa.py` 或 `tests_to_qa.py` 生成的 edge 类 `parts` 记录格式为：

```json
{
  "task": "edge_is_a",
  "source_id": "is_a::math_1_cpt1::math_1_cpt2",
  "relationship": "is_a | 毫米 -> 厘米",
  "question": "为什么说毫米是厘米的更小单位？",
  "answer": "因为 ... 所以 ..."
}
```

字段说明：

- `task`：任务名，如 `edge_is_a`
- `source_id`：边的稳定来源标识
- `relationship`：关系类型和可读描述
- `question`：问题
- `answer`：答案

当前关系字符串格式：

- `is_a | {source_name} -> {target_name}`
- `prerequisites_for | {source_name} -> {target_name}`
- `relates_to | {source_name} <-> {target_name}`
- `verifies | {source_name} -> {target_name}`

对 `tests_concept` / `tests_skill`，当前代码也会写入 `relationship` 字段，但来源是规则模板而不是 LLM。

### 5.3 final 层：node.jsonl

当前 `merge_qa.py` 输出的 `node.jsonl` 结构为：

```json
{
  "id": "node_item_1",
  "name": "concept | 秒",
  "question": "秒是什么时间单位？",
  "answer": "\\box{...} ..."
}
```

注意：

- `name` 目前保留了 `concept |` / `skill |` 前缀
- 这一点和某些更早的设计草稿不同，属于当前代码行为

### 5.4 final 层：edge.jsonl

当前 `merge_qa.py` 输出的 `edge.jsonl` 结构为：

```json
{
  "id": "edge_item_1",
  "relationship": "is_a | 毫米 -> 厘米",
  "question": "为什么说毫米是厘米的更小单位？",
  "answer": "因为 ... 所以 ..."
}
```

### 5.5 final 层：exercise.jsonl

当前 `merge_qa.py` 预留的 exercise 结构为：

```json
{
  "id": "exercise_item_1",
  "stem": "...",
  "question": "...",
  "answer": "..."
}
```

但需要强调：

- 这是合并层已经预留好的 schema
- 当前仓库里尚未完整落地对应的 exercise QA 生成流程

## 6. Prompt 设计

### 6.1 当前 prompt 文件

当前仓库中真实存在的 prompt 文件是：

- [`prompts/concept.txt`](./prompts/concept.txt)
- [`prompts/skill.txt`](./prompts/skill.txt)
- [`prompts/is_a.txt`](./prompts/is_a.txt)
- [`prompts/prerequisites_for.txt`](./prompts/prerequisites_for.txt)
- [`prompts/relates_to.txt`](./prompts/relates_to.txt)
- [`prompts/verifies.txt`](./prompts/verifies.txt)

这说明当前 `sft_qa` 已经不是“一个统一 node prompt”，而是明确区分：

- `Concept`
- `Skill`
- 不同 edge 类型

### 6.2 Prompt 字段映射

从当前代码看，prompt 渲染分两类：

#### node 类

`generate_qa.py` 会注入：

- `label`
- `name`
- `properties_json`

#### edge 类

`generate_qa.py` 会注入：

- `type`
- `source_name`
- `target_name`
- `properties_json`

### 6.3 当前实现细节

当前 prompt 管线还有几个值得记下的实现点：

1. 支持 `--num-samples`
   - 单个输入可以生成多条 QA
2. 默认优先使用 OpenAI Structured Outputs
   - 除非传 `--disable-json-schema`
3. 原始输出会被完整追加到 `raw/*.txt`
4. 即使解析失败，也会把错误信息记到 raw 文件里

## 7. CLI 设计

### 7.1 generate_qa.py

当前 CLI 重点参数：

- `--subject-stage`
- `--input-json`
- `--workspace-dir`
- `--tasks`
- `--limit`
- `--resume`
- `--model`
- `--temperature`
- `--max-tokens`
- `--num-samples`
- `--disable-json-schema`

职责：

- 读取 subject_stage 图谱
- 选择指定任务
- 调用 LLM
- 写出 `parts/*.jsonl`
- 写出 `raw/*_raw.txt`

### 7.2 tests_to_qa.py

当前 CLI 重点参数：

- `--subject-stage`
- `--input-json`
- `--workspace-dir`

职责：

- 从图谱中读取 `tests_concept` / `tests_skill`
- 合并同题多目标答案
- 写出：
  - `edge_tests_concept.jsonl`
  - `edge_tests_skill.jsonl`

### 7.3 merge_qa.py

当前 CLI 重点参数：

- `--subject-stage`
- `--workspace-dir`

职责：

- 合并 `parts/` 下的 node、edge、exercise 结果
- 重排最终 `id`
- 写出 `final/`

## 8. 合并逻辑

### 8.1 node 合并

当前 `merge_qa.py` 会按固定顺序拼接：

1. `node_concept.jsonl`
2. `node_skill.jsonl`

然后依次重排为：

- `node_item_1`
- `node_item_2`
- ...

### 8.2 edge 合并

当前 `merge_qa.py` 会按固定顺序拼接：

1. `edge_is_a.jsonl`
2. `edge_prerequisites_for.jsonl`
3. `edge_relates_to.jsonl`
4. `edge_verifies.jsonl`
5. `edge_tests_concept.jsonl`
6. `edge_tests_skill.jsonl`

然后依次重排为：

- `edge_item_1`
- `edge_item_2`
- ...

### 8.3 exercise 合并

当前 `merge_qa.py` 会尝试读取：

- `exercise.jsonl`

如果不存在，就输出空文件或空结果。

## 9. 与当前 `sft.md` 的关系

可以把两份文档理解成：

- [`sft.md`](./sft.md)
  - 偏“模块说明版”
  - 适合快速读懂 `sft_qa` 是干什么的
- [`sft_v2.md`](./sft_v2.md)
  - 偏“细化设计版”
  - 适合承接原本写在 `docs.md` 里的细节

## 10. 当前代码与 v2 目标的差异

下面这些点值得单独标出来：

1. `sft.md` 里写到的 `exercise.jsonl` 生成流程，在当前代码中还没有真正闭环。
2. 当前代码实际存在的 prompt 文件是 `concept.txt` 和 `skill.txt`，不是更早草稿里笼统的 `node.txt`。
3. 当前 final `node.jsonl` 的 `name` 字段保留了 `concept |` / `skill |` 前缀；如果未来想和 demo 样例完全一致，可能还需要再统一一次 schema。
4. 当前主输入仍然是 `subject_stage_kg`，而不是更高层的 `global_kg`。
5. 当前 `tests_to_qa.py` 已经从设计草稿中的 `tests_to_qa_jsonl.py` 收敛成了更短的脚本实现。
