# benchmark设计

## 目的

考察 LLM 对 K12 课程结构的掌握，而不是考察它在看到知识图谱后的 in-context learning 能力。

因此 benchmark 的设计原则是：

1. 题目尽量直接由图谱结构生成，而不是先让 LLM 出题。
2. 正确答案和干扰项都优先从图结构中构造。
3. 如需后处理，LLM 只负责润色题干、筛掉有争议的干扰项，而不是决定题目的逻辑。

---


## 总体生成流程

当前 `src/benchmark/generate_benchmark.py` 的流程可以拆成四步：

1. 读取已经合并好的图数据目录。
2. 构建节点表 `NodeInfo` 和一组边索引。
3. 对每个 task 按图关系生成：
   - query
   - 正确答案 `answer`
   - 候选干扰项 `candidates`
4. 输出为 JSONL。

当前代码里真正参与 benchmark 生成的任务有：

- `task1_subtask1`
- `task1_subtask2`
- `task2_subtask1`
- `task2_subtask2`
- `task3`
- `task4_subtask1`
- `task4_subtask2`
- `task5_subtask1`
- `task5_subtask2`

注意：

- `task0` 目前没有实现。

---

## 图索引定义

`generate_benchmark.py` 会先把边整理成一组索引，后续所有 task 都依赖这些索引。

### 基础邻接

- `rel_adj[a]`
  - 与 `a` 存在 `relates_to` 的节点，按无向处理。
- `isa_adj[a]`
  - 与 `a` 存在 `is_a` 的节点，按无向处理。
- `prereq_in[a]`
  - 所有指向 `a` 的前置节点，即 `x ->(prerequisites_for) a` 中的 `x`。
- `prereq_out[a]`
  - 所有由 `a` 指向的后置节点，即 `a ->(prerequisites_for) x` 中的 `x`。

### 层级结构索引

- `parent_to_children_is_a[parent]`
  - `is_a` 下所有子节点。
- `child_to_parents_is_a[child]`
  - `is_a` 下所有父节点。
- `parent_to_children_prereq[parent]`
  - 注意名字容易误导。
  - 它实际上记录的是：所有把同一个节点当作“后置目标”的前置节点集合。
  - 也就是如果 `a -> b`、`c -> b`，那么在这里 `b` 会映射到 `{a, c}`。
  - 代码里用它来定义“共享同一个后置目标的 prerequisites_for 兄弟节点”。

### exercise / verifies / appears_in 索引

- `tests_concept_by_ex[exercise] -> concept集合`
- `tests_skill_by_ex[exercise] -> skill集合`
- `ex_by_concept[concept] -> exercise集合`
- `ex_by_skill[skill] -> exercise集合`
- `verifies_exp_to_concept[experiment] -> concept集合`
- `verifies_concept_to_exp[concept] -> experiment集合`
- `appears_src_to_sec[src] -> section集合`
- `part_of_parent[src] -> parent集合`

---

## 通用辅助定义

### 1-hop / 2-hop

代码里的 `two_hop_nodes(idx, node_id)` 不是只看 `is_a + relates_to`，而是把下面三类边都看成无向边：

- `relates_to`
- `is_a`
- `prerequisites_for`

因此：

- 1-hop = 在这三类边构成的无向图中与 query 直接相邻的节点
- 2-hop = 先走一步再走一步能到达的节点集合

### 兄弟节点

代码里“兄弟节点”的本质定义是：

- 如果两个节点共享同一个“父节点/目标节点”，那么它们是兄弟。
- 这里的“父节点/目标节点”既可以来自 `is_a`，也可以来自 `prerequisites_for`。

也就是说，如果：

- `A ->(is_a) P`
- `B ->(is_a) P`

则 `A` 和 `B` 是兄弟。

如果：

- `A ->(prerequisites_for) T`
- `B ->(prerequisites_for) T`

则 `A` 和 `B` 也被视为兄弟。

更进一步，如果：

- `A ->(is_a) P`
- `B ->(prerequisites_for) P`

那么只要它们共享同一个节点 `P`，也同样视为兄弟。

换句话说，兄弟关系是“共享目标节点”的定义，而不是“必须来自同一种边类型”。

### 干扰项最终去重

所有 task 最后都会经过 `make_candidates_from_ids()`：

1. 先把候选 ID 转成名称。
2. 去掉所有与正确答案重名的候选。
3. 去重。
4. 按 `sample_key` 做稳定哈希排序。

这意味着文档里描述干扰项时，默认都还要再经过这一层“去答案、去重、稳定排序”。

### 干扰项生成总原则

这一轮对 task1~4 的修改目标是：

1. 不再把干扰项主要压在“离正确答案极近的一圈”。
2. 先放宽候选来源，再用 LLM 过滤掉“不合理但看起来很近”的候选。
3. 尽量避免因为过滤过严导致题目最后没有干扰项。

统一采用三段式流程：

1. **图上扩圈取候选**
   - 先取高相关候选（1-hop 或兄弟节点）。
   - 如果数量不足，再扩到 2-hop。
   - 如果仍不足，再扩到 3-hop 或更大范围的同类型池。
2. **规则过滤**
   - 去掉正确答案本身。
   - 去掉和正确答案重名的候选。
   - 去掉按题意显然仍可判为正确的候选。
3. **LLM 过滤**
   - 让 LLM 判断候选项是否“对于该题来说明显不应算正确答案”。
   - 只保留被判为错误但又具有迷惑性的候选。

### LLM 过滤目标

LLM 过滤阶段不是为了“挑最像答案的干扰项”，而是为了删除两类坏候选：

1. **伪干扰项**
   - 实际上也可以算正确答案。
   - 例如和正确答案语义等价、上下位关系过近、教学顺序上本就可接受的节点。
2. **弱干扰项**
   - 和题目几乎没有关系。
   - 即使保留，也只会让题目变简单。

### 候选数量策略

为了避免过滤后没有候选项，建议每道题按下面的方式构造：

1. 图规则阶段先收集一批较宽的候选池。
   - 目标是得到至少 `8~20` 个原始候选。
2. LLM 过滤后，希望还能剩下至少 `3~8` 个候选。
3. 如果过滤后不足：
   - 回退到更远一层的图候选池
   - 或从同学科 / 同学段 / 同类型全局池补充

### 扩圈时的优先级

对于 task1~4，统一建议按下面顺序扩圈：

1. 第一圈：query 或 answer 的 1-hop 高相关节点
2. 第二圈：2-hop 节点
3. 第三圈：3-hop 节点
4. 兜底池：同类型、同学科或同学段的全局节点池

只有当前一层候选数量不足时，才进入下一层。

---

## Tasks

## 1. 针对 node 的

### Task0

当前未实现。

---

## 2. 针对 edge 的

## 干扰项构造总体规则【重要！】

- 对于每一题，干扰项来源有多层，每一层单独一个列表，列表内按sim_score排序
- 第n层的干扰项列表里需要去掉所有＜n层的干扰项列表里出现过的节点
- 如果某一层加入后，当前所有候选candidates数量大于等于10，就不需要下一层了，可以直接进入下一题

### sim_score（当前实现）

当前实现里，每一层内部的 `sim_score` 已改为 **embedding 语义相似度**，不再使用之前那种纯教材层级启发式分数。

使用方式是：

1. 先按图规则决定“这一层有哪些候选”
2. 再只在这一层内部，按 `sim_score` 从高到低排序

#### embedding 模型

- 当前使用本地 `fastembed`
- 模型为 `BAAI/bge-small-zh-v1.5`

#### 文本取法

- `candidate_text`
  - 普通节点取节点 `name`
  - `Exercise` 优先取 `stem`，没有再退回 `name`
- `question_text`
  - 直接使用该样本最终生成的 `question`
- `answer_text`
  - 对每个正确答案，普通节点取 `name`
  - `Exercise` 优先取 `stem`

#### 公式

记：

- `E(x)` = 文本 `x` 的 embedding
- `cos(a, b)` = cosine similarity
- `norm_cos(a, b) = (cos(a, b) + 1) / 2`

则对候选 `c`：

- `sim_question(c) = norm_cos(E(candidate_text), E(question_text))`
- `sim_answer(c) = max_i norm_cos(E(candidate_text), E(answer_text_i))`
- `sim_score(c) = (sim_question(c) + sim_answer(c)) / 2`

#### 排序规则

每一层内部当前按下面顺序排序：

1. `sim_score` 降序
2. `sim_question` 降序
3. `sim_answer` 降序
4. 候选名称字典序
5. 候选 ID

## Task 1: Knowledge Grounding

### 考察关系

- `tests_concept`
- `tests_skill`

### 目标

测试模型能否把题目和对应知识节点对齐。

---

### subtask1

#### 任务形式

输入一道 exercise，输出它考察的核心 `Concept` 或 `Skill`。

#### 设问方式

- concept 题：随机抽取“【X】这道题主要考察了什么核心概念？”、“要解决【X】这道题，主要需要用到哪些知识？”
- skill 题：随机抽取【X】这道题主要考察了什么核心方法？”、“要解决【X】这道题，主要需要用到哪些知识？”

#### 正确选项

- concept 题：
  - 取 `tests_concept_by_ex[exercise]`
- skill 题：
  - 取 `tests_skill_by_ex[exercise]`

#### 类型约束

- concept 样本只保留 `Concept`
- skill 样本只保留 `Skill`

#### 干扰项构造规则

这一题的目标是“exercise -> 核心 concept/skill”，因此干扰项需要看起来像是做这道题可能会用到的知识，但又不能真的属于标准答案。

把干扰项来源分成五层：

1. 第一层：正确答案的 2-hop 节点
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
2. 第二层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
3. 第三层：和正确答案属于同个chapter、和正确答案同类型的其他节点
4. 第四层：和正确答案属于同本book、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉所有正确答案的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）
3. 如果题目是 concept 题，只保留 `Concept`
4. 如果题目是 skill 题，只保留 `Skill`

最后做 LLM 过滤：

1. 让 LLM 判断候选是否“也可能被教师视为本题核心知识”
2. 删除这类有明显争议的候选
3. 保留剩余候选作为最终干扰项

---

### subtask2

#### 任务形式

输入一个 `Concept` 或 `Skill`，输出教材里哪些 exercise 考察了它。

#### 设问方式

- 随机抽取“以下哪些题目主要考察了{学科}中的【X】这个知识点？”、"围绕{学科}中的【X】这一知识点，教材里有哪些例题？"

#### 正确选项

- concept 题：
  - 取 `ex_by_concept[query]`
- skill 题：
  - 取 `ex_by_skill[query]`

#### 干扰项构造规则

这一题是“concept/skill -> exercise”，核心问题是找“和近邻知识绑定的其他 exercise”。

干扰节点分层如下：

1. 第一层：query 节点的 2-hop 节点对应的 exercise
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
2. 第二层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
3. 第三层：和正确答案属于同个chapter、和正确答案同类型的其他节点
4. 第四层：和正确答案属于同本book、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉所有query节点的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）对应的exercise
3. 只保留 `Exercise`

最后做 LLM 过滤：

1. 删除“其实也能合理视作在考察 query”的 exercise

---

## Task 2: Prerequisite Reasoning

### 考察关系

- `prerequisites_for`

### 目标

测试模型是否掌握课程中的先修逻辑。

---

### subtask1（前置）

#### 任务形式

输入一个 `Concept / Skill / Experiment`，输出它的前置知识集合。

#### 设问方式

- 随机抽取"要掌握{学科}中的【X】，应先具备哪些前置知识？"、"在学习{学科}中的【X】之前，需要学习哪些知识？" 

#### 正确选项

不是只取“直接前置”，而是取完整前置闭包。

代码通过 `collect_all_predecessors()` 沿 `prereq_in` 反复向前 DFS：

- 如果 `a -> b -> c`
- 当 query = `c`
- 正确答案是 `{a, b}`，不是只有 `{b}`

#### 干扰项构造规则

这一题问的是“前置知识集合”，所以干扰项应该优先来自两类区域：

1. query 的后置方向
2. 正确前置树外围的节点

相比单纯从 query 的近邻里取候选，这两类节点更容易形成“看起来相关，但其实不应算前置”的干扰项。

候选层次如下：

1. 第一层：query节点的整棵后置节点树
   - 直接后置节点、后置的后置节点、……
2. 第二层：前置节点树周围的 2-hop 节点
   - 以前置闭包中的每个节点为中心
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
   - 去掉完整前置闭包本身
3. 第三层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
4. 第四层：和正确答案属于同个chapter、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同本book、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
7. 第七层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉 query 本身
3. 去掉所有正确答案的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）
4. 只保留 `Concept / Skill / Experiment`

LLM 过滤重点：

1. 删除“作为教学顺序也合理属于前置知识”的候选
2. 删除“虽然不在图上前置，但教师依然可能认可”的候选
3. 优先保留位于后置方向、或位于前置树外围但不属于前置闭包的节点

---

### subtask2（后置）

#### 任务形式

输入一个 `Concept / Skill / Experiment`，输出它最直接的后置知识。

#### 设问方式

- 随机抽取"在学习了{学科}中的【X】之后，下一步**最适合**学习什么知识？"、"以下哪些知识是{学科}中的【X】的**最直接**后置知识？"、"掌握{学科}中的【X】后，通常会**马上**继续学习哪些内容？"

#### 正确选项

- 只取直接后置：
  - `prereq_out[query]`

#### 干扰项构造规则

这一题问的是“最直接后置”，所以干扰项最该优先考虑的是：

1. query 的完整前置闭包
2. 比正确答案更远层的后置方向节点

这两类节点既和 query 强相关，又不应被认作“最直接后置”。

候选层次如下：

1. 第一层：query节点的完整前置闭包
2. 第二层：正确答案的后置节点的后置节点
3. 第三层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
4. 第四层：和正确答案属于同个chapter、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同本book、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
7. 第七层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉 query 本身
3. 去掉所有正确答案的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）
4. 只保留 `Concept / Skill / Experiment`

LLM 过滤重点：

1. 删除“也可以被解释为下一步最适合学习”的候选
2. 删除那些虽然属于更远后置层，但仍可能被接受为“直接后置”的候选

---

## Task 3: Neighbor Recommendation

### 考察关系

- `relates_to`
- `is_a`

### 任务形式

输入一个 `Concept`，输出与它“直接相关”的 concept。

### 设问方式

- 随机抽取"以下哪些概念与{学科}中的【沸点】**直接相关**（包括分类关系或紧密关联）？"

### 正确选项

当前实现只取：

- `relates_to` 邻居
- `is_a` 邻居

也就是：

- `rel_adj[query] | isa_adj[query]`

注意：

- 这里**不把** `prerequisites_for` 邻居算进正确答案。

#### 干扰项构造规则

因为正确答案本身定义为“直接邻居”，所以干扰项最自然的来源就是“距离只差一层”的近邻外圈。

分层如下：

1. 第一层：正确答案的 2-hop 节点
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
2. 第二层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
3. 第三层：和正确答案属于同个chapter、和正确答案同类型的其他节点
4. 第四层：和正确答案属于同本book、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉 query 节点
3. 去掉所有正确答案的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）
4. 去掉可以通过两次 `relates_to` 从正确答案到达的节点
5. 只保留 `Concept`

LLM 过滤重点：

1. 删除“虽然是 2-hop / 3-hop，但教师仍可能认为是直接相关”的候选
2. 保留那些和 query 主题接近、但明显不属于直接邻居的 concept

---

## Task 4: 实验证据链

### 考察关系

- `verifies`

### 目标

测试模型是否能建立“实验 - 原理”对应关系。

---

### subtask1（concept -> experiment）

#### 任务形式

输入一个 concept，输出哪些 experiment 验证了它。

#### 设问方式

- 随机抽取"围绕{学科}中的【X】，教材安排了哪些验证实验？"、"教材中哪些实验可以验证{学科}中的【X】？"

#### 正确选项

- `verifies_concept_to_exp[concept]`

#### 干扰项构造规则

为了减少“其实也能验证该概念”的争议，建议 concept -> experiment 的干扰项优先来自更远一层的 concept 邻域，而不是只盯最近邻。

候选层次如下：

1. 第一层：query 节点的 2-hop 节点对应的 experiment
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
2. 第二层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
3. 第三层：和正确答案属于同个chapter、和正确答案同类型的其他节点
4. 第四层：和正确答案属于同本book、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉所有query节点的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）对应的experiment
3. 只保留 `Experiment`

LLM 过滤重点：

1. 删除“教师也可能接受为验证该 concept 的实验”
2. 保留那些和 query concept 处于相近知识域、但验证目标不同的 experiment

---

### subtask2（experiment -> concept）

#### 任务形式

输入一个 experiment，输出它验证了哪些 concept。

#### 设问方式

- 随机抽取"以下哪些概念可由{学科}中的【X】实验验证？"、"通过{学科}中的【X】实验，可以支持哪些核心概念？"、"{学科}中的【X】实验，在教材中被用来验证什么原理？"

#### 正确选项

- `verifies_exp_to_concept[experiment]`

#### 干扰项构造规则

这一题和 Task1 subtask1 很像，但因为 query 是 experiment，正确答案往往比较集中，所以更需要往外扩圈。

候选层次如下：

1. 第一层：正确答案的 2-hop 节点
   - 底层 hop 图采用 `relates_to + is_a + prerequisites_for` 的无向图
2. 第二层：第一层中的concept节点对应的experiment验证的其他concept
3. 第三层：和正确答案属于同个section、和正确答案同类型的其他节点（小学数学没有这一层）
4. 第四层：和正确答案属于同个chapter、和正确答案同类型的其他节点
5. 第五层：和正确答案属于同本book、和正确答案同类型的其他节点
6. 第六层：和正确答案属于同个subject_stage、和正确答案同类型的其他节点
7. 第七层：和正确答案属于同个subject、和正确答案同类型的其他节点

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 去掉所有正确答案的 1-hop 高相关节点（包括relates_to、is_a、prerequisites_for，无论边的方向）
3. 如果题目是 concept 题，只保留 `Concept`
4. 如果题目是 skill 题，只保留 `Skill`

LLM 过滤重点：

1. 删除“也可以被该实验合理支持”的 concept
2. 删除过于宽泛的上位概念
3. 保留那些与实验主题相近、但验证对象明显不同的 concept

---

## Task 5: 跨章节索引 / 章节依赖

### subtask1：首次出现位置

#### 考察关系

- `appears_in`
- `is_part_of`

#### 任务形式

输入一个 `Concept / Skill / Experiment`，输出它出现在教材的哪个章节位置。

#### 设问方式

- 随机抽取"{学科}中的【X】这一知识点**最早**出现在教材的哪个章节？"、"学生**第一次**学习{学科}中的【X】是在以下哪个章节？"

#### 正确选项

当前 `task5 subtask1` 的正确答案不能直接信任旧 JSONL 里的 `id / meta.source_id / answer`，而是要按下面流程重建：

1. 从题面里抽出 query 名称
   - 也就是 `【...】` 中的字符串。
2. 用这个 query 名称去 `global_kg` 的 `Concept / Skill / Experiment` 节点里重新匹配真实节点。
3. 用真实节点的最新 `appears_in` 边重算“首次出现位置”。
4. 用真实节点 ID 重写：
   - `id = task5::subtask1::<new_source_id>`
   - `meta.source_id = <new_source_id>`
   - `meta.source_label = 真实节点标签`
5. 如果多条旧题映射到同一个新节点，则按新 `id` 去重，只保留一条。

这里的“首次出现”不是任意挑一个，而是按教材顺序排序后取最早位置。

教材顺序固定为：

- `1a`
- `1b`
- `2a`
- `2b`
- `3a`
- `3b`
- `4a`
- `4b`
- `5a`
- `5b`
- `6a`
- `6b`
- `7a`
- `7b`
- `8a`
- `8b`
- `9a`
- `9`
- `9b`
- `bx1`
- `bx2`
- `bx3`
- `xzxbx1`
- `xzxbx2`
- `xzxbx3`

在同一本书内，再按 `chapter` 序数、`section` 序数升序比较。（注意不是按字典序，是按数字的大小顺序，就是1、2、3、……、9、10、11、……这样）

#### 答案粒度

- 小学数学：
  - 如果某个知识点在小学数学部分是直接 `appears_in -> Chapter`，则答案粒度为 `Chapter`。
- 其他情况：
  - 答案粒度为 `Section`。

换句话说：

- 小学数学保留 `Chapter` 级答案。
- 其他学科、其他学段仍然保留 `Section` 级答案。

#### 多次出现时的 answer / candidates 处理

一个 query 可能在多个位置出现。

当前规则是：

1. 把所有出现位置按上述教材顺序排序。
2. 最早那个位置进入 `answer`。
3. 其余出现位置先进入现有 `candidates` 的前部补充区。

这里要注意：

- 如果答案粒度是 `Section`，那么“其余出现位置”也是其余 `Section`。
- 如果答案粒度是 `Chapter`，那么“其余出现位置”是其余 `Chapter`。

#### 干扰项构造规则

1. 第一层：query节点appears_in的其他section（小学数学是chapter）
2. 第二层：和正确答案属于同个chapter的其他section（小学数学没有这一层）
3. 第三层：和正确答案属于同本book、比正确答案序数小的其他section（小学数学是chapter）
4. 第四层：和正确答案属于同个subject_stage、比正确答案序数小的其他section（小学数学是chapter）
5. 第五层：和正确答案属于同个subject、比正确答案序数小的其他section（小学数学是chapter）
6. 第六层：和正确答案属于同本book、比正确答案序数大的其他section（小学数学是chapter，只有前几层总数不足3个时才需要这一层）

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 如果query节点属于小学数学，只保留 `Chapter`；否则，只保留 `Section`

---

### subtask2：章节基础依赖

#### 考察关系

- `leads_to`

#### 任务形式

输入一个 `Chapter`，输出哪些 `Chapter` 是它的基础。

也就是说，query 是一个 chapter，正确答案是所有通过 `leads_to` 边指向该 chapter 的 source chapters。

如果：

- `A ->(leads_to) B`

那么在本题里：

- query = `B`
- `A` 属于正确答案

#### 设问方式

- "以下哪些章节的知识是{学段}{学科}【X】的基础？"

#### 正确选项

- 取所有满足 `source ->(leads_to) query_chapter` 的 `source chapter`

也就是：

- `leads_to_in[query_chapter]`

如果某个 chapter 没有任何 `leads_to` 入边，则不生成该题。

#### 干扰项构造规则

1. 第一层：query节点的前一个chapter、query节点的前两个chapter、query节点的前三个chapter、每个正确答案的前一个chapter、每个正确答案的前两个chapter
  - 可以通过chapter的序号来判断前1/2/3个，如果同本书的到头了就进入取上一本书的（书的顺序同task5 subtask1中的定义） 
2. 第二层：同学科同学段的其他chapter（**只有第一层不足3个时才需要这一层**）

每一层都需要做规则过滤：

1. 去掉所有正确答案
2. 只保留 `Chapter`

---

## 对应代码位置

主要逻辑都在：

- `/home/tuna/K12-Dataset/src/benchmark/generate_benchmark.py`

关键函数：

- `build_edge_indexes()`
- `two_hop_nodes()`
- `task1_subtask1_distractors()`
- `gen_task1()`
- `gen_task2()`
- `gen_task3()`
- `gen_task4()`
- `gen_task5()`
