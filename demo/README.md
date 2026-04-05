# Demo 示例文件

## 目录结构

```text
demo/
├── manifest.json
├── kg/
│   └── math_7a_rjb.json
├── benchmark/
│   ├── task1_subtask1.jsonl
│   ├── task1_subtask2.jsonl
│   ├── task2_subtask1.jsonl
│   ├── task2_subtask2.jsonl
│   ├── task3.jsonl
│   ├── task4_subtask1.jsonl
│   ├── task4_subtask2.jsonl
│   ├── task5_subtask1.jsonl
│   └── task5_subtask2.jsonl
└── sft_qa/
    ├── math_primary_rjb_node.sample.jsonl
    └── math_primary_rjb_edge.sample.jsonl
```

## 文件说明

- `kg/math_7a_rjb.json`：一个书级知识图谱样例
    - 顶层包括nodes和edges
    - nodes包含6类节点：`Book`、`Chapter`、`Section`、`Concept`、`Skill`、`Experiment`（这个样例里没有）、`Exercise`
    - edges包含7类边：`appears_in`、`is_a`、`is_part_of`、`prerequisites_for`、`relates_to`、`tests_concept`、`tests_skill`
- `benchmark/*.jsonl`：一组裁剪后的 benchmark 子任务样例
    - task1：测试模型是否能定位到正确知识节点
        - subtask1：输入题目，选出题目考察的核心`Concept`或`Skill`
        - subtask2：输入`Concept`或`Skill`，选出对应的题目
    - task2：测试模型是否掌握了课程的先修逻辑
        - subtask1：输入`Concept`或`Skill`，选出前置集合
        - subtask2：输入`Concept`或`Skill`，选出**最直接**后置知识
    - task3：测试模型对相关联知识、迁移概念的理解
        - 输入`Concept`，选出与它**直接**相关的概念
    - task4：测试模型对“原理-现象”的因果联想
        - subtask1：输入`Concept`，选出能够验证它的实验
        - subtask2：输入实验名，选出可由该实验验证的`Concept`
    - task5：测试模型对教材结构、课程大纲的掌握
        - subtask1：输入`Concept` / `Skill` / `Experiment`，选出该知识点**最早**出现的教材章节
        - subtask2：输入章节或单元，选出哪些章节是它的基础
- `sft_qa/math_primary_rjb_node.sample.jsonl`：一个裁剪后的节点 QA 样例
    - 每条记录包含4个字段：`id`、`name`、`question`、`answer`
        - 针对`concept`：\box{核心定义} + \box{核心公式 / 关键结论 / 具体规则} + 1-2句解释（可选）+ 1个例子（可选）
        - 针对`skill`：\box{描述 / 步骤} + 补充说明好处 / 用途等（可选）+ 1个例子（可选）
- `sft_qa/math_primary_rjb_edge.sample.jsonl`：一个裁剪后的边 QA 样例
    - 每条记录包含4个字段：`id`、`relationship`、`question`、`answer`
        - 针对`is_a`：因为【A 的定义】、【B 的定义】，【A 满足 B 的定义】，所以【A 属于 B】
        - 针对`prerequisites_for`：因为【学习 B 需要的知识/能力】，而【A 提供这种知识/能力】，所以【在学习 B 之前需要先学习 A】
        - 针对`relates_to`：【A 和 B 的关系是什么】。【解释 A 】；【解释 B 】。
- `manifest.json`：demo 清单与元数据文件，其中路径都是相对于当前 `demo/` 目录的相对路径。
