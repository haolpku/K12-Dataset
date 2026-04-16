# K12-Dataset（K12-GraphBench Pipeline）

面向 K12 教材的 **知识图谱构建** 与 **下游数据资产**（Benchmark、SFT QA）生产流水线。

---

## Modules

仓库按职责分为四大模块，可单独使用，推荐顺序是 **kg → benchmark / sft_qa → eval**。

1. **教材图谱（`src/kg/`）**  
   从 `books.yaml` 注册的书目出发：PDF 或 Markdown → 章节切分 → 章节级 KG 抽取 → 多层级合并（book / subject_stage / subject / global）→ 课后题抽取与补全 → 合并图上的质量检测。

2. **Benchmark 生成（`src/benchmark/`）**  
   基于图谱节点与关系生成结构化评测任务。

3. **SFT 数据生成（`src/sft_qa/`）**  
   基于图谱节点与关系生成 SFT 问答对，导出训练用 JSONL。

4. **多选题评测（`eval/`）**  
   对已产出的 benchmark JSONL 调用 OpenAI 兼容 API 或本地 vLLM，写入逐条预测与 `summary.json`；模型与端点通过 `eval/configs/models/*.yaml` 与本地 `.env` 配置。

共享逻辑在 **`src/utils/`**（配置解析、LLM 客户端、IO 等）。格式样例见 **`demo/`**（含图谱、benchmark、SFT 的裁剪示例）。

---

## Quick Start

### 获取代码

```bash
git clone https://github.com/haolpku/K12-Dataset.git
cd K12-Dataset
```

### 安装依赖

请从仓库根目录安装依赖文件：

```bash
pip install -r requirements.txt
```

图谱主线若使用 **PDF → Markdown**，需额外安装并可在 shell 中调用 **MinerU**（默认命令名见 `config/default.yaml` 中 `mineru.command`）。

### 配置密钥与端点

主线 kg、sft_qa 等读取 `config/` 下的环境变量：

```bash
cp config/.env.example config/.env
# 编辑 config/.env：OPENAI_API_KEY、OPENAI_BASE_URL 等
```

### 运行图谱主线

1. 在仓库根目录维护 **`books.yaml`**，为至少一条书目配置书目所在路径 `source_pdf` 或 `source_md`。  
2. 在仓库根目录执行（将 `<YourBookPrefix>` 换成 `books.yaml` 里真实的 `book_prefix`）：

```bash
python src/kg/run_pipeline.py \
  --config config/default.yaml \
  --filter-prefix <YourBookPrefix>
```

默认 **`data/`** 为图谱与课后题等最终输出，**`workspace/`** 为中间产物，路径由 `config/default.yaml` 的 `paths` 统一定义。

```bash
python src/kg/run_pipeline.py --help
```

可查看 `--limit`、`--skip-check` 等选项。

### 构建 Benchmark 与 SFT QA（在已有 `data/` 产物之后）

```bash
python src/benchmark/run_pipeline.py --help
python src/sft_qa/run_pipeline.py --help
```

### 在 Benchmark 上进行评测（`eval/`）

```bash
cp eval/configs/.env.example eval/configs/.env
# 按需填写 OPENAI_BASE_URL、本地 vLLM 地址等

chmod +x eval/run.sh   # 若尚未可执行
./eval/run.sh <模型配置的 stem>
```

`<stem>` 对应 **`eval/configs/models/<stem>.yaml`**。需先启动本地 vLLM 时，可参考 **`eval/vllm_scripts/`** 中的示例脚本。

---

## 目录结构

```text
.
├── config/               # 管线默认配置（paths、LLM、merge）
├── demo/                 # 格式样例（说明见 demo/README.md）
├── eval/                 # 多选题评测脚本与模型 YAML
├── src/kg|benchmark|sft_qa|utils/
├── workspace/            # 默认中间产物
├── data/                 # 默认最终数据输出
├── books.yaml            # 书目注册表
└── requirements.txt
```

---

## 数据分发说明

完整数据集与版本快照见 Hugging Face：[`tunaaa126/K12-Dataset`](https://huggingface.co/datasets/tunaaa126/K12-Dataset)
