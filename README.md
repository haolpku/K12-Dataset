# K12-Dataset (K12-GraphBench Pipeline)

A pipeline for **knowledge-graph construction** over K-12 textbooks and **downstream data-asset generation** (benchmark items and SFT QA pairs).

> 中文版说明见 [`README_zh.md`](README_zh.md).

---

## Modules

The repository is split into four modules by responsibility. They can be used independently; the recommended order is **kg → benchmark / sft_qa → eval**.

1. **Textbook Knowledge Graph (`src/kg/`)**
   Starting from the books registered in `books.yaml`: PDF or Markdown → section splitting → per-section KG extraction → hierarchical merging (book / subject_stage / subject / global) → exercise extraction and completion → quality checks on the merged graph.

2. **Benchmark Generation (`src/benchmark/`)**
   Produces structured evaluation tasks from graph nodes and relations.

3. **SFT Data Generation (`src/sft_qa/`)**
   Produces SFT QA pairs from graph nodes and relations and exports training-ready JSONL.

4. **Multiple-Choice Evaluation (`eval/`)**
   Runs the generated benchmark JSONL against an OpenAI-compatible API or a local vLLM endpoint, writing per-item predictions and a `summary.json`. Models and endpoints are configured via `eval/configs/models/*.yaml` together with a local `.env` file.

Shared logic lives in **`src/utils/`** (configuration parsing, LLM client, I/O, etc.). Format examples are provided under **`demo/`** (trimmed samples of the graph, benchmark, and SFT data).

---

## Quick Start

### Get the code

```bash
git clone <repository-url>
cd K12-Dataset
```

### Install dependencies

Install from the repository root:

```bash
pip install -r requirements.txt
```

If the graph pipeline is run on **PDF → Markdown** inputs, **MinerU** must additionally be installed and callable from the shell (the default command name is defined under `mineru.command` in `config/default.yaml`).

### Configure API keys and endpoints

The `kg` and `sft_qa` pipelines read environment variables from `config/`:

```bash
cp config/.env.example config/.env
# Edit config/.env: set OPENAI_API_KEY, OPENAI_BASE_URL, etc.
```

### Run the graph pipeline

1. In the repository root, maintain **`books.yaml`** and set `source_pdf` or `source_md` for at least one book entry.
2. From the repository root, run (replace `<YourBookPrefix>` with a real `book_prefix` from `books.yaml`):

```bash
python src/kg/run_pipeline.py \
  --config config/default.yaml \
  --filter-prefix <YourBookPrefix>
```

By default, **`data/`** stores final outputs (graph, exercises, etc.) and **`workspace/`** stores intermediate artifacts. The paths are centrally defined in `config/default.yaml` under `paths`.

```bash
python src/kg/run_pipeline.py --help
```

shows additional options such as `--limit` and `--skip-check`.

### Build Benchmark and SFT QA (after `data/` artifacts are available)

```bash
python src/benchmark/run_pipeline.py --help
python src/sft_qa/run_pipeline.py --help
```

### Evaluate on the benchmark (`eval/`)

```bash
cp eval/configs/.env.example eval/configs/.env
# Fill in OPENAI_BASE_URL, local vLLM address, etc., as needed.

chmod +x eval/run.sh   # if not already executable
./eval/run.sh <model-config-stem>
```

`<stem>` refers to **`eval/configs/models/<stem>.yaml`**. Example scripts for launching a local vLLM server can be found under **`eval/vllm_scripts/`**.

---

## Repository layout

```text
.
├── config/               # Default pipeline configuration (paths, LLM, merge)
├── demo/                 # Format samples (see demo/README.md)
├── eval/                 # Multiple-choice evaluation scripts and model YAMLs
├── src/kg|benchmark|sft_qa|utils/
├── workspace/            # Default intermediate artifacts
├── data/                 # Default final data outputs
├── books.yaml            # Book registry
└── requirements.txt
```

---

## Data distribution

The complete dataset and versioned snapshots will be distributed through an external data-hosting platform; the link will be updated at the time of official release.
