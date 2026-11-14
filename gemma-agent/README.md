# gemma-agent

Simple ReAct agent
Agent generated with `agents-cli` version `1.13.0`

## Project Structure

```
gemma-agent/
├── app/         # Core agent code
│   ├── agent.py               # Main agent logic
│   ├── fast_api_app.py        # FastAPI Backend server
│   └── app_utils/             # App utilities and helpers
├── tests/                     # Unit, integration, and load tests
├── GEMINI.md                  # AI-assisted development guide
└── pyproject.toml             # Project dependencies
```

> 💡 **Tip:** Use [Antigravity CLI](https://antigravity.google/) for AI-assisted development - project context is pre-configured in `GEMINI.md`.

## Requirements

Before you begin, ensure you have:
- **uv**: Python package manager (used for all dependency management in this project) - [Install](https://docs.astral.sh/uv/getting-started/installation/) ([add packages](https://docs.astral.sh/uv/concepts/dependencies/) with `uv add <package>`)
- **agents-cli**: Agents CLI - Install with `uv tool install google-agents-cli`
- **Google Cloud SDK**: For GCP services - [Install](https://cloud.google.com/sdk/docs/install)


## Quick Start

Install `agents-cli` and its skills if not already installed:

```bash
uvx google-agents-cli setup
```

Install required packages:

```bash
agents-cli install
```

Test the agent with a local web server:

```bash
agents-cli playground
```

You can also use features from the [ADK](https://adk.dev/) CLI with `uv run adk`.

## Commands

| Command              | Description                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `agents-cli install` | Install dependencies using uv                                                         |
| `agents-cli playground` | Launch local development environment                                                  |
| `agents-cli lint`    | Run code quality checks                                                               |
| `agents-cli eval`    | Evaluate agent behavior (generate, grade, analyze, and more — see `agents-cli eval --help`) |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests                                                        || [A2A Inspector](https://github.com/a2aproject/a2a-inspector) | Launch A2A Protocol Inspector                                                        |

## 🛠️ Project Management

| Command | What It Does |
|---------|--------------|
| `agents-cli scaffold enhance` | Add CI/CD pipelines and Terraform infrastructure |
| `agents-cli infra cicd` | One-command setup of entire CI/CD pipeline + infrastructure |
| `agents-cli scaffold upgrade` | Auto-upgrade to latest version while preserving customizations |

---

## Development

Edit your agent logic in `app/agent.py` and test with `agents-cli playground` - it auto-reloads on save.

## Deployment

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```

To add CI/CD and Terraform, run `agents-cli scaffold enhance`.
To set up your production infrastructure, run `agents-cli infra cicd`.

## Observability

Built-in telemetry exports to Cloud Trace, BigQuery, and Cloud Logging.

## A2A Inspector

This agent supports the [A2A Protocol](https://a2a-protocol.org/). Use the [A2A Inspector](https://github.com/a2aproject/a2a-inspector) to test interoperability.
See the [A2A Inspector docs](https://github.com/a2aproject/a2a-inspector) for details.


## EmbeddingGemma Fine-Tuned Model Inference Guide

Inference execution guide for fine-tuned EmbeddingGemma models using Sentence Transformers (supporting both text-to-text and multimodal text-to-image retrieval).

## 1. Environment Setup

Install the required dependencies using `uv` or `agents-cli` from the project root directory (`gemma-agent`):

```bash
# Option 1: Install with uv (recommended)
uv sync

# Option 2: Install with agents-cli
agents-cli install

```

> **Note**: Run all commands from the **project root directory** (`gemma-agent`) because scripts reference local image assets under `./assets/`. If using `uv`, you can also prefix commands with `uv run` (e.g., `uv run python app/...`).

---

## 2. Running Inference Scripts

### Text Embeddings and Task Prompts (Base Model Inference)

- **Online / Standard Load**
  ```bash
  uv run python app/inference_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline (Local Cache Only)**
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/inference_embeddinggemma_with_sentence_transformers_off.py
  ```

### Multimodal Embeddings & Working with Images (Base Model)

- **Online / Standard Load**
  ```bash
  uv run python app/multimodal_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline / Local Images (Uses images from `./assets/`)**
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/multimodal_embeddinggemma_with_sentence_transformers_off.py
  ```

### Fine-Tuned Text-to-Text Retrieval

- **Online / Standard Load**
  ```bash
  uv run python app/fine_tuning_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline (Local Cache Only)**
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/fine_tuning_embeddinggemma_with_sentence_transformers_off.py
  ```

### Multimodal Retrieval (Text-to-Image)

- **Online (Fetches remote image URLs)**
  ```bash
  uv run python app/fine_tuning_multimodal_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline / Local Images (Uses images from `./assets/`)**
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/fine_tuning_multimodal_embeddinggemma_with_sentence_transformers_off.py
  ```

### RAG with EmbeddingGemma 2 (Low-Spec CPU Edition)

Simple two-stage RAG over a sample corporate knowledge base: EmbeddingGemma 2 retrieves the best category → best document, then Gemma 4 E2B answers the question using only the retrieved document.

The script is tuned for low-spec, CPU-only PCs (tested on Intel N100 / 4 cores / 3.7 GB RAM / no GPU):

- **Retrieval**: `google/embeddinggemma-2` via Sentence Transformers, text-only (vision/audio encoders disabled).
- **Generation**: `litert-community/gemma-4-E2B-it-litert-lm` via **LiteRT-LM** on CPU, instead of the full transformers checkpoint (`google/gemma-4-E2B-it` needs ~10 GB RAM).
- The two models are **never loaded at the same time**: all retrieval runs first, the embedding model is freed, then the generation model is loaded.
- Tunables at the top of the script: `NUM_THREADS`, `MAX_NUM_TOKENS` (context size), `MAX_OUTPUT_TOKENS`.

**Dependency**: `litert-lm-api` (included in `pyproject.toml`).

- **Online / Standard Load** (downloads ~3 GB of models on first run)
  ```bash
  uv run python app/RAG_with_EmbeddingGemma_on.py
  ```

- **Offline (Local Cache Only)** (run the online version once first so both models are cached)
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/RAG_with_EmbeddingGemma_off.py
  ```

> **Note**: On the first run, LiteRT-LM also builds an XNNPACK cache file next to the `.litertlm` model, so start-up is slow. Later runs are faster.
> If the PC runs out of memory, close other applications or lower `MAX_NUM_TOKENS` / `NUM_THREADS`.

### EmbeddingGemma 2 LiteRT: RAG on Edge (Low-Spec CPU Edition)

On-device RAG over the [Gemma 4 technical report](https://arxiv.org/abs/2607.02770): the PDF is parsed and chunked with **edgeparse**, embedded with **EmbeddingGemma 2 (LiteRT-LM)**, stored in a local **Qdrant Edge** shard, and answered by **Gemma 4 E2B (LiteRT-LM)** from the top retrieved chunks. Based on `notebooks/EmbeddingGemma2_LiteRT_RAG.ipynb`.

Low-spec CPU design (tested on Intel N100 / 4 cores / 3.7 GB RAM / no GPU):

- Both models run on the LiteRT-LM **CPU** backend.
- The vector index is persisted in `db_gemma/`. Once built, later runs **skip PDF parsing and embedding**. Delete `db_gemma/` to rebuild it.
- Chunks are embedded in small batches (`EMBED_BATCH_SIZE`) to keep peak memory low.
- The two models are **never loaded at the same time**: all queries are embedded and retrieved first, the embedding engine is closed, then Gemma is loaded.
- Tunables at the top of the script: `NUM_THREADS`, `EMBED_BATCH_SIZE`, `MAX_NUM_TOKENS`, `RETRIEVAL_LIMIT`. Edit `QUERIES` to ask your own questions.

**Dependencies**: `litert-lm-api`, `edgeparse`, `qdrant-edge-py` (included in `pyproject.toml`).

Files created in the project root: `data/gemma4_technical_report.pdf` (document), `db_gemma/` (vector index), `.cache/` (LiteRT-LM embedding cache).

- **Online / Standard Load** (downloads the two `.litertlm` models and the PDF on first run)
  ```bash
  uv run python app/EmbeddingGemma2_LiteRT_RAG_on.py
  ```

- **Offline (Local Cache Only)** (run the online version once first so the models and `data/gemma4_technical_report.pdf` exist locally)
  ```bash
  HF_HUB_OFFLINE=1 uv run python app/EmbeddingGemma2_LiteRT_RAG_off.py
  ```

---

> **Benchmark**: Execution time and memory usage of every script on a low-spec CPU-only PC are documented in [docs/benchmark.md](docs/benchmark.md).

---

## 3. Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks for training, fine-tuning, and exploratory inference:

- `fine-tuning-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning EmbeddingGemma for text retrieval.
- `fine-tuning-multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning multimodal EmbeddingGemma with text-image pairs.
- `inference-embeddinggemma-with-sentence-transformers.ipynb`: Interactive text retrieval inference.
- `multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Interactive multimodal retrieval inference.
- `RAG_with_EmbeddingGemma.ipynb`: RAG with EmbeddingGemma 2 and Gemma 4 E2B (transformers, GPU recommended).
- `EmbeddingGemma2_LiteRT_RAG.ipynb`: On-device RAG with EmbeddingGemma 2 / Gemma 4 E2B on LiteRT-LM and Qdrant Edge.
