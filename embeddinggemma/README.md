# EmbeddingGemma Fine-Tuned Model Inference Guide

Inference execution guide for fine-tuned EmbeddingGemma models using Sentence Transformers (supporting both text-to-text and multimodal text-to-image retrieval).

## 1. Environment Setup

Create and activate a virtual environment in the project root directory (`embeddinggemma`), then install the required dependencies:

```bash
# Create and activate virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Run all commands from the **project root directory** (`embeddinggemma`) because scripts reference local image assets under `./assets/`.

---

## 2. Running Inference Scripts

### Text Embeddings and Task Prompts (Base Model Inference)

- **Online / Standard Load**
  ```bash
  python app/inference_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline (Local Cache Only)**
  ```bash
  HF_HUB_OFFLINE=1 python app/inference_embeddinggemma_with_sentence_transformers_off.py
  ```

### Multimodal Embeddings & Working with Images (Base Model)

- **Online / Standard Load**
  ```bash
  python app/multimodal_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline / Local Images (Uses images from `./assets/`)**
  ```bash
  HF_HUB_OFFLINE=1 python app/multimodal_embeddinggemma_with_sentence_transformers_off.py
  ```

### Fine-Tuned Text-to-Text Retrieval

- **Online / Standard Load**
  ```bash
  python app/fine_tuning_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline (Local Cache Only)**
  ```bash
  HF_HUB_OFFLINE=1 python app/fine_tuning_embeddinggemma_with_sentence_transformers_off.py
  ```

### Multimodal Retrieval (Text-to-Image)

- **Online (Fetches remote image URLs)**
  ```bash
  python app/fine_tuning_multimodal_embeddinggemma_with_sentence_transformers_on.py
  ```

- **Offline / Local Images (Uses images from `./assets/`)**
  ```bash
  HF_HUB_OFFLINE=1 python app/fine_tuning_multimodal_embeddinggemma_with_sentence_transformers_off.py
  ```

### RAG with EmbeddingGemma 2 (Low-Spec CPU Edition)

Simple two-stage RAG over a sample corporate knowledge base: EmbeddingGemma 2 retrieves the best category → best document, then Gemma 4 E2B answers the question using only the retrieved document.

The script is tuned for low-spec, CPU-only PCs (tested on Intel N100 / 4 cores / 3.7 GB RAM / no GPU):

- **Retrieval**: `google/embeddinggemma-2` via Sentence Transformers, text-only (vision/audio encoders disabled).
- **Generation**: `litert-community/gemma-4-E2B-it-litert-lm` via **LiteRT-LM** on CPU, instead of the full transformers checkpoint (`google/gemma-4-E2B-it` needs ~10 GB RAM).
- The two models are **never loaded at the same time**: all retrieval runs first, the embedding model is freed, then the generation model is loaded.
- Tunables at the top of the script: `NUM_THREADS`, `MAX_NUM_TOKENS` (context size), `MAX_OUTPUT_TOKENS`.

**Dependency**: `litert-lm-api` (included in `requirements.txt`; or install it alone with `pip install litert-lm-api`).

- **Online / Standard Load** (downloads ~3 GB of models on first run)
  ```bash
  python app/RAG_with_EmbeddingGemma_on.py
  ```

- **Offline (Local Cache Only)** (run the online version once first so both models are cached)
  ```bash
  HF_HUB_OFFLINE=1 python app/RAG_with_EmbeddingGemma_off.py
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

**Dependencies**: `litert-lm-api`, `edgeparse`, `qdrant-edge-py` (included in `requirements.txt`).

Files created in the project root: `data/gemma4_technical_report.pdf` (document), `db_gemma/` (vector index), `.cache/` (LiteRT-LM embedding cache).

- **Online / Standard Load** (downloads the two `.litertlm` models and the PDF on first run)
  ```bash
  python app/EmbeddingGemma2_LiteRT_RAG_on.py
  ```

- **Offline (Local Cache Only)** (run the online version once first so the models and `data/gemma4_technical_report.pdf` exist locally)
  ```bash
  HF_HUB_OFFLINE=1 python app/EmbeddingGemma2_LiteRT_RAG_off.py
  ```

---

## 3. Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks for training, fine-tuning, and exploratory inference:

- `fine-tuning-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning EmbeddingGemma for text retrieval.
- `fine-tuning-multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning multimodal EmbeddingGemma with text-image pairs.
- `inference-embeddinggemma-with-sentence-transformers.ipynb`: Interactive text retrieval inference.
- `multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Interactive multimodal retrieval inference.
- `RAG_with_EmbeddingGemma.ipynb`: RAG with EmbeddingGemma 2 and Gemma 4 E2B (transformers, GPU recommended).
- `EmbeddingGemma2_LiteRT_RAG.ipynb`: On-device RAG with EmbeddingGemma 2 / Gemma 4 E2B on LiteRT-LM and Qdrant Edge.
