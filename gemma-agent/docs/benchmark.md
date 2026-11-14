# Benchmark: EmbeddingGemma Scripts on a Low-Spec CPU-Only PC

This document reports the execution time and memory usage of the inference scripts in `apps/`, measured on a low-spec, CPU-only PC with models cached locally.

## 1. Test Environment

| Item | Value |
|---|---|
| CPU | Intel N100 (4 cores, 4 threads, AVX2, no AVX-512) |
| RAM | 3.7 GiB (+ 1.0 GiB swap) |
| GPU | None |
| OS | Ubuntu 24.04.3 LTS on WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| Python | 3.12.3 |
| torch | 2.15.0 (CPU) |
| transformers | 5.22.2 |
| sentence-transformers | 6.2.0 |
| litert-lm-api | 0.19.1 |
| qdrant-edge-py | 0.9.0 |
| edgeparse | 0.3.2 |
| huggingface_hub | 2.4.0 |

## 2. Methodology

- Every script was run from `gemma-agent/` with the project virtual environment (`.venv`) using `uv run python ../apps/<script>.py`.
- **Total time** and **peak memory** (maximum resident set size, `MaxRSS`) were measured with `/usr/bin/time`.
- **Per-step timings** were derived by prefixing each line of stdout with a timestamp (`python -u`, unbuffered output).
- Models and assets were present in the local cache / `./assets/` directory (simulating typical steady-state execution).
- Timings on WSL2 vary with the OS page cache and background system load, so differences of roughly ±10–20 s between runs should not be over-interpreted.

## 3. Summary

| Script | Total time | Peak memory | Notes |
|---|---:|---:|---|
| `inference_embeddinggemma_with_sentence_transformers.py` | 69.8 s | 2.18 GiB | Text-only model |
| `multimodal_embeddinggemma_with_sentence_transformers.py` | 454.0 s | 2.77 GiB | Vision + text model |
| `fine_tuning_embeddinggemma_with_sentence_transformers.py` | 50.1 s | 2.30 GiB | Fine-tuned text model (inference only) |
| `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers.py` | 121.2 s | 2.89 GiB | Fine-tuned multimodal model (inference only) |
| `RAG_with_EmbeddingGemma.py` | 94.3 s | 2.18 GiB | Sentence Transformers + Gemma 4 E2B (LiteRT-LM), 4 questions |
| `EmbeddingGemma2_LiteRT_RAG.py` | 51.3 s | 1.46 GiB | LiteRT-LM only, 2 questions, index reused |

> Note: The `fine_tuning_*` scripts run inference using already fine-tuned models (`shogoorg/my-embeddinggemma-2-finetuned`, `shogoorg/my-multimodal-embeddinggemma-2-finetuned`). On the very first run without a local cache, additional time is spent downloading model weights.

## 4. Detailed Results

### 4.1 `inference_embeddinggemma_with_sentence_transformers.py`

Text-only EmbeddingGemma 2 (vision and audio encoders disabled, 271,002,624 parameters, float32).

| Step | Time |
|---|---:|
| Python start-up + model load | 58.6 s |
| Embed 3 words | 3.3 s |
| Similarity of 3 sentence pairs (no prompt) | 0.7 s |
| Similarity of 3 sentence pairs (`STS` prompt) | 0.6 s |
| Interpreter shutdown (after last output) | 6.4 s |
| **Total** | **69.8 s** |

- Once the model is loaded, encoding a short sentence pair takes only **~0.2 s**.
- **Model loading accounts for 70–85% of total time.**

### 4.2 `multimodal_embeddinggemma_with_sentence_transformers.py`

EmbeddingGemma 2 with the vision encoder (audio encoder disabled, 438,760,448 parameters, float32). Each similarity step encodes 1 query image and 4 candidates (1 image + 3 texts).

| Step | Time |
|---|---:|
| Python start-up + model load | 41.7 s |
| Cross-modal similarity, default image token budget | 73.4 s |
| Cross-modal similarity, `max_soft_tokens=1120` | 319.1 s |
| Interpreter shutdown | 19.8 s |
| **Total** | **454.0 s** |

- **Image encoding is the main bottleneck on CPU.** Two images take about 1 minute with the default token budget.
- With `max_soft_tokens=1120` (maximum budget), the same step takes **5–6 minutes (~5x slower)**.

### 4.3 `fine_tuning_embeddinggemma_with_sentence_transformers.py`

Fine-tuned text model, 1 query and 1 document.

| Step | Time |
|---|---:|
| Python start-up + model load + encode | 44.6 s |
| Interpreter shutdown | 5.5 s |
| **Total** | **50.1 s** |

- Similarity score: `0.8493`.
- Model loading accounts for almost all of the execution time.

### 4.4 `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers.py`

Fine-tuned multimodal model, 1 text query and 3 images.

| Step | Time |
|---|---:|
| Python start-up + model load + encode | 107.3 s |
| Interpreter shutdown | 13.9 s |
| **Total** | **121.2 s** |

- Similarities: `[0.7091, 0.6116, 0.6654]`.
- This script loads the model without `config_kwargs`, so the audio encoder is loaded as well. It has the **highest peak memory of all scripts (2.89 GiB)**, close to the 3.7 GiB limit of this PC.

### 4.5 `RAG_with_EmbeddingGemma.py`

Two-stage retrieval (category → document) with Sentence Transformers (text-only, float32), followed by answer generation with Gemma 4 E2B on the LiteRT-LM CPU backend. The two models are never held in memory at the same time.

**Per-step breakdown** (~107 s total)

| Step | Time |
|---|---:|
| Python start-up + EmbeddingGemma load | 56.9 s |
| Free embedding model + load Gemma 4 E2B | 5.5 s |

| Question | Retrieval | Generation | Total |
|---|---:|---:|---:|
| How many days of annual paid leave do I get? | 8.9 s | 9.6 s | 18.5 s |
| How do I reset my password? | 5.0 s | 6.8 s | 11.8 s |
| What are the reimbursable expense guidelines for business travel? | 1.9 s | 7.1 s | 9.0 s |
| Can I receive personal packages at the office? | 3.2 s | 2.2 s | 5.4 s |
| **Average** | **4.8 s** | **6.4 s** | **11.2 s** |

- The first retrieval is slower because of one-time warm-up work.
- Peak memory: **2.18 GiB**.

### 4.6 `EmbeddingGemma2_LiteRT_RAG.py`

RAG over the Gemma 4 technical report (45 chunks) with EmbeddingGemma 2 and Gemma 4 E2B, both on the LiteRT-LM CPU backend, with Qdrant Edge as the vector store.

**Per-step breakdown** (with index in `db_gemma/` reused)

| Step | Time |
|---|---:|
| Python start-up + resolve model paths | 1.1 s |
| Load embedding engine + open existing index | 2.8 s |
| Embed and retrieve 2 queries | 1.9 s (~0.9 s each) |
| Release embedding engine + load Gemma 4 E2B | 1.9 s |

| Question | Generation |
|---|---:|
| What architecture is Gemma 4 based on? | 18.9 s |
| What context length does Gemma 4 support? | 12.0 s |

- Total execution time: **51.3 s**; Peak memory: **1.46 GiB**.
- Answers are streamed, so the first words appear before generation finishes.

## 5. Key Findings

1. **Loading the Sentence Transformers (PyTorch) model is the dominant fixed cost: 38–59 s.** For text-only scripts, actual encoding takes well under 10 s. In contrast, LiteRT-LM loads its models in about 1–3 s.
2. **`EmbeddingGemma2_LiteRT_RAG.py` is the fastest and most memory-efficient end-to-end pipeline** (51 s, 1.46 GiB peak), even though it includes PDF retrieval and LLM generation.
3. **Image embedding on CPU is slow.** Two images take ~1 minute with the default token budget and 5–6 minutes with `max_soft_tokens=1120`. On low-spec PCs, keep the token budget small.
4. **Memory is the tightest constraint.** Multimodal scripts reach 2.75–2.89 GiB on a 3.7 GiB machine. Disabling unused encoders (`config_kwargs={"audio_config": None}`, or vision too for text-only use) noticeably reduces memory use.
5. **Per-question latency for RAG is about 5–20 s**, almost all of which is answer generation by Gemma 4 E2B.
6. **Interpreter shutdown takes 5–20 s** after the last output because large tensors are freed.

## 6. Recommendations for Low-Spec PCs

- For RAG, prefer **`EmbeddingGemma2_LiteRT_RAG.py`** (LiteRT-LM for both embedding and generation).
- Keep the model loaded across queries (e.g., interactive loop) to avoid paying the load cost repeatedly.
- For multimodal workloads, use a small `max_soft_tokens` value and disable the audio encoder when not needed.
- Close other applications before running multimodal scripts to avoid swapping.
- To reduce RAG generation time, lower `RETRIEVAL_LIMIT` or `CHUNK_SIZE`.
