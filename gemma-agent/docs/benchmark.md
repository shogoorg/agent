# Benchmark: EmbeddingGemma Scripts on a Low-Spec CPU-Only PC

This document reports the execution time and memory usage of every script in `app/`, measured on a low-spec, CPU-only PC.

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

- Every script was run from the project root (`embeddinggemma/`) with the project virtual environment (`.venv`).
- **Total time** and **peak memory** (maximum resident set size, `MaxRSS`) were measured with `/usr/bin/time`.
- **Per-step timings** were derived by prefixing each line of the script's stdout with a timestamp (`python -u`, unbuffered output). The time between two consecutive printed lines is attributed to the work done in between.
- **Offline** (`*_off.py`) scripts were run with `HF_HUB_OFFLINE=1`. **Online** (`*_on.py`) scripts were run with network access.
- No scripts were modified for measurement.
- Unless stated otherwise, all models were already in the local Hugging Face cache, so "online" runs only perform Hub metadata checks (and, for the multimodal script, download two small sample images).
- Each configuration was run **once**. Timings on WSL2 vary with the OS page cache (a model file read recently is loaded faster) and with network conditions, so differences of roughly ±20 s between runs of the same script should not be over-interpreted.

## 3. Summary

| Script | Mode | Total time | Peak memory | Notes |
|---|---|---:|---:|---|
| `inference_embeddinggemma_with_sentence_transformers` | off | 69.8 s | 2.18 GiB | Text-only model |
| | on | 57.4 s | 2.30 GiB | |
| `multimodal_embeddinggemma_with_sentence_transformers` | off | 454.0 s | 2.77 GiB | Vision + text model |
| | on | 499.0 s | 2.86 GiB | |
| `fine_tuning_embeddinggemma_with_sentence_transformers` | off | 50.1 s | 2.30 GiB | Fine-tuned text model (inference only) |
| | on | 43.5 s | 2.30 GiB | |
| `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers` | off | 121.2 s | 2.89 GiB | Fine-tuned multimodal model (inference only) |
| | on | 142.2 s | 2.75 GiB | |
| `RAG_with_EmbeddingGemma` | off | 94.3 s | 2.18 GiB | Sentence Transformers + Gemma 4 E2B (LiteRT-LM), 4 questions |
| | on (first run) | 896.3 s | 2.20 GiB | Includes re-downloading Gemma 4 E2B (~2.6 GB) |
| `EmbeddingGemma2_LiteRT_RAG` | off | 51.3 s | 1.46 GiB | LiteRT-LM only, 2 questions, index reused |
| | on (first run) | 329.7 s | 1.53 GiB | Includes downloading the embedding model and PDF, and building the index |

> The `fine_tuning_*` scripts do **not** train. They only run inference with already fine-tuned models (`shogoorg/my-embeddinggemma-2-finetuned`, `shogoorg/my-multimodal-embeddinggemma-2-finetuned`).

All scripts completed successfully (exit code 0). The numeric outputs (similarity scores, retrieved documents, and generated answers) were identical between the online and offline versions of each script.

## 4. Detailed Results

### 4.1 `inference_embeddinggemma_with_sentence_transformers`

Text-only EmbeddingGemma 2 (vision and audio encoders disabled, 271,002,624 parameters, float32).

| Step | off | on |
|---|---:|---:|
| Python start-up + model load | 58.6 s | 40.2 s |
| Embed 3 words | 3.3 s | 5.5 s |
| Similarity of 3 sentence pairs (no prompt) | 0.7 s | 2.4 s |
| Similarity of 3 sentence pairs (`STS` prompt) | 0.6 s | 0.7 s |
| Interpreter shutdown (after last output) | 6.4 s | 8.4 s |
| **Total** | **69.8 s** | **57.4 s** |

- Once the model is loaded, encoding a short sentence pair takes only **~0.2 s**.
- **Model loading accounts for 70–85% of the total time.** The online run was faster only because it ran right after the offline run, when the model file was already in the OS page cache.

### 4.2 `multimodal_embeddinggemma_with_sentence_transformers`

EmbeddingGemma 2 with the vision encoder (audio encoder disabled, 438,760,448 parameters, float32). Each similarity step encodes 1 query image and 4 candidates (1 image + 3 texts).

| Step | off | on |
|---|---:|---:|
| Python start-up + model load | 41.7 s | 47.9 s |
| Cross-modal similarity, default image token budget | 73.4 s | 57.0 s |
| Cross-modal similarity, `max_soft_tokens=1120` | 319.1 s | 375.8 s |
| Interpreter shutdown | 19.8 s | 18.3 s |
| **Total** | **454.0 s** | **499.0 s** |

- **Image encoding is the main bottleneck on CPU.** Two images take about 1 minute with the default token budget.
- With `max_soft_tokens=1120` (the maximum budget), the same step takes **5–6 minutes (~5x slower)**.
- The online version fetches the images from GitHub, while the offline version reads them from `./assets/`. The difference in timing is within run-to-run variance.

### 4.3 `fine_tuning_embeddinggemma_with_sentence_transformers`

Fine-tuned text model, 1 query and 1 document.

| Step | off | on |
|---|---:|---:|
| Python start-up + model load + encode | 44.6 s | 38.6 s |
| Interpreter shutdown | 5.5 s | 4.9 s |
| **Total** | **50.1 s** | **43.5 s** |

- Similarity: 0.8493 (both modes).
- Almost all of the time is model loading.

### 4.4 `fine_tuning_multimodal_embeddinggemma_with_sentence_transformers`

Fine-tuned multimodal model, 1 text query and 3 images from `./assets/`.

| Step | off | on |
|---|---:|---:|
| Python start-up + model load + encode | 107.3 s | 123.1 s |
| Interpreter shutdown | 13.9 s | 18.9 s |
| **Total** | **121.2 s** | **142.2 s** |

- Similarities: `[0.7091, 0.6116, 0.6654]` (both modes).
- This script loads the model **without** `config_kwargs`, so the audio encoder is loaded as well. It has the **highest peak memory of all scripts (2.89 GiB)**, close to the 3.7 GiB limit of this PC.
- The offline version logs `Could not reach the Hub ... offline mode is enabled`. This is only a warning: the model is loaded from the local cache.

### 4.5 `RAG_with_EmbeddingGemma`

Two-stage retrieval (category → document) with Sentence Transformers (text-only, float32), followed by answer generation with Gemma 4 E2B on the LiteRT-LM CPU backend. The two models are never loaded at the same time.

**Total time**

| Run | Total | Peak memory |
|---|---:|---:|
| Online, first run (including re-downloading the Gemma 4 E2B `.litertlm` file, ~2.6 GB) | 896.3 s (14:56) | 2.20 GiB |
| Offline (cached models) | 94.3 s (1:34) | 2.18 GiB |

**Per-step breakdown** (offline, from a separate timestamped run, ~107 s in total)

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
- The answers for the annual-leave and personal-package questions are `I don't know.` The original notebook (`notebooks/RAG_with_EmbeddingGemma.ipynb`) returns the same answer for the annual-leave question, so this is caused by the original prompt, not by the low-spec changes.

### 4.6 `EmbeddingGemma2_LiteRT_RAG`

RAG over the Gemma 4 technical report (45 chunks) with EmbeddingGemma 2 and Gemma 4 E2B, both on the LiteRT-LM CPU backend, with Qdrant Edge as the vector store.

**Total time**

| Run | Total | Peak memory |
|---|---:|---:|
| Online, first run (downloading the embedding model and PDF, parsing the PDF, embedding and indexing 45 chunks) | 329.7 s (5:30) | 1.53 GiB |
| Offline (index in `db_gemma/` reused) | 51.3 s | 1.46 GiB |

**Per-step breakdown** (offline, from a separate timestamped run, ~39 s up to the last answer)

| Step | Time |
|---|---:|
| Python start-up + resolve model paths | 1.1 s |
| Load embedding engine + open the existing index | 2.8 s |
| Embed and retrieve 2 queries | 1.9 s (~0.9 s each) |
| Release embedding engine + load Gemma 4 E2B | 1.9 s |

| Question | Generation |
|---|---:|
| What architecture is Gemma 4 based on? | 18.9 s |
| What context length does Gemma 4 support? | 12.0 s |

- Answers are streamed, so the first words appear before the generation time shown above has elapsed (time to first token was not measured).
- Generation takes longer than in `RAG_with_EmbeddingGemma` because each prompt includes two 1,024-character chunks and the answers are longer.
- The answers were identical to those of the original notebook (`notebooks/EmbeddingGemma2_LiteRT_RAG.ipynb`).

## 5. Key Findings

1. **Loading the Sentence Transformers (PyTorch) model is the dominant fixed cost: 38–59 s.** For the text-only scripts, actual encoding takes well under 10 s. LiteRT-LM loads its models in about 1–3 s.
2. **`EmbeddingGemma2_LiteRT_RAG` is the fastest and most memory-efficient end-to-end pipeline** (51 s offline, 1.46 GiB peak), even though it includes PDF retrieval and LLM generation.
3. **Image embedding on CPU is slow.** Two images take ~1 minute with the default token budget and 5–6 minutes with `max_soft_tokens=1120`. On low-spec PCs, use a small budget (70, 140, or 280).
4. **Memory is the tightest constraint.** Multimodal scripts reach 2.75–2.89 GiB on a 3.7 GiB machine. Disabling unused encoders (`config_kwargs={"audio_config": None}`, or vision too for text-only use) noticeably reduces memory use.
5. **Per-question latency for RAG is about 5–20 s**, almost all of which is answer generation by Gemma 4 E2B. Adding more questions increases total time linearly, but the one-time model loading cost does not grow.
6. **First runs are dominated by downloads and indexing.** Subsequent online runs use the cache and perform almost the same as offline runs.
7. **Interpreter shutdown takes 5–20 s** after the last output, because large tensors are freed. This is included in the total times.

## 6. Recommendations for Low-Spec PCs

- For RAG, prefer **`EmbeddingGemma2_LiteRT_RAG_*.py`** (LiteRT-LM for both embedding and generation).
- Keep the model loaded across many queries (for example, with an interactive loop) to avoid paying the model loading cost every time.
- For multimodal workloads, use a small `max_soft_tokens` value and disable the audio encoder when it is not needed.
- Close other applications before running the multimodal scripts to avoid swapping.
- To reduce RAG generation time, lower `RETRIEVAL_LIMIT` or `CHUNK_SIZE` (rebuild `db_gemma/` after changing `CHUNK_SIZE`), at the cost of retrieval recall.
