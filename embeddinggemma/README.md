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

---

## 3. Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks for training, fine-tuning, and exploratory inference:

- `fine-tuning-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning EmbeddingGemma for text retrieval.
- `fine-tuning-multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Fine-tuning multimodal EmbeddingGemma with text-image pairs.
- `inference-embeddinggemma-with-sentence-transformers.ipynb`: Interactive text retrieval inference.
- `multimodal-embeddinggemma-with-sentence-transformers.ipynb`: Interactive multimodal retrieval inference.
