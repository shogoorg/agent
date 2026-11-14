import os
import urllib.request
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

# 1. Model Initialization
# Auto-detects local cache: loads offline if cached, downloads online otherwise.
model_id = "google/embeddinggemma-2"
try:
    model = SentenceTransformer(
        model_id,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=True,
    )
    print(f"Loaded model from local cache: {model_id}")
except Exception:
    print(f"Downloading model from Hugging Face: {model_id}...")
    model = SentenceTransformer(
        model_id,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=False,
    )

print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
print(f"Total parameters (Vision + Text): {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)


def resolve_image(local_rel_path: str, remote_url: str) -> str:
    """Use local asset if present, otherwise download or fallback to URL."""
    path = Path(local_rel_path)
    if path.exists():
        return str(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {remote_url} -> {local_rel_path}...")
        urllib.request.urlretrieve(remote_url, str(path))
        return str(path)
    except Exception:
        return remote_url


# 2. Image Embedding and Cross-Modal Similarity
img_source = resolve_image(
    "./assets/GoldenGate.png",
    "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
)
example_img_source = resolve_image(
    "./assets/kitchen_painting.jpg",
    "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/kitchen_painting.jpg",
)

print(f"Query Image: {img_source}")
emb_goldengate = model.encode({"image": img_source})

candidates = [
    {"image": example_img_source},
    "San Francisco",
    "a cute cat walking",
    "a friendly dog sitting in the grass",
]

emb_candidates = model.encode(candidates)
similarities = model.similarity(emb_goldengate, emb_candidates)[0]

print("\nCross-Modal Similarity (Default token budget):")
print("-" * 60)
for idx, (candidate, score) in enumerate(zip(candidates, similarities)):
    label = candidate if isinstance(candidate, str) else f"[Image: {os.path.basename(candidate['image'])}]"
    print(f"  Candidate {idx + 1}: {label}")
    print(f"   `-> Similarity Score: {score.item():.4f}")
print("=" * 60)

# 3. Customizing Image Processing Parameters (max_soft_tokens)
img_processing_args = {
    "image": {
        "max_soft_tokens": 1120,
    }
}

print("Computing Cross-Modal Similarity with custom token budget (max_soft_tokens=1120)...")
emb_goldengate_custom = model.encode({"image": img_source}, processing_kwargs=img_processing_args)
emb_candidates_custom = model.encode(candidates, processing_kwargs=img_processing_args)
similarities_custom = model.similarity(emb_goldengate_custom, emb_candidates_custom)[0]

print("\nCross-Modal Similarity (Custom max_soft_tokens=1120):")
print("-" * 60)
for idx, (candidate, score) in enumerate(zip(candidates, similarities_custom)):
    label = candidate if isinstance(candidate, str) else f"[Image: {os.path.basename(candidate['image'])}]"
    print(f"  Candidate {idx + 1}: {label}")
    print(f"   `-> Similarity Score: {score.item():.4f}")
print("=" * 60)
