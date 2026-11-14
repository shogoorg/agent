import torch
from sentence_transformers import SentenceTransformer

# 1. Model Initialization
# Selectively omit the audio encoder to optimize RAM while retaining image and text support.
model_id = "google/embeddinggemma-2"
model = SentenceTransformer(
    model_id,
    config_kwargs={"audio_config": None},
    model_kwargs={"torch_dtype": torch.float32},
)

print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
print(f"Total parameters (Vision + Text): {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)

# 2. Image Embedding and Cross-Modal Similarity (Default Settings)
img_url = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png"
example_img_url = "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/kitchen_painting.jpg"

print(f"Query Image: {img_url}")
emb_goldengate = model.encode({"image": img_url})

candidates = [
    {"image": example_img_url},
    "San Francisco",
    "a cute cat walking",
    "a friendly dog sitting in the grass",
]

emb_candidates = model.encode(candidates)
similarities = model.similarity(emb_goldengate, emb_candidates)[0]

print("\nCross-Modal Similarity (Default token budget):")
print("-" * 60)
for idx, (candidate, score) in enumerate(zip(candidates, similarities)):
    label = candidate if isinstance(candidate, str) else f"[Image: {example_img_url.split('/')[-1]}]"
    print(f"  Candidate {idx + 1}: {label}")
    print(f"   `-> Similarity Score: {score.item():.4f}")
print("=" * 60)

# 3. Customizing Image Processing Parameters (max_soft_tokens)
# You can customize vision token budget per inference call (options: 70, 140, 280, 560, or 1120).
img_processing_args = {
    "image": {
        "max_soft_tokens": 1120,
    }
}

print("Computing Cross-Modal Similarity with custom token budget (max_soft_tokens=1120)...")
emb_goldengate_custom = model.encode({"image": img_url}, processing_kwargs=img_processing_args)
emb_candidates_custom = model.encode(candidates, processing_kwargs=img_processing_args)
similarities_custom = model.similarity(emb_goldengate_custom, emb_candidates_custom)[0]

print("\nCross-Modal Similarity (Custom max_soft_tokens=1120):")
print("-" * 60)
for idx, (candidate, score) in enumerate(zip(candidates, similarities_custom)):
    label = candidate if isinstance(candidate, str) else f"[Image: {example_img_url.split('/')[-1]}]"
    print(f"  Candidate {idx + 1}: {label}")
    print(f"   `-> Similarity Score: {score.item():.4f}")
print("=" * 60)

