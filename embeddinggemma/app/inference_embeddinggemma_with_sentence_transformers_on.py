import torch
from sentence_transformers import SentenceTransformer

# 1. Model Initialization (optimized for text-only inference to minimize memory usage)
model_id = "google/embeddinggemma-2"
model = SentenceTransformer(
    model_id,
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": torch.float32},
)

print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)

# 2. Generating Text Embeddings
words = ["king", "queen", "car"]
embeddings = model.encode(words)

print("Generated Embeddings:")
for word, emb in zip(words, embeddings):
    print(f"  Word: '{word}' -> Embedding shape: {emb.shape}")
print("=" * 60)

# 3. Determining Semantic Similarity (Raw Embeddings)
sentence_high = [
    "The chef prepared a delicious meal for the guests.",
    "A tasty dinner was cooked by the chef for the visitors.",
]
sentence_medium = [
    "She is an expert in machine learning.",
    "He has a deep interest in artificial intelligence.",
]
sentence_low = [
    "The weather in Tokyo is sunny today.",
    "I need to buy groceries for the week.",
]

pairs = [
    ("High Similarity", sentence_high),
    ("Medium Similarity", sentence_medium),
    ("Low Similarity", sentence_low),
]

print("Semantic Similarity (without task prompts):")
for label, pair in pairs:
    pair_embeddings = model.encode(pair)
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f"  [{label}] Score: {score:.4f}")
    print(f"    Sentence A: {pair[0]}")
    print(f"    Sentence B: {pair[1]}")
print("=" * 60)

# 4. Optimizing Embeddings with Task Prompts (STS)
print("Available Task Prompts:")
for name, prefix in model.prompts.items():
    print(f"  {name}: {prefix!r}")
print("-" * 60)

print("Semantic Similarity with STS Task Prompt (prompt_name='STS'):")
for label, pair in pairs:
    pair_embeddings = model.encode(pair, prompt_name="STS")
    score = model.similarity(pair_embeddings[0], pair_embeddings[1]).item()
    print(f"  [{label}] STS Score: {score:.4f}")
print("=" * 60)
