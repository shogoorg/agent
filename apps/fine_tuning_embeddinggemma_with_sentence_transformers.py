import torch
from sentence_transformers import SentenceTransformer

model_id = "shogoorg/my-embeddinggemma-2-finetuned"

try:
    model = SentenceTransformer(
        model_id,
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=True,
    )
    print(f"Loaded model from local cache: {model_id}")
except Exception:
    print(f"Downloading model from Hugging Face: {model_id}...")
    model = SentenceTransformer(
        model_id,
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=False,
    )

queries = ["How do I open a NISA account?"]
documents = [
    "Opening a NISA Account",
    "Opening a Regular Savings Account",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
similarities = model.similarity(query_embeddings, document_embeddings)

print(f"Similarity: {similarities.item():.4f}")
