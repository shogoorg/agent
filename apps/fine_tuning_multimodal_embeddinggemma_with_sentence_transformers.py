import urllib.request
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

model_id = "shogoorg/my-multimodal-embeddinggemma-2-finetuned"

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

queries = [
    "Shin Yunbok",
]

image_urls = [
    "https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_0.jpg",
    "https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_1.jpg",
    "https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_2.jpg",
]

documents = []
for idx, url in enumerate(image_urls):
    local_path = Path(f"./assets/image_{idx}.jpg")
    if local_path.exists():
        documents.append(str(local_path))
    else:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {url} to {local_path}...")
            urllib.request.urlretrieve(url, str(local_path))
            documents.append(str(local_path))
        except Exception:
            documents.append(url)

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print("Shapes:", query_embeddings.shape, document_embeddings.shape)

similarities = model.similarity(query_embeddings, document_embeddings)
print("Similarities:", similarities)
