import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "shogoorg/my-multimodal-embeddinggemma-2-finetuned",
    model_kwargs={"torch_dtype": torch.float32},
)

queries = [
    'Shin Yunbok',
]
documents = [
    'https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_0.jpg',
    'https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_1.jpg',
    'https://huggingface.co/shogoorg/my-multimodal-embeddinggemma-2-finetuned/resolve/main/assets/image_2.jpg',
]
query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)

similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
