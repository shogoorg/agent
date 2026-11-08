import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "shogoorg/my-embeddinggemma-2-finetuned",
    model_kwargs={"torch_dtype": torch.float32},
    local_files_only=True,
)

queries = ["How do I open a NISA account?"]
documents = [
    "What is the procedure for starting a new tax-free investment account?",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)