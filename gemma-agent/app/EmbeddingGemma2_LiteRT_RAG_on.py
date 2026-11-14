"""EmbeddingGemma 2 LiteRT: RAG on Edge (low-spec CPU edition).

On-device RAG over the Gemma 4 technical report:
parse and chunk the PDF -> store embeddings in Qdrant Edge -> answer questions with Gemma 4 E2B.

Designed for low-memory, CPU-only PCs (e.g. 4 cores / 4 GB RAM / no GPU):
  - Both models run on the LiteRT-LM CPU backend.
  - The vector index (db_gemma/) is persisted; once built, later runs skip PDF parsing and embedding.
  - Chunks are embedded in small batches to keep peak memory low.
  - The two models are never held in memory at the same time: all queries are embedded and
    their contexts retrieved first, the embedding engine is closed, and only then is Gemma loaded.
"""

import gc
import urllib.request
from pathlib import Path

import edgeparse
import litert_lm
from huggingface_hub import hf_hub_download
from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)

# Low-spec settings
NUM_THREADS = 4  # Number of CPU threads for both engines
EMBED_BATCH_SIZE = 8  # Chunks embedded per batch (smaller = less memory)
MAX_NUM_TOKENS = 2048  # Gemma context size (smaller = less KV-cache memory)
RETRIEVAL_LIMIT = 2  # Number of chunks passed to Gemma as context

# Paths (relative to the project root `embeddinggemma`)
DOCUMENT_URL = "https://arxiv.org/pdf/2607.02770.pdf"
DOCUMENT_PATH = "data/gemma4_technical_report.pdf"
CACHE_DIR = ".cache"
SHARD_PATH = "db_gemma"

# Index settings
MODEL_DIM = 768
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 0

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

# 1. Download Models and Document
embed_model_path = hf_hub_download(
    repo_id="litert-community/embeddinggemma-2-740m-litert-lm",
    filename="embeddinggemma-2-740m.litertlm",
)
gemma_model_path = hf_hub_download(
    repo_id="litert-community/gemma-4-E2B-it-litert-lm",
    filename="gemma-4-E2B-it.litertlm",
)
print(f"Embedding model: {embed_model_path}")
print(f"Generation model: {gemma_model_path}")

if not Path(DOCUMENT_PATH).exists():
    print(f"Downloading document: {DOCUMENT_URL}")
    Path(DOCUMENT_PATH).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DOCUMENT_URL, DOCUMENT_PATH)
print(f"Document: {DOCUMENT_PATH}")
print("=" * 60)

# 2. Load the Embedding Engine (EmbeddingGemma 2, CPU backend)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
embedding_engine = litert_lm.EmbeddingEngine(
    embed_model_path,
    backend=litert_lm.Backend.CPU(thread_count=NUM_THREADS),
    cache_dir=CACHE_DIR,
)


# 3. Indexing (parse PDF -> chunk -> embed -> store in Qdrant Edge)
def load_or_create_shard(shard_path: str) -> EdgeShard:
    config = EdgeConfig(
        vectors=EdgeVectorParams(size=MODEL_DIM, distance=Distance.Cosine)
    )
    path = Path(shard_path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        return EdgeShard.load(shard_path)
    return EdgeShard.create(shard_path, config)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    return [
        chunk
        for start in range(0, len(text), chunk_size - overlap)
        if (chunk := text[start : start + chunk_size].strip())
    ]


shard = load_or_create_shard(SHARD_PATH)
num_points = shard.count(CountRequest(exact=True))

if num_points > 0:
    print(f"Index already built: {num_points} chunks in '{SHARD_PATH}/' (skipping indexing)")
else:
    print("Parsing PDF...")
    markdown = edgeparse.convert(DOCUMENT_PATH, format="markdown")
    chunks = chunk_text(markdown, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Embedding {len(chunks)} chunks (batch size: {EMBED_BATCH_SIZE})...")

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        responses = embedding_engine.compute_embedding_batch(batch)
        shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        start + offset + 1,
                        response.embedding,
                        {
                            "text": chunk,
                            "source": Path(DOCUMENT_PATH).name,
                            "chunk_index": start + offset,
                        },
                    )
                    for offset, (chunk, response) in enumerate(zip(batch, responses))
                ]
            )
        )
        print(f"  {min(start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)}")

    shard.optimize()
    print(f"Indexed {len(chunks)} chunks into '{SHARD_PATH}/'")
print("=" * 60)


# 4. Retrieval Phase: embed all queries and fetch their contexts
def retrieve_context(user_query: str, limit: int = RETRIEVAL_LIMIT) -> str:
    query_vector = embedding_engine.compute_embedding(user_query).embedding
    results = shard.query(
        QueryRequest(
            query=Query.Nearest(query_vector),
            limit=limit,
            with_payload=True,
        )
    )
    return "".join(point.payload["text"] for point in results)


QUERIES = [
    "What architecture is Gemma 4 based on?",
    "What context length does Gemma 4 support?",
]

retrieved = [(user_query, retrieve_context(user_query)) for user_query in QUERIES]
print(f"Retrieved context for {len(retrieved)} queries")

# Release the embedding engine and the index before loading the generation model
embedding_engine.close()
shard.close()
del embedding_engine, shard
gc.collect()
print("=" * 60)

# 5. Generation Phase: Load Gemma 4 E2B (LiteRT-LM, CPU backend)
print("Loading generation model...")
generation_engine = litert_lm.Engine(
    gemma_model_path,
    backend=litert_lm.Backend.CPU(thread_count=NUM_THREADS),
    max_num_tokens=MAX_NUM_TOKENS,
)
print("=" * 60)


def system_messages() -> list[litert_lm.Message]:
    return [
        litert_lm.Message.system(
            "You are a Gemma 4 technical report assistant. "
            "Answer the user question using only the provided context. "
            "If the answer is not in the context, say: I don't know"
            "Be smart to understand the context well and do not guess or add extra explanation."
            "Answer in 1-2 short sentences. Do not elaborate unless told by user"
        )
    ]


def answer_stream(user_query: str, context: str):
    with generation_engine.create_conversation(
        messages=system_messages(),
        sampler_config=litert_lm.SamplerConfig(top_k=1),
    ) as conversation:
        for chunk in conversation.send_message_async(
            f"Context:\n{context}\n\nQuestion:\n{user_query}"
        ):
            for item in chunk.get("content", []):
                if item.get("type") == "text" and item.get("text"):
                    yield item["text"]


# 6. RAG: stream an answer for each query
try:
    for i, (user_query, context) in enumerate(retrieved, start=1):
        print(f"[{i}] {user_query}")
        for text in answer_stream(user_query, context):
            print(text, end="", flush=True)
        print()
        print("-" * 60)
finally:
    generation_engine.close()
