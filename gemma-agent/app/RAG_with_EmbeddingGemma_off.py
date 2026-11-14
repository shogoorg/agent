"""RAG with EmbeddingGemma 2 (low-spec CPU edition, offline mode).

Offline mode: both models are loaded from the local Hugging Face cache only
(local_files_only=True). Run the online version once first to download them.

Designed for low-memory, CPU-only PCs (e.g. 4 cores / 4 GB RAM / no GPU):
  - Retrieval: EmbeddingGemma 2 via Sentence Transformers (text-only, vision/audio encoders disabled).
  - Generation: Gemma 4 E2B via LiteRT-LM on CPU (quantized .litertlm file, ~3 GB on disk, memory-mapped)
    instead of the full transformers checkpoint, which would need ~10 GB of RAM.
  - The two models are never held in memory at the same time: all retrieval runs first,
    the embedding model is released, and only then is the generation model loaded.
"""

import gc

import litert_lm
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

# Low-spec settings
NUM_THREADS = 4  # Number of CPU threads for both models
MAX_NUM_TOKENS = 2048  # LiteRT-LM context size (smaller = less KV-cache memory)
MAX_OUTPUT_TOKENS = 256  # Upper bound on generated answer length

torch.set_num_threads(NUM_THREADS)

# 1. Corporate Knowledge Base
corp_knowledge_base = [
    {
        "category": "HR & Leave Policies",
        "documents": [
            {
                "title": "Procedure for Unscheduled Absence",
                "content": "In the event of an illness or emergency preventing you from working, please notify both your direct manager and the HR department via email by 9:30 AM JST. The subject line should be 'Sick Leave - [Your Name]'. If the absence extends beyond two consecutive days, a doctor's certificate (診断書) will be required upon your return.",
            },
            {
                "title": "Annual Leave Policy",
                "content": "Full-time employees are granted 10 days of annual paid leave in their first year. This leave is granted six months after the date of joining and increases each year based on length of service. For example, an employee in their third year of service is entitled to 14 days per year. For a detailed breakdown, please refer to the attached 'Annual Leave Accrual Table'.",
            },
        ],
    },
    {
        "category": "IT & Security",
        "documents": [
            {
                "title": "Account Password Management",
                "content": "If you have forgotten your password or your account is locked, please use the self-service reset portal at https://reset.ourcompany. You will be prompted to answer your pre-configured security questions. For security reasons, the IT Help Desk cannot reset passwords over the phone or email. If you have not set up your security questions, please visit the IT support desk on the 12th floor of the Shibuya office with your employee ID card.",
            },
            {
                "title": "Software Procurement Process",
                "content": "All requests for new software must be submitted through the 'IT Service Desk' portal under the 'Software Request' category. Please include a business justification for the request. All software licenses require approval from your department head before procurement can begin. Please note that standard productivity software is pre-approved and does not require this process.",
            },
        ],
    },
    {
        "category": "Finance & Expenses",
        "documents": [
            {
                "title": "Expense Reimbursement Policy",
                "content": "To ensure timely processing, all expense claims for a given month must be submitted for approval no later than the 5th business day of the following month. For example, all expenses incurred in July must be submitted by the 5th business day of August. Submissions after this deadline may be processed in the next payment cycle.",
            },
            {
                "title": "Business Trip Expense Guidelines",
                "content": "Travel expenses for business trips will, as a rule, be reimbursed based on the actual cost of the most logical and economical route. Please submit a travel expense application in advance when using the Shinkansen or airplanes. Taxis are permitted only when public transportation is unavailable or when transporting heavy equipment. Receipts are mandatory.",
            },
        ],
    },
    {
        "category": "Office & Facilities",
        "documents": [
            {
                "title": "Conference Room Booking Instructions",
                "content": "All conference rooms in the Shibuya office can be reserved through your Calendar App. Create a new meeting invitation, add the attendees, and then use the 'Room Finder' feature to select an available room. Please be sure to select the correct floor. For meetings with more than 10 people, please book the 'Sakura' or 'Fuji' rooms on the 14th floor.",
            },
            {
                "title": "Mail and Delivery Policy",
                "content": "The company's mail services are intended for business-related correspondence only. For security and liability reasons, employees are kindly requested to refrain from having personal parcels or mail delivered to the Shibuya office address. The front desk will not be able to accept or hold personal deliveries.",
            },
        ],
    },
]

# 2. Questions and Similarity Threshold
questions = [
    "How many days of annual paid leave do I get?",
    "How do I reset my password?",
    "What are the reimbursable expense guidelines for business travel?",
    "Can I receive personal packages at the office?",
]

# Minimum confidence threshold for a match to be considered valid
similarity_threshold = 0.4


# 3. Helper Functions for Semantic Search
def _calculate_best_match(similarities):
    if similarities is None or similarities.nelement() == 0:
        return None, 0.0

    # Find the index and value of the highest score
    best_index = similarities.argmax().item()
    best_score = similarities[0, best_index].item()
    return best_index, best_score


def find_best_category(model, query, candidates):
    """Finds the most relevant category name using the Clustering prompt."""
    if not candidates:
        return None, 0.0

    query_embedding = model.encode(query, prompt_name="Clustering")
    candidate_embeddings = model.encode(candidates, prompt_name="Clustering")
    return _calculate_best_match(model.similarity(query_embedding, candidate_embeddings))


def find_best_doc(model, query, candidates):
    """Finds the most relevant document (dict with 'title' and 'content')."""
    if not candidates:
        return None, 0.0

    # Encode the query for retrieval
    query_embedding = model.encode(query, prompt_name="Retrieval-query")

    # Encode the documents with the "title: ... | text: ..." document prompt format
    doc_texts = [
        f"title: {doc.get('title', 'none')} | text: {doc.get('content', '')}"
        for doc in candidates
    ]
    candidate_embeddings = model.encode(doc_texts)
    return _calculate_best_match(model.similarity(query_embedding, candidate_embeddings))


def retrieve(model, query):
    """Two-stage retrieval: category first, then document within the category."""
    # Step 1: Find the most relevant category
    print("Step 1: Finding the best category...")
    categories = [item["category"] for item in corp_knowledge_base]
    best_category_index, category_score = find_best_category(model, query, categories)

    if category_score < similarity_threshold:
        print(f"  -> No relevant category found. The highest score was only {category_score:.2f}.")
        return None

    best_category = corp_knowledge_base[best_category_index]
    print(f"  -> Category Found: '{best_category['category']}' (Score: {category_score:.2f})")

    # Step 2: Find the most relevant document within that category
    print("Step 2: Finding the best document in that category...")
    best_document_index, document_score = find_best_doc(model, query, best_category["documents"])

    if document_score < similarity_threshold:
        print(f"  -> No relevant document found. The highest score was only {document_score:.2f}.")
        return None

    best_document = best_category["documents"][best_document_index]
    print(f"  -> Document Found: '{best_document['title']}' (Score: {document_score:.2f})")
    return best_document


# 4. Retrieval Phase: Load the Embedding Model (offline mode using locally cached weights)
# Note: EmbeddingGemma 2 activations are incompatible with float16. Use bfloat16 or float32.
embedding_model_id = "google/embeddinggemma-2"
model = SentenceTransformer(
    embedding_model_id,
    device="cpu",
    config_kwargs={"vision_config": None, "audio_config": None},
    model_kwargs={"torch_dtype": torch.float32},
    local_files_only=True,
)

print(f"Device: {model.device}")
print(f"Dtype: {model.dtype}")
print(f"Total parameters (text-only): {sum(p.numel() for p in model.parameters()):,}")
print("=" * 60)

print("Available Task Prompts:")
for name, prefix in model.prompts.items():
    print(f"  {name}: {prefix!r}")
print("=" * 60)

retrieved = []
with torch.inference_mode():
    for question in questions:
        print(f"Question: {question}")
        retrieved.append((question, retrieve(model, question)))
        print("-" * 60)

# Release the embedding model before loading the generation model
del model
gc.collect()
print("=" * 60)

# 5. Generation Phase: Load Gemma 4 E2B (LiteRT-LM, CPU backend, locally cached file)
litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

llm_model_path = hf_hub_download(
    repo_id="litert-community/gemma-4-E2B-it-litert-lm",
    filename="gemma-4-E2B-it.litertlm",
    local_files_only=True,
)
print(f"Loading generation model: {llm_model_path}")
generation_engine = litert_lm.Engine(
    llm_model_path,
    backend=litert_lm.Backend.CPU(thread_count=NUM_THREADS),
    max_num_tokens=MAX_NUM_TOKENS,
)
print("=" * 60)

qa_prompt_template = """Answer the following QUESTION based only on the CONTEXT provided. If the answer cannot be found in the CONTEXT, write "I don't know."

---
CONTEXT:
{context}
---
QUESTION:
{question}
"""


def generate_answer(question, document):
    if not document or "content" not in document:
        return "I'm sorry, I could not find a relevant document to answer that question."

    prompt = qa_prompt_template.format(context=document["content"], question=question)

    # A fresh conversation per question so earlier Q&A does not leak into the context
    with generation_engine.create_conversation(
        sampler_config=litert_lm.SamplerConfig(top_k=1),  # Greedy decoding for stable answers
        max_output_tokens=MAX_OUTPUT_TOKENS,
    ) as conversation:
        response = conversation.send_message(prompt)

    return "".join(
        item.get("text", "")
        for item in response.get("content", [])
        if item.get("type") == "text"
    ).strip()


# 6. Run the RAG Pipeline (answer generation with the retrieved context)
try:
    for question, best_document in retrieved:
        print(f"Question: {question}")
        answer = generate_answer(question, best_document)
        if best_document:
            print(f"Using document: {best_document['title']}")
        print(f"Answer: {answer}")
        print("=" * 60)
finally:
    generation_engine.close()
