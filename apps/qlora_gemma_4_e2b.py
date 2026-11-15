"""Inference with Fine-Tuned Gemma 4 E2B QLoRA (Text-to-SQL).

Loads the fine-tuned LoRA adapter from Hugging Face Hub (shogoorg/gemma-text-to-sql)
on top of the base model (google/gemma-4-E2B).

Auto-detects local cache:
  - If base model, adapter, and tokenizer are cached locally, loads them offline.
  - If not cached, downloads them automatically from Hugging Face.
"""

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoTokenizer,
)

base_model_id = "google/gemma-4-E2B"
adapter_id = "shogoorg/gemma-text-to-sql"

# 1. Auto-detect local cache for base model and adapter
try:
    AutoConfig.from_pretrained(base_model_id, local_files_only=True)
    is_base_cached = True
    print(f"Loaded base model config from local cache: {base_model_id}")
except Exception:
    is_base_cached = False
    print(f"Downloading base model config from Hugging Face: {base_model_id}...")

try:
    AutoConfig.from_pretrained(adapter_id, local_files_only=True)
    is_adapter_cached = True
    print(f"Loaded adapter config from local cache: {adapter_id}")
except Exception:
    is_adapter_cached = False
    print(f"Downloading adapter config from Hugging Face: {adapter_id}...")

# 2. Load Tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, local_files_only=True)
    print(f"Loaded tokenizer from local cache: {adapter_id}")
except Exception:
    print(f"Downloading tokenizer from Hugging Face: {adapter_id}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_id, local_files_only=False)

# 3. Device & Dtype
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        dtype_str = "bfloat16"
    else:
        dtype = torch.float16
        dtype_str = "float16"
    device = "cuda"
else:
    dtype = torch.float32
    dtype_str = "float32"
    device = "cpu"

print(f"Device: {device}, Precision: {dtype_str}")

# 4. Load Base Model and Fine-Tuned Adapter
print("Loading base model...")
base_model = AutoModelForMultimodalLM.from_pretrained(
    base_model_id,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
    local_files_only=is_base_cached,
)

print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_id,
    local_files_only=is_adapter_cached,
)
model.eval()
print("Model loaded successfully.")
print("=" * 60)

# 5. Text-to-SQL Prompt Template
system_message = (
    "You are a text to SQL query translator. "
    "Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
)

user_prompt_template = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

# Sample Schema and Questions
schema = """CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10, 2),
    hire_date DATE
);

CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50),
    budget DECIMAL(12, 2)
);"""

questions = [
    "What is the average salary of employees in the Engineering department?",
    "List the first name, last name, and salary of all employees hired after January 1, 2023.",
]

for idx, question in enumerate(questions, start=1):
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": user_prompt_template.format(context=schema, question=question),
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print(f"[{idx}] Question: {question}")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    generated_text = generated_text.removesuffix("<turn|>").strip()

    print(f"Generated SQL:\n{generated_text}")
    print("-" * 60)
