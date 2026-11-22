# Coding Agent Guide

## Prerequisites

Install the CLI (one-time):
```bash
uv tool install google-agents-cli
```

---

## Development Phases

### Phase 1: Understand Requirements
Before writing any code, understand the project's requirements, constraints, and success criteria.

### Phase 2: Build and Implement
Implement agent logic in `app/`. Use `agents-cli playground` for interactive testing. Iterate based on user feedback.

### Phase 3: The Evaluation Loop (Main Iteration Phase)
Start with 1-2 eval cases, run `agents-cli eval run`, iterate by making changes and rerunning it until satisfied. Expect 5-10+ iterations. Once you have a baseline, reach for `agents-cli eval compare` (regression diffs), `agents-cli eval analyze` (cluster failure modes), and `agents-cli eval optimize` (auto-tune prompts). See the **Evaluation Guide** for metrics, dataset schema, LLM-as-judge config, and common gotchas.

### Phase 4: Pre-Deployment Tests
Run `uv run pytest tests/unit tests/integration`. Fix issues until all tests pass.

### Phase 5: Deploy to Dev
**Requires explicit human approval.** Run `agents-cli deploy` only after user confirms. See the **Deployment Guide** for details.

### Phase 6: Production Deployment
Ask the user: Option A (simple single-project) or Option B (full CI/CD pipeline with `agents-cli infra cicd`).

## Development Commands

| Command | Purpose |
|---------|---------|
| `agents-cli playground` | Interactive local testing |
| `uv run pytest tests/unit tests/integration` | Run unit and integration tests |
| `agents-cli eval dataset synthesize` | Synthesize multi-turn eval scenarios for your agent |
| `agents-cli eval run` | Run the agent over the eval dataset and grade the traces |
| `agents-cli eval generate` / `agents-cli eval grade` | Decoupled form: produce traces, then grade them |
| `agents-cli eval compare` | Compare two grade-results files (regression check) |
| `agents-cli eval analyze` | Cluster failure modes from grade results |
| `agents-cli eval metric list` | List built-in metrics available in the SDK |
| `agents-cli eval optimize` | Auto-tune agent prompts using eval data |
| `agents-cli lint` | Check code quality |
| `agents-cli infra single-project` | Set up project infrastructure (Terraform) |
| `agents-cli deploy` | Deploy to dev |
| `agents-cli scaffold enhance` | Add deployment target or CI/CD to project |
| `agents-cli scaffold upgrade` | Upgrade project to latest version |

---

## Operational Guidelines for Coding Agents

- **Code preservation**: Only modify code directly targeted by the user's request. Preserve all surrounding code, config values (e.g., `model`), comments, and formatting.
- **NEVER change the model** unless explicitly asked.
- **Model 404 errors**: Fix `GOOGLE_CLOUD_LOCATION` (e.g., `global` instead of `us-east1`), not the model name.
- **ADK tool imports**: Import the tool instance, not the module: `from google.adk.tools.load_web_page import load_web_page`
- **Run Python with `uv`**: `uv run python script.py`. Run `agents-cli install` first.
- **Stop on repeated errors**: If the same error appears 3+ times, fix the root cause instead of retrying.
- **Terraform conflicts** (Error 409): Use `terraform import` instead of retrying creation.

---

## Gemma 4 / EmbeddingGemma 2 / LiteRT-LM Guidelines

Always consult the reference notebooks in `notebooks/` and scripts in `apps/` in this repository before modifying model code or API calls.

### Gemma 4 (Fine-Tuning & Inference)
- **Libraries**: Use `transformers>=5.10.1`, `peft>=0.19.0`, `trl`, `bitsandbytes`, and `torchao`.
- **Model Loading**: Load Gemma 4 multimodal checkpoints with `AutoModelForMultimodalLM` and `AutoProcessor`.
- **Chat Template & EOS**: Set `eos_token_id` to `[processor.tokenizer.convert_tokens_to_ids("<turn|>")]` when generating text.
- **QLoRA Export Prep**: Run `merge_and_unload()` and save the full merged checkpoint (`merged_model/`) before converting to LiteRT-LM.
- **Hardware Precision**: L4/A100 GPUs support `bfloat16` and optional `flash-attn`; T4 GPUs do not support native `bfloat16` (fall back to `fp16` if needed).

### EmbeddingGemma 2 (`google/embeddinggemma-2`)
- **Libraries**: Use `sentence-transformers>=6.0.0`, `transformers>=5.0.0,<6.0.0`, `torch`, `torchvision`, `torchcodec>=0.16.0`, and `Pillow`.
- **Precision Constraint**: EmbeddingGemma 2 activations are **incompatible with `float16`**. Always use `bfloat16` (supported GPUs) or `float32` (CPU).
- **Memory Optimization (Low-RAM / CPU Environments)**:
  - Text-only (~271M params): pass `config_kwargs={"vision_config": None, "audio_config": None}` to `SentenceTransformer`.
  - Vision + Text (~439M params): pass `config_kwargs={"audio_config": None}`.
- **Task Prompts**: Apply built-in `prompt_name` values (e.g., `"STS"`, `"Retrieval-query"`, `"Retrieval-document"`, `"Clustering"`, `"Classification"`) for **text inputs only**. Pass image/audio/video inputs directly without text prefixes.
- **Offline Execution**: Pass `local_files_only=True` to `SentenceTransformer`, set `HF_HUB_OFFLINE=1`, and reference local files under `./assets/`.

### LiteRT-LM (`.litertlm` Export, Verification & Hub Upload)
- **Converter V2**: Export `merged_model/` using `litert_torch.generative.export_hf.export(..., use_v2=True)` with `tf-nightly`, `litert-torch-nightly`, `litert-quantizer-nightly`, `litert-lm-nightly`, `litert-lm-builder`, and `litert-converter`, pinning ABI-sensitive packages (`numpy`, `pyarrow`, `pillow`).
- **Quantization Recipes**: Default to `dynamic_wi4b32_octav_afp32` (~1.5 GB for E2B) or `gemma4_mixed48_b32_octav` for higher accuracy.
- **Verification**: Validate `.litertlm` packages with `litert_lm.Engine` in Python or the local `litert-lm` CLI.
- **Hugging Face Hub Uploads**: For large `.litertlm` files (~1.5 GB+), install `hf_transfer`, set `HF_HUB_ENABLE_HF_TRANSFER=1`, and verify file existence/size before `HfApi.upload_file` to prevent HTTP/Colab timeouts. Ensure the HF token has **Write** permissions.
