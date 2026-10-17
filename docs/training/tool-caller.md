# Fine-Tuning Recipe: Tool Calling Model (PLAN.md §3, §8, §12)

Fine-tuning recipe for LFM2.5-350M (and FunctionGemma 270M) on Grace's operational tool schema for fast, offline, structured tool calling.

---

## 1. Overview & Objectives

- **Target Architecture:** LFM2.5-350M (Liquid Foundation Models) / FunctionGemma 270M
- **Training Method:** Parameter-Efficient Fine-Tuning (PEFT LoRA) on NVIDIA RTX 4060 (8 GB VRAM)
- **Deployment Format:** GGUF (Q4_K_M) served via local `llama-server` or direct C++ engine
- **Objective:** Translate conversational voice utterances into exact JSON API tool invocations matching Grace's frozen contract without schema hallucinations.

---

## 2. Dataset Specifications

- **Corpus Utterances:** Extracted from Grace's real interaction tapes (`bench/datasets/tool_calling.jsonl`).
- **Synthetic Schema Calls:** Systematic expansions generated from `src/grace/intent/tools.py` covering:
  - System tools: `open_app`, `close_app`, `adjust_volume`, `lock_computer`, `search_files`, `open_file`, `delete_file`, `read_pdf`, `summarize_pdf`, `undo`, `describe_screen`, `set_speech_rate`.
  - Computer Use Actuator (CUA) tools: `cua_click`, `cua_type_text`, `cua_press_key`, `cua_scroll`, `cua_drag`, `cua_set_value`.
- **Formatting:** Standard ChatML syntax:
  ```text
  <|im_start|>user
  click the search box and type hello
  <|im_end|>
  <|im_start|>assistant
  {"name": "cua_click", "arguments": {"element_id": 2}}<|im_end|>
  ```

---

## 3. Hyperparameters & Resource Profile

| Hyperparameter | Value |
| :--- | :--- |
| **LoRA Rank ($r$)** | 16 |
| **LoRA Alpha ($\alpha$)** | 32 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **LoRA Dropout** | 0.05 |
| **Learning Rate** | `2e-4` (Cosine learning rate decay with 10% warmup) |
| **Batch Size** | 4 (Effective batch size 8 with gradient accumulation steps = 2) |
| **Optimizer** | `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01) |
| **Epochs** | 3 |
| **Checkpoint Interval** | 180 seconds (atomic checkpointing with metadata) |
| **VRAM Consumption** | ~26 MB peak on smoke test adapter; **~2.8 GB** on 4-bit base model LoRA |
| **Step Latency (Smoke Run)** | **~304.7 ms / step** measured on RTX 4060 |
| **Estimated Epoch Duration** | ~1.5 minutes per epoch on full dataset |

---

## 4. User Launch Instructions

Ensure training dependencies are installed (`pip install -r requirements-train.txt`), then run:

```bash
# Smoke test (runs 2 steps to verify GPU and checkpointing)
./venv/Scripts/python.exe -m training.train_tool_caller --smoke-run --output-dir models/fine_tunes/tool_caller

# Full training run on RTX 4060
./venv/Scripts/python.exe -m training.train_tool_caller --batch-size 4 --epochs 3 --lr 2e-4 --output-dir models/fine_tunes/tool_caller

# Convert LoRA checkpoint to GGUF
python llama.cpp/convert_hf_to_gguf.py models/fine_tunes/tool_caller --outtype q4_k_m --outfile models/lfm2.5-350m-tool-q4.gguf
```

---

## 5. Evaluation & Model Replacement Gate

The model replacement gate (`ModelReplacementGate`) validates the checkpoint against held-out validation utterances. A newly trained adapter replaces the live model only if:
- Exact tool accuracy exceeds baseline (> 95.0%).
- Arguments match accuracy improves over baseline (> 90.0%).
- JSON validity rate equals 100.0%.
