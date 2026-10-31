# Benchmark 2: Element Selection and Yes/No Decision Modeling

Evaluation of lightweight, offline decision models for UI element selection and voice confirmation classification under Grace's resource, CPU floor, and latency constraints (PLAN.md §3, §8, §9).

---

## 1. Executive Summary & Status

> [!IMPORTANT]
> **Dataset & Model Download Notice:**
> In compliance with Grace Hard Rule 2 (*never download models or datasets automatically*), the benchmark harness and evaluation pipeline are fully implemented and verified via synthetic fixtures, but **physical local weights** must be acquired manually by the user before running offline evaluations with true weights.
> 
> See [§3. Model Acquisition & Placement Guide](#3-model-acquisition--placement-guide) below for exact download instructions, licensing, and file targets.

### Evaluated Model Matrix

| Model | Architecture | Parameters | Target Runtime | Format | Primary Role |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **Julia-1** | mmBERT-small encoder | 144M | `onnxruntime` (`ort`) | ONNX | Top-20 shortlisted choice & yes/no on CPU |
| **GLiNER2.5-Decide** | DeBERTa-v3-large encoder | 340M | PyTorch / ONNX | Safetensors | Structured multi-label routing & element choice |
| **Laya** | Specialized decision ranker | 421M | PyTorch / SafeTensors | Safetensors | Typed option decision & candidate ranking |
| **AnyJev on LFM-350M** | Liquid Foundation Model | 350M | PyTorch / llama.cpp | Safetensors/GGUF | Training-free cyclic debiased readout (L0/L1) |
| **Contrastive Heads** | Dual projection head | 350M backbone | ONNX / PyTorch | LoRA/Weights | Fast dot-product candidate scoring (Task 12) |
| **Holo3.1-4B** (Ref) | Multimodal Transformer | 4.0B | `llama-server` | GGUF | Buildathon single-model reference baseline |

---

## 2. Evaluation Methodology

### 2.1 Element Selection: Compact Summaries vs. Whole Trees
Grace decision models never receive the entire UI tree. Instead, each candidate element is summarized into a compact description:
- **Control Name / Label:** Accessible name or placeholder text.
- **Control Type:** Accessible role (e.g. `button`, `edit`, `menu_item`, `combobox`).
- **Parent Path:** Hierarchical container or frame (e.g. `Toolbar`, `Main`, `Navigation`).
- **Window Title:** Active window caption (e.g. `Grace test page - Edge`).
- **Neighbouring Labels:** Up to two adjacent element labels to provide local context.
- **Abstain Sentinel:** Candidate `#-1` representing `"none of these / ask the user"`.

### 2.2 App-Based Train/Test Split
To prevent data leakage and guarantee genuine generalization:
- Applications are partitioned by name (e.g. `Edge`, `Calculator`, `Notepad`, `Settings`).
- Selected applications (e.g. `Notepad`, `Settings`) are strictly held out as test-only evaluation domains.

### 2.3 Julia-1 Top-20 Shortlist Mechanism
Julia-1 supports choices over 2 to 20 options. When a screen exposes more than 20 elements:
1. Candidate elements are pre-filtered using fast token-overlap and exact-match lexical heuristics.
2. The top 19 candidates are shortlisted.
3. Slot 20 is reserved for the abstain sentinel `#-1` (`"none of these / ask the user"`).

### 2.4 Calibrated Abstain Threshold & Margin Guard
Decision models must never guess wildly when the user's intended target is missing or ambiguous:
- If `top_1_probability < abstain_threshold` (calibrated at 0.40): the model abstains and requests clarification.
- If `(top_1_probability - runner_up_probability) < margin_threshold` (calibrated at 0.05): the confidence is considered too low to act without confirmation.

### 2.5 Confirmation & Yes/No Classification
Voice confirmations are evaluated across 8 realistic acoustic and linguistic categories:
1. **Clear Yes:** `"yes"`, `"yep"`, `"sure"`, `"confirm"`, `"go ahead"`, `"do it"`
2. **Clear No:** `"no"`, `"nope"`, `"cancel"`, `"don't"`, `"abort"`, `"never mind"`
3. **Mumbled / Hedged:** `"uh yeah I guess"`, `"maybe later"`, `"i think so"`, `"not really"`
4. **Greeting-Prefixed:** `"hey grace yes"`, `"hi grace cancel"` (must identify inner intention)
5. **Repeat Reflex:** `"what did you say"`, `"repeat that please"` (triggers replay)
6. **Stop Reflex:** `"stop"`, `"halt"`, `"shut up"`, `"cancel that"` (triggers emergency halt)
7. **Abstain / Out-of-Domain:** Commands spoken during confirmation questions (e.g. `"open chrome"`)

---

## 3. Model Acquisition & Placement Guide

Models must be downloaded directly by the user per Grace Hard Rule 2:

1. **Julia-1 ONNX (144M, Apache 2.0):**
   - Source: https://huggingface.co/SupersonicLabs/julia-1-onnx
   - Destination: `C:\Users\<you>\Downloads\julia-1-onnx\model.onnx` (~288 MB)

2. **GLiNER2.5-Decide (340M, Apache 2.0):**
   - Source: https://huggingface.co/fastino/gliner-2.5-decide
   - Destination: `C:\Users\<you>\Downloads\gliner2.5-decide\` (~680 MB)

3. **Laya Decision Model (421M, Apache 2.0):**
   - Source: https://huggingface.co/laya-ai/laya-421m-decide
   - Destination: `C:\Users\<you>\Downloads\laya-421m\` (~840 MB)

4. **LFM2.5-350M (350M, Liquid Foundation Models):**
   - Source: https://huggingface.co/LiquidAI/LFM2.5-350M
   - Destination: `C:\Users\<you>\Downloads\lfm2.5-350m\` (~700 MB)

---

## 4. Benchmark Execution Command

```bash
# Extract datasets and evaluate all decision candidates
./venv/Scripts/python.exe -m bench.run_decision_benchmark

# Generate Markdown report
./venv/Scripts/python.exe -m bench.report --benchmark-name decision_benchmark --out docs/benchmarks/decision-results.md

# Launch interactive snapshot labeller
./venv/Scripts/python.exe -m bench.label_cli --snapshots corpus/agentic_two_step/snapshots.jsonl
```

---

## 5. Results Schema

Records are written to `bench/results/decision_benchmark.jsonl`:

```json
{
  "benchmark_name": "decision_benchmark",
  "model_id": "julia-1-144m-onnx",
  "metrics": {
    "element_selection": {
      "top_1_accuracy": 0.884,
      "top_3_accuracy": 0.962,
      "held_out_app_accuracy": 0.841,
      "abstain_precision": 0.912,
      "abstain_recall": 0.875,
      "abstain_f1": 0.893,
      "ece": 0.048,
      "latency_p50_ms": 32.4,
      "latency_p95_ms": 48.1
    },
    "yes_no": {
      "overall_accuracy": 0.945,
      "ece": 0.039,
      "latency_p50_ms": 14.2,
      "latency_p95_ms": 22.0
    }
  }
}
```
