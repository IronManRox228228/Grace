# Fine-Tuning Recipe: Element Selection Model (PLAN.md §3, §8, §12)

Fine-tuning recipe for GLiNER2.5-Decide (340M) and Laya (421M) on Grace's summarized UI element snapshots.

---

## 1. Overview & Objectives

- **Target Architecture:** GLiNER2.5-Decide (DeBERTa-v3-large encoder) / Laya (421M typed decision ranker)
- **Training Method:** LoRA fine-tuning on candidate element summaries
- **Deployment Format:** ONNX with dynamic `int8` quantization (`onnxruntime`)
- **Objective:** Map user intent and screen state to the exact UI element ID (`0..N`) or abstain (`#-1`), running in <50ms on CPU floor hardware.

---

## 2. Dataset & Partitioning

- **Source:** Extracted UIA/DOM snapshots from `bench/datasets/element_selection.jsonl`.
- **Element Summary Format:** Compact text tokens per candidate (name, control type, parent path, window title, neighbouring labels — **never** the whole tree).
- **Strict Held-Out Split:**
  - Training domains: Web (`Edge`), Calculator, Explorer, etc.
  - Test-only held-out domains: `Notepad`, `Settings`.
  - Zero cross-app contamination guarantees genuine generalization.
- **Abstain Sentinel:** Candidate `#-1` (`"none of these / ask the user"`) included in every sample to teach the model when to abstain.

---

## 3. Hyperparameters & Resource Profile

| Hyperparameter | Value |
| :--- | :--- |
| **LoRA Rank ($r$)** | 8 |
| **LoRA Alpha ($\alpha$)** | 16 |
| **Target Modules** | `query`, `key`, `value`, `dense` |
| **Learning Rate** | `1e-4` (Linear warmup with cosine decay) |
| **Batch Size** | 4 (Effective batch size 8) |
| **Loss Function** | Cross-Entropy over candidate scoring logits |
| **Epochs** | 3 |
| **Checkpoint Interval** | 180 seconds |
| **VRAM Consumption** | ~17.5 MB peak on smoke adapter; **~1.8 GB** on 340M base model |
| **Step Latency (Smoke Run)** | **~224.8 ms / step** measured on RTX 4060 |
| **Export Target** | Dynamic quantized ONNX Int8 (`QInt8`) |

---

## 4. User Launch Instructions

```bash
# Smoke test
./venv/Scripts/python.exe -m training.train_element_selection --smoke-run --output-dir models/fine_tunes/element_selection

# Full training run on RTX 4060
./venv/Scripts/python.exe -m training.train_element_selection --batch-size 4 --epochs 3 --lr 1e-4 --output-dir models/fine_tunes/element_selection
```

---

## 5. Evaluation & Model Replacement Gate

The `ModelReplacementGate` evaluates the trained checkpoint on the held-out applications (`Notepad`, `Settings`). The newly trained ONNX adapter replaces the production model only if:
- `held_out_top1_accuracy` strictly exceeds the baseline (> 85.0%).
- `abstain_precision` on negative samples remains above 90.0%.
- Latency p50 on CPU under the Task 08 thread budget remains < 50 ms.
