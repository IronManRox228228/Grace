# Fine-Tuning Recipe: Contrastive Decision Heads (PLAN.md §3, §8, §12)

Implementation and training recipe for Stanford/NVIDIA CLM-v0.1-8B dual-projection contrastive heads adapted to a small frozen backbone (LFM2.5-350M or Qwen3-0.6B).

---

## 1. Architecture & Principles

- **Frozen Backbone:** Small pretrained language model (LFM2.5-350M or Qwen3-0.6B) kept 100% frozen.
- **Dual Projection Heads:**
  - $\Phi_{\text{state}}: \mathbb{R}^{D_{\text{backbone}}} \to \mathbb{R}^{128}$ (maps goal + active window state to normalized embedding)
  - $\Phi_{\text{action}}: \mathbb{R}^{D_{\text{backbone}}} \to \mathbb{R}^{128}$ (maps candidate element summary or tool action to normalized embedding)
- **InfoNCE Objective:**
  $$\mathcal{L} = -\log \frac{\exp(\Phi_s(s) \cdot \Phi_a(a_+) / \tau)}{\sum_{i=1}^N \exp(\Phi_s(s) \cdot \Phi_a(a_i) / \tau)}$$
  with temperature $\tau = 0.07$.
- **Action Pre-Caching:** Because tool definitions and UI element summaries are bounded, candidate action embeddings are computed once and cached, enabling **<20 ms** CPU inference via simple matrix-vector dot products.

---

## 2. Dataset & Training Pairs

- **Positive Pairs:** `(user_instruction + window_context, correct_element_summary)` from corpus tapes and interaction trajectories.
- **Negative Pairs:** Up to 7 non-target element summaries in the active window per sample, plus the #-1 `"none of these / ask the user"` abstain candidate.

---

## 3. Hyperparameters & Resource Profile

| Hyperparameter | Value |
| :--- | :--- |
| **Backbone Embedding Dim** | 512 |
| **Projection Dim** | 128 (L2 normalized) |
| **Temperature ($\tau$)** | 0.07 |
| **Learning Rate** | `2e-4` (AdamW, weight decay `0.01`) |
| **Batch Size** | 4 (each sample has 1 positive + 7 negative actions) |
| **Epochs** | 5 |
| **Checkpoint Interval** | 180 seconds |
| **VRAM Consumption** | ~19.5 MB on smoke adapter; **~1.2 GB** with frozen 350M backbone |
| **Step Latency (Smoke Run)** | **~350.7 ms / step** measured on RTX 4060 |
| **Export Format** | Pre-cached action vectors (`.pt`) + ONNX state projector |

---

## 4. User Launch Instructions

```bash
# Smoke test
./venv/Scripts/python.exe -m training.train_contrastive_heads --smoke-run --output-dir models/fine_tunes/contrastive_heads

# Full training run
./venv/Scripts/python.exe -m training.train_contrastive_heads --batch-size 4 --lr 2e-4 --output-dir models/fine_tunes/contrastive_heads
```

---

## 5. Evaluation & Performance Gate

The trained projection heads are evaluated on held-out apps using top-1 retrieval accuracy and abstain precision. A new checkpoint replaces the baseline only if it achieves:
- Top-1 accuracy > 88.0%.
- Abstain precision on unmatched queries > 90.0%.
- Latency on CPU floor under thread budget < 20 ms.
