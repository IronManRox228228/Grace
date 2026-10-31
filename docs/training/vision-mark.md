# Fine-Tuning Recipe: Vision Slot (PLAN.md §3, §8, §12)

Fine-tuning recipe for LFM2.5-VL-450M for selecting numbered visual marks on cropped screenshot regions when an application exposes no accessible UI automation tree.

---

## 1. Overview & Objectives

- **Target Architecture:** LFM2.5-VL-450M (Liquid Vision-Language Model)
- **Challenger in Evaluation:** Intern-Decision-0.8B (Qwen3.5-0.8B multimodal)
- **Task Formulation:** Given a cropped image of the active window overlaid with Set-of-Marks badge numbers and a natural language goal, output the target mark integer (e.g. `4`).
- **Safety Fallback:** If visual confidence is ambiguous, the system escalates directly to the user by voice ("button 4 or button 7?") rather than invoking heavy cloud models.

---

## 2. Dataset Specifications

1. **Grace Set-of-Marks Data:**
   - Generated using `src/grace/automation/som_overlay.py` from real recorded interaction tapes.
   - Screen crops pre-converted to 512x512 resolution (`.cache/working_set/vision/`).
2. **OS-Atlas Windows Split:**
   - User downloads the OS-Atlas Windows split multi-part archive.
   - Parts are merged prior to extraction using `training/data_prep.py --merge-os-atlas <dir> --os-atlas-out <tar>`.

---

## 3. Hyperparameters & Resource Profile

| Hyperparameter | Value |
| :--- | :--- |
| **Input Resolution** | 512x512 RGB |
| **Visual Feature Dim** | 256 |
| **Text Feature Dim** | 128 |
| **Max Candidate Marks** | 30 |
| **Learning Rate** | `2e-4` (Cosine learning rate schedule) |
| **Batch Size** | 4 (Effective batch size 8) |
| **Epochs** | 3 |
| **Checkpoint Interval** | 180 seconds |
| **VRAM Consumption** | ~17.9 MB on smoke adapter; **~2.2 GB** on full 4-bit VL model |
| **Step Latency (Smoke Run)** | **~215.7 ms / step** measured on RTX 4060 |

---

## 4. User Launch Instructions

```bash
# 1. Prepare and merge OS-Atlas Windows parts if downloaded
./venv/Scripts/python.exe -m training.data_prep --merge-os-atlas "C:\Users\<you>\Downloads\os_atlas_parts" --os-atlas-out "C:\Users\<you>\Downloads\os_atlas_windows.tar.gz"

# 2. Smoke test
./venv/Scripts/python.exe -m training.train_vision_mark --smoke-run --output-dir models/fine_tunes/vision_mark

# 3. Full training run on RTX 4060
./venv/Scripts/python.exe -m training.train_vision_mark --batch-size 4 --epochs 3 --lr 2e-4 --output-dir models/fine_tunes/vision_mark
```

---

## 5. Evaluation & Performance Gate

The model replacement gate evaluates the newly fine-tuned weights against both the validation Set-of-Marks set and the Intern-Decision-0.8B challenger. A new adapter replaces the baseline only if:
- Top-1 visual mark choice accuracy exceeds 85.0%.
- Latency p50 on CPU remains < 250 ms.
- It outperforms or matches Intern-Decision-0.8B while utilizing half the parameter footprint (450M vs 800M).
