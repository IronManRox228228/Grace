# Benchmark 3: Tool Calling Bake-off

Evaluation of compact, edge-capable tool and function calling models on Grace's frozen operational schema (PLAN.md §3, §8, §9).

---

## 1. Executive Summary & Status

> [!IMPORTANT]
> **Dataset & Model Download Notice:**
> In compliance with Grace Hard Rule 2 (*never download models or datasets automatically*), the benchmark harness and evaluation pipeline are fully implemented and verified via synthetic fixtures, but **physical local weights** must be acquired manually by the user before running offline evaluations with true weights.
> 
> See [§3. Model Acquisition & Placement Guide](#3-model-acquisition--placement-guide) below for exact download instructions, licensing, and placement.

### Evaluated Model Matrix

| Model | Parameters | Creator | Target Runtime | Primary Advantage |
| :--- | :---: | :--- | :--- | :--- |
| **FunctionGemma 270M** | ~270M | Google DeepMind | Hugging Face / ONNX | Ultra-compact memory footprint (~300MB RAM), ~125 tok/s on CPU |
| **LFM2.5-350M** | ~350M | Liquid AI | PyTorch / llama.cpp | Native structured generation, dual tool-caller and decision head |
| **xLAM-2-1b-fc-r** | ~1.0B | Salesforce APG | Hugging Face / vLLM | Top Berkeley Function Calling Leaderboard performer in 1B class |
| **Holo3.1-4B** (Ref) | ~4.0B | Community | `llama-server` (GGUF) | Buildathon reference row (unified single-model baseline) |

---

## 2. Evaluation Methodology

### 2.1 Evaluated Tool Schemas
Evaluations test models directly against Grace's operational schema (`src/grace/intent/tools.py`):
1. **System Tools:**
   - `open_app(app_name: str)`
   - `close_app(app_name: str)`
   - `adjust_volume(direction: str, step: int = 10, level: Optional[int] = None)`
   - `lock_computer()`
   - `search_files(query: str)`
   - `open_file(path: str)`
   - `delete_file(path: str)`
   - `read_pdf(path: str)`
   - `summarize_pdf(path: str)`
   - `undo()`
   - `describe_screen()`
   - `set_speech_rate(rate: float)`
2. **Computer Use Actuator (CUA) Tools:**
   - `cua_click(element_id: int)`
   - `cua_type_text(text: str)`
   - `cua_press_key(key: str)`
   - `cua_scroll(direction: str, amount: int = 3)`
   - `cua_drag(from_id: int, to_id: int)`
   - `cua_set_value(element_id: int, value: str)`

### 2.2 Metrics
- **Exact Tool Accuracy:** Percentage of test utterances mapped to the correct tool identifier.
- **Arguments Match Accuracy:** Percentage of calls where both tool name AND arguments match ground-truth specifications.
- **JSON Validity Rate:** Percentage of outputs parseable into valid JSON without syntax or schema exceptions.
- **Latency (p50, p95):** Measured on CPU using the Task 08 central thread budget.

---

## 3. Model Acquisition & Placement Guide

Models must be acquired manually per Grace Hard Rule 2:

1. **FunctionGemma 270M (Gemma Terms of Use):**
   - Source: https://huggingface.co/google/functiongemma-270m-it
   - Target: `C:\Users\Ashman Das\Downloads\functiongemma-270m` (~540 MB)

2. **LFM2.5-350M (Open Weights):**
   - Source: https://huggingface.co/LiquidAI/LFM2.5-350M
   - Target: `C:\Users\Ashman Das\Downloads\lfm2.5-350m` (~700 MB)

3. **xLAM-2-1b-fc-r (Apache 2.0):**
   - Source: https://huggingface.co/Salesforce/xLAM-2-1b-fc-r
   - Target: `C:\Users\Ashman Das\Downloads\xlam-2-1b-fc-r` (~2.0 GB)

---

## 4. Benchmark Execution Command

```bash
# Run tool calling bake-off
./venv/Scripts/python.exe -m bench.run_tool_calling_benchmark

# Render Markdown report
./venv/Scripts/python.exe -m bench.report --benchmark-name tool_calling_benchmark --out docs/benchmarks/tool-calling-results.md
```

---

## 5. Standard Results Schema

Records are written to `bench/results/tool_calling_benchmark.jsonl`:

```json
{
  "benchmark_name": "tool_calling_benchmark",
  "model_id": "functiongemma-270m",
  "metrics": {
    "overall": {
      "exact_tool_accuracy": 0.962,
      "args_match_accuracy": 0.925,
      "json_validity_rate": 1.0,
      "total_samples": 40
    },
    "single_turn": {
      "exact_tool_accuracy": 0.970,
      "args_match_accuracy": 0.935,
      "total_samples": 33
    },
    "multi_turn": {
      "exact_tool_accuracy": 0.920,
      "args_match_accuracy": 0.880,
      "total_samples": 7
    },
    "latency_ms": {
      "p50": 18.5,
      "p95": 28.2,
      "mean": 19.4
    }
  }
}
```
