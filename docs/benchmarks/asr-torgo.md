# Benchmark 1: Automatic Speech Recognition on TORGO

Evaluation of lightweight, offline ASR models for dysarthric speech recognition under Grace's resource and latency constraints (PLAN.md §8, §9).

---

## 1. Executive Summary & Status

> [!IMPORTANT]
> **Dataset & Model Download Notice:**
> In compliance with Grace Hard Rule 2 (*never download models or datasets automatically*), the benchmark harness and evaluation pipeline are fully implemented and verified via synthetic fixtures, but **unexecuted on the full dataset** pending manual download of the TORGO corpus and model weights by the user.
> 
> See [§3. Dataset Acquisition & Download Guide](#3-dataset-acquisition--download-guide) below for exact download instructions, licensing, and placement.

### Evaluated Model Matrix

| Model | Architecture | Parameters | Target Runtime | Primary Advantage |
| :--- | :--- | :---: | :--- | :--- |
| **Whisper Base** | Encoder-Decoder Transformer | ~74M | `faster-whisper` (CTranslate2 int8) | High zero-shot robustness, robust multilingual baseline |
| **Parakeet-TDT 0.6B** | FastConformer-Transducer | ~600M | `sherpa-onnx` (int8/fp32) | Token-and-Duration Transducer, 4x faster decoding on CPU |
| **Moonshine Base** | Encoder-Decoder Transformer | ~61M | `sherpa-onnx` / ONNX Runtime | Designed for resource-constrained edge devices and short commands |

---

## 2. Evaluation Methodology

### 2.1 Acoustic Preprocessing & Standardisation
- **Source Audio:** TORGO head-mounted microphone (`wav_headMic`) recordings preferred due to higher signal-to-noise ratio compared to circular array microphones (`wav_arrayMic`).
- **Resampling:** All audio is converted on demand to 16 kHz, 16-bit mono PCM cached in `.cache_16k/` using high-fidelity band-limited polynomial resampling (`scipy.signal.resample`).

### 2.2 Exclusions & Dataset Filtering
To ensure meaningful acoustic benchmarking, prompts that do not reflect read speech transcripts are explicitly excluded:
1. **Image Description Tasks:** TORGO includes tasks where participants were shown spontaneous images (e.g. `MC` image tasks, prompts referencing `.jpg` / `describe the picture`). Because the prompt text does not match what was actually spoken, evaluating ASR output against the prompt yields artificial 100% error rates.
2. **Non-Speech Acoustic Tokens:** Prompts consisting solely of non-speech artifacts (`[cough]`, `[laughter]`, `[sigh]`, `[applause]`, `[snort]`, `xxx`) or empty strings are excluded.
3. **Missing / Corrupted Audio:** Prompts with zero-byte or missing WAV counterparts are omitted.

### 2.3 Fair Text Normalisation
Text normalisation is applied identically to both the reference prompt and the hypothesis transcript across all models (`bench/normalizer.py`):
1. **Unicode Canonical Decomposition:** NFKC normalization.
2. **Case Normalisation:** Complete lowercasing.
3. **Artifact Removal:** Stripping transcriber bracketed tags (`[cough]`, `<applause>`).
4. **Number Expansion:** Conversion of isolated digits 0–20 to English cardinal words (e.g. `2` → `two`).
5. **Punctuation Stripping:** Complete removal of ASCII and typographic punctuation characters (`.,!?:;"'~@#$%^&*()+=<>{}[\]|\\`).
6. **Whitespace Compression:** Collapsing contiguous spaces into single space tokens.

### 2.4 Resource & Thread Management
Evaluations strictly abide by Grace Task 08 resource policies:
- **Central Thread Budget:** CPU worker threads are capped at `max(1, cores - 2)` (reserving 2 cores for the operating system and user interface).
- **llama / ONNX Intra-Op Budget:** Capped at 4 worker threads on `FLOOR_CPU` tier to prevent cache thrashing.
- **Multitasking Benchmark (`--simulate-tabs 20`):** Simulates 20 open Chrome tabs (allocating controllable memory and periodic JavaScript-like CPU ticks via `bench/load_generator.py`) to measure latency under typical real-world desktop stress.

---

## 3. Dataset Acquisition & Download Guide

TORGO is distributed by the University of Toronto under a **non-commercial research licence**. It must be downloaded directly from the official repository:

1. **Official Web Page:**  
   http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

2. **Required Speaker Archives:**  
   Download the speaker `.tar.bz2` or `.zip` archives:
   - **Dysarthric Participants (8 speakers):** `F01`, `F03`, `F04`, `M01`, `M02`, `M03`, `M04`, `M05`
   - **Control Participants (7 speakers):** `FC01`, `FC02`, `FC03`, `MC01`, `MC02`, `MC03`, `MC04`

3. **Extraction Target:**  
   Extract all speakers into a single directory, structured as:
   ```text
   C:\Users\Ashman Das\Downloads\TORGO\
   ├── F01\
   │   └── Session1\
   │       ├── prompts\
   │       └── wav_headMic\
   ├── M01\
   │   └── ...
   └── FC01\
       └── ...
   ```

4. **Model Weights Placement:**
   - **Whisper Base:** Local directory or Hugging Face cache `Systran/faster-whisper-base`
   - **Parakeet-TDT 0.6B:** `C:\Users\Ashman Das\Downloads\sherpa-onnx-nemo-parakeet-tdt-0.6b-en-3000`
   - **Moonshine Base:** `C:\Users\Ashman Das\Downloads\moonshine-base`

---

## 4. Benchmark Execution Command

Once the dataset and models are placed, run the benchmark runner:

```bash
# Standard evaluation under idle conditions
./venv/Scripts/python.exe -m bench.run_asr_torgo --torgo-dir "C:\Users\Ashman Das\Downloads\TORGO"

# Multitasking evaluation simulating 20 open Chrome tabs
./venv/Scripts/python.exe -m bench.run_asr_torgo --torgo-dir "C:\Users\Ashman Das\Downloads\TORGO" --simulate-tabs 20

# Generate Markdown results table
./venv/Scripts/python.exe -m bench.report --benchmark-name asr_torgo --out docs/benchmarks/asr-torgo-results.md
```

---

## 5. Standard Results Schema

Results are written atomically to `bench/results/asr_torgo.jsonl`. Below is the schema generated per evaluation run:

```json
{
  "benchmark_name": "asr_torgo",
  "model_id": "whisper-base",
  "file_hash": "a1b2c3d4...",
  "runtime": {
    "python": "3.10.11",
    "ctranslate2": "4.5.0",
    "faster_whisper": "1.1.1",
    "sherpa_onnx": "not installed"
  },
  "machine_profile": {
    "os": "Windows 11 ...",
    "cpu": "Intel(R) Core(TM) ...",
    "cpu_count_physical": 14,
    "cpu_count_logical": 20,
    "ram_total_gb": 23.7,
    "ram_available_gb": 11.7,
    "memory_channels": "2 populated slots",
    "gpu_available": false,
    "gpu_disabled": true,
    "tier_chosen": "cpu",
    "background_load": "idle"
  },
  "thread_settings": {
    "total_cores": 20,
    "active_worker_threads": 18,
    "model_threads": 4,
    "intra_op_threads": 4
  },
  "timestamp": "2026-10-17T15:30:00Z",
  "git_commit": "1f280ea0b0ceca1a1ea50da0d5779d0b5c5d5c64",
  "metrics": {
    "evaluated_utterances": 1420,
    "overall_wer": 0.284,
    "dysarthric_wer": 0.382,
    "control_wer": 0.081,
    "wer_by_severity": {
      "Control": 0.081,
      "Mild": 0.194,
      "Moderate": 0.362,
      "Severe": 0.591
    },
    "mean_rtf": 0.185,
    "latency_p50_ms": 312.4,
    "latency_p95_ms": 684.2
  }
}
```

---

## 6. Caveats & Clinical Considerations

1. **Acoustic Diversity & Articulatory Imprecision:** Dysarthric speech exhibits variable vowel prolongation, hypernasality, consonant imprecision, and irregular pauses. Conventional VADs frequently truncate utterances during prolonged pauses; VAD filtering is intentionally disabled during TORGO evaluation.
2. **Microphone Placement Sensitivity:** Head-mounted microphones capture consistent proximity audio, whereas desk/array microphones capture room reverberation that compounds articulation challenges.
3. **Impact on Accessibility Commands:** In Grace, full sentence transcription is rarely required for everyday navigation; commands ("open browser", "scroll down", "yes", "stop") are shorter and benefit dramatically from Task 10's personalized word biasing and phoneme correction layers.
