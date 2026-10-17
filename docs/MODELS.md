# Grace — Model Weight Catalog & Download Guide (PLAN.md §3, §4, Task 08)

Grace is designed to run completely offline. In accordance with Grace's hard rules, models are never automatically downloaded at runtime. This document provides the official sources, file names, expected sizes, and destination paths for the runtime slimming and micromodel fleet engines.

---

## 1. Speech Recognition (ASR) Models

### Parakeet-TDT 0.6B int8 (Primary ASR Pick)
- **Role:** High-accuracy offline speech-to-text for default and high tiers.
- **Source:** [k2-fsa/sherpa-onnx-nemo-parakeet-tdt-ctc-0.6b-en-int8](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) / [Hugging Face](https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-ctc-0.6b-en-int8)
- **Expected Files:**
  - `model.int8.onnx` (~620 MB)
  - `tokens.txt` (~25 KB)
- **Target Directory:** `models/sherpa-onnx/sherpa-onnx-nemo-parakeet-tdt-ctc-0.6b-en-int8/`
- **Config Key:** `SHERPA_PARAKEET_MODEL_PATH`

### Moonshine Base int8 (CPU Floor Tier ASR)
- **Role:** Ultra-low-resource offline speech-to-text optimized for Intel Core i5 laptops with 8 GB RAM.
- **Source:** [k2-fsa/sherpa-onnx-moonshine-base-en-int8](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) / [Hugging Face](https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8)
- **Expected Files:**
  - `preprocess.onnx` (~1.2 MB)
  - `encode.int8.onnx` (~35 MB)
  - `uncached_decode.int8.onnx` (~22 MB)
  - `cached_decode.int8.onnx` (~24 MB)
  - `tokens.txt` (~30 KB)
- **Total Size:** ~82 MB
- **Target Directory:** `models/sherpa-onnx/sherpa-onnx-moonshine-base-en-int8/`
- **Config Key:** `SHERPA_MOONSHINE_MODEL_PATH`

### Faster-Whisper Base.en (Interim ASR Baseline)
- **Role:** Existing offline baseline ASR engine.
- **Source:** `Systran/faster-whisper-base.en` via CTranslate2
- **Expected Size:** ~140 MB
- **Target Directory:** `models/faster-whisper/base.en/` or standard huggingface cache.
- **Config Key:** `WHISPER_MODEL_PATH`

---

## 2. Text-to-Speech (TTS) Models

### Kokoro-82M ONNX (Sherpa-ONNX Engine)
- **Role:** Lightweight, low-latency offline voice synthesis without PyTorch dependency.
- **Source:** [k2-fsa/sherpa-onnx-kokoro-en-v0_19](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) / [Hugging Face](https://huggingface.co/csukuangfj/sherpa-onnx-kokoro-en-v0_19)
- **Expected Files:**
  - `model.onnx` (~310 MB)
  - `voices.bin` (~21 MB)
  - `tokens.txt` (~10 KB)
  - `espeak-ng-data/` directory (phonemizer data, ~5 MB)
- **Total Size:** ~336 MB
- **Target Directory:** `models/sherpa-onnx/kokoro-en-v0_19/`
- **Config Key:** `SHERPA_KOKORO_MODEL_PATH`

---

## 3. Voice Activity Detection (VAD) Models

### Silero VAD v5 ONNX
- **Role:** Real-time speech boundary detection for natural turn-taking.
- **Source:** [snakers4/silero-vad](https://github.com/snakers4/silero-vad) / [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
- **Expected Files:** `silero_vad.onnx`
- **Size:** ~2.1 MB
- **Target File:** `models/sherpa-onnx/silero_vad.onnx`
- **Config Key:** `SHERPA_SILERO_VAD_MODEL_PATH`

---

## 4. Keyword Spotting (Wake Word) Models

### Sherpa-ONNX Zipformer KWS (English)
- **Role:** Offline keyword spotting replacing Vosk substring matching for "Grace".
- **Source:** [k2-fsa/sherpa-onnx-kws-zipformer-en-2023-06-26](https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models)
- **Expected Files:**
  - `encoder.onnx` (~45 MB)
  - `decoder.onnx` (~12 MB)
  - `joiner.onnx` (~15 MB)
  - `tokens.txt` (~10 KB)
  - `keywords.txt` (defines the keyword phrase "Grace")
- **Total Size:** ~72 MB
- **Target Directory:** `models/sherpa-onnx/sherpa-onnx-kws-zipformer-en/`
- **Config Key:** `SHERPA_KWS_MODEL_PATH`

### Vosk Small English Model (Current Baseline)
- **Role:** Offline baseline wake-word and fallback keyword detector.
- **Source:** `alphacep/vosk-model-small-en-us-0.15` (AlphaCephei)
- **Size:** ~40 MB
- **Target Directory:** `models/vosk-model-small-en-us-0.15/`
- **Config Key:** `VOSK_MODEL_PATH`

---

## 5. Large Language & Vision Models (Buildathon Interim)

### Holo3.1-4B-Instruct (GGUF)
- **Role:** Unified local LLM for intent routing, multi-step planning, tool calling, and vision grounding.
- **Source:** [Holo-AI/Holo3.1-4B-Instruct-GGUF](https://huggingface.co/Holo-AI/Holo3.1-4B-Instruct-GGUF)
- **Expected Files:**
  - `Holo3.1-4B-Instruct-Q4_K_M.gguf` (~2.5 GB)
  - `mmproj-Holo3.1-4B-Instruct-f16.gguf` (~600 MB)
- **Target Directory:** `models/llm/` or user `Downloads/` directory referenced in `.env`.
- **Config Keys:** `LLAMA_MODEL_PATH`, `LLAMA_MMPROJ_PATH`

---

## Summary of Install Sizes

| Component | Legacy Runtime | Slimmed Sherpa-ONNX Stack |
|---|---|---|
| **Python Virtual Environment** | 5.42 GB (with PyTorch CUDA) | **1.13 GB** (without PyTorch) |
| **ASR Models** | ~140 MB (Whisper base) | **82 MB** (Moonshine) / **620 MB** (Parakeet) |
| **TTS Models** | ~320 MB (Kokoro pt) | **336 MB** (Kokoro ONNX) |
| **Wake + VAD Models** | ~42 MB (Vosk) | **74 MB** (Zipformer KWS + Silero) |
| **Total Model Weights (Everyday)** | ~3.0 GB (with 4B LLM) | **~3.4 GB** (with 4B LLM) |
| **Target Combined Disk Footprint** | ~9.8 GB | **< 4.8 GB** |
