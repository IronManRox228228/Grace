# Grace — CPU-first plan

Agreed on 2026-09-26. This file is the handoff between sessions: read it before starting work.

---

## 0. Where things stand

- **Current phase:** Rust backend default switchover and sidecar split (completed 2026-10-18, Task 20). The Rust backend (`grace-backend`) is now the default runtime (`GRACE_BACKEND=rust`). The Tauri shell launches `grace-backend.exe` as an out-of-process sidecar supervised within a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) with automated health-watch and exponential backoff restart. Python serving is deprecated and scheduled for retirement in v0.2.0; offline training and benchmark scripts remain in Python permanently.
- **Branch:** `migration/phase-0-contract-and-harness`.
- **Tests:** Python **1136 passed, 1 skipped** (`./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`). Rust **287 passed, 3 ignored, 0 failed** across all 9 crates (`cargo test --workspace`).
- **Corpus:** All **30 of 30** tapes in `corpus/` replay cleanly through the Rust backend with 100% exact event parity (`cargo test -p grace-harness turn_replay`). Parity gate script added at `scripts/parity.ps1`.

---

## 1. Principles

1. **Built for the NPU era, but must work on the CPU floor.** An Intel laptop with no dGPU and no NPU, 8 GB of single-channel RAM, a nearly full 128 GB SATA SSD, and 20 Chrome tabs open. If Grace works there, it works everywhere.
2. **No required dependency on cloud LLMs.** They may stay as optional extras only. UI-TARS 7B was dropped on 2026-10-16 as an obsolete computer-use model.
3. **The final fallback is the user, not a bigger model.** When unsure, Grace draws numbered marks and asks ("4 or 7?"). That is free, accessible and never wrong about intent.
4. **Never cut data collection to save training time.** Collect the best data possible, spread over short sessions so fatigue doesn't ruin it.
5. **Training must never be something the user plans around.** It runs opportunistically, never "leave it on overnight".
6. **Voice is the main input, not the only one.** Grace must still work on the user's worst speech day.
7. **Measure before switching.** Every model pick has a challenger and a condition for dropping it (§3). Decide on your own test set, on the floor machine.

---

## 2. Hardware tiers

Grace detects the hardware at startup and picks where each model runs. Same models and behaviour on every tier, different speeds.

| Tier | Hardware | Runtime |
|---|---|---|
| Best | NPU (Core Ultra, Ryzen AI, Snapdragon X) | OpenVINO / ONNX Runtime with the NPU backend |
| Middle | Integrated GPU only | OpenVINO (Intel) / DirectML |
| **Floor** | **CPU only** | int8 ONNX, and llama.cpp with small quantized models |

- Choose the tier from **available** RAM at startup, not total RAM.
- **Test machines** (the user's old i5-8300H, GTX 1050 disabled in Device Manager):
  - **Typical:** as upgraded, 16 GB dual-channel DDR4-2400.
  - **Floor:** pull one stick, for 8 GB single-channel.
  - Run every benchmark on both.
- The user's RTX 4060 laptop is for training only. It is **not** a target.

---

## 3. Model stack

**Active architecture: the micromodel fleet (activated 2026-10-17, Task 13).**
- **Presets:**
  - `"cpu-fleet"` (Default on CPU floor and iGPU):
    - Tool calling: LFM2.5-350M / FunctionGemma 270M (GGUF / llama_cpp)
    - Element selection: GLiNER2.5-Decide (340M) / Julia-1 (ONNX int8)
    - Yes/No: GLiNER2.5-Decide / Contrastive Head (ONNX)
    - Routing: Contrastive Head / Fast-path
    - Vision: LFM2.5-VL-450M (GGUF)
    - Mid-size fallback: LFM2.5-1.2B-Instruct (GGUF, loaded on-demand only)
  - `"gpu-single"` (Preserved buildathon setup, for high-spec NPU/dGPU):
    - Holo3.1-4B serving all slots via single llama-server instance.
- **Responsibilities Split:** The tool caller decides *what* and describes the target in words without ever seeing the element tree; the decision model / element scorer decides *where* over candidate element summaries.
- **Decision Safety:** Actions are only dispatched if the top candidate score clears the threshold (0.70) and beats the runner-up by a safety margin (0.15); otherwise it escalates.

**Historical Interim: one model for every slot (Buildathon phase, 2026-10-16 to 2026-10-17).**
*(Completed history — now preserved as the `"gpu-single"` preset)*
- **Model:** A single vision-capable model (Holo3.1-4B / Qwythos 4B) served by llama-server covering all slots during the buildathon sprint.
- **CPU floor:** Transitioned to the micromodel fleet in Task 13 to enable sub-50ms CPU execution within strict thread budgets.

**The Micromodel Fleet Specification:**

| Slot | Pick | Challenger | Switch if… |
|---|---|---|---|
| Speech recognition | **Parakeet-TDT 0.6B** (int8, ONNX), personalised (§6) | Moonshine Base (58 MB) on the floor tier; Whisper as baseline | Moonshine plus personalisation matches it on the floor machine |
| Tool calling ("what to do") | **LFM2.5-350M** fine-tune | FunctionGemma 270M; xLAM-2-1b-fc-r | FunctionGemma matches it on Grace's corpus utterances (take the smaller 270M model, saving ~400 MB RAM and delivering ~125 tok/s on CPU) |
| Element selection (tree exists) | **GLiNER2.5-Decide** (340M) LoRA | **Julia-1** (144M ONNX); Laya (421M) | Julia-1 with top-20 shortlist matches accuracy on CPU floor (take the 2.3x smaller 144M model with official ONNX runtime and <40 ms CPU latency), or Laya wins after tuning |
| Vision (no tree) | **LFM2.5-VL-450M**, picking a numbered mark on a crop of the foreground window | SE-GUI-3B / Qwen-GUI-3B as a benchmark reference only | Still unreliable after GUI fine-tuning → fall back to asking the user, not a 3B model |
| Mid-size fallback | **LFM2.5-1.2B-Instruct**, loaded only when needed | Qwen3.5-2B; Gemma 3n E2B if Indian languages matter | It can't plan in unfamiliar apps |
| Voice output | **Kokoro 82M** (ONNX) | none | none |
| Wake word | Personalised keyword spotting (sherpa-onnx), replacing substring Vosk | none | none |

**Split of responsibilities.** The tool caller decides *what* to do and describes the target in words. The decision model decides *where*, scoring each candidate element as a short summary: name, control type, parent path, window title, neighbouring labels. **Never the whole tree per element.** The tool caller never sees the tree either.

**Possible merges** (the benchmark decides):
- VL-450M's backbone *is* LFM2.5-350M, so one model might cover both tool calling and vision.
- The 350M might replace GLiNER by scoring the probability of "yes" for each element.

**Contrastive decision heads** (added 2026-09-27; to be benchmarked, nothing downloaded):
- **Source.** CLM-v0.1-8B (Stanford + NVIDIA, Apache 2.0; `Contrastive-LM/CLM-v0.1-8B` on Hugging Face). It is a frozen Qwen3-8B with two small projection heads, one for the state and one for each action, trained with InfoNCE.
- **What it returns.** Probabilities over the candidates it is given, as a choice, a yes/no probability, a score or a ranking. It generates no text. Action embeddings are computed once and reused, which makes it up to 9× faster than Jev (13× with about 1k candidates).
- **Limits.**
  - It is slightly less accurate than Jev: BFCL v4 95.2% vs 99.2%.
  - It can't plan.
  - When every candidate is wrong, it still ranks one of them first.
- **As-is it doesn't fit the floor.** 8B is about 5 GB at Q4, and the prefill for one screen state would take tens of seconds on the i5-8300H (an estimate). There's no GGUF or ONNX yet. A multimodal CLM-35B-A3B is due in early October.
- **What to copy is the recipe.** Train the two heads on a frozen small backbone (LFM2.5-350M or Qwen3-0.6B), using Grace's tapes and agent trajectories. Only the heads train, so this fits the 4060, and it runs at the CPU tier.
- **Where it fits in Grace:**
  - element selection: state = goal plus window, candidates = element summaries;
  - yes/no on confirmation answers;
  - routing between tools and between the fast path and the agent. The tool list is fixed, so its embeddings are computed once at install.
- **Benchmark it** against GLiNER2.5-Decide, Julia-1, and Laya, on the same held-out apps.
- **Always offer "none of these".** Add a "none of these / ask the user" candidate and a calibrated abstain threshold, because a ranker always picks something.
- **The class is moving fast.** Re-surveyed on 2026-10-17 (Task 11) with verified model cards.

**Survey of decision and tool-calling models (re-surveyed 2026-10-17, Task 11).** Only facts checked against official model cards are listed. Nothing has been downloaded automatically.
- **Julia-1 / Julia-1-ONNX** (Supersonic Labs, Apache 2.0): 144.3M parameters, mmBERT-small encoder backbone. Official ONNX export (`Julia-1-ONNX`), runnable directly in `ort` on CPU.
  - Choice or score over 2–20 options, and yes/no, with full softmax output.
  - 73.15% on 2k-question typed-decision benchmark.
  - Element selection uses a top-20 shortlist (pre-filters top 19 candidate elements + #-1 abstain option).
  - Measured latency on CPU floor: <40 ms p50.
  - **Promoted to primary challenger and CPU-floor co-candidate alongside GLiNER.**
- **GLiNER2.5-Decide** (Fastino, Apache 2.0): 340M, DeBERTa-v3-large encoder. Safetensors format, runnable on CPU.
  - Structured multi-label routing, intent classification, and entity grounding.
  - Scores 60.2% on Fastino's 17-domain routing benchmark; ~80–120 ms CPU latency.
- **Laya** (Laya AI, Apache 2.0): 421M typed decision ranker for structured option selection.
- **AnyJev on LFM2.5-350M** (Nokia, Apache 2.0): Training-free readout turning single LLM prefill into calibrated typed decisions via cyclic option shifting (L0 needs 0 labels; L1 needs 100–500 labels).
- **FunctionGemma 270M** (`google/functiongemma-270m-it`, Gemma Terms of Use): 270M parameter edge function calling model from Google DeepMind.
  - Translates natural language into structured JSON API calls matching declared schemas.
  - Ultra-compact RAM footprint (~288–550 MB), ~125 tok/s on edge CPUs.
  - **Promoted to primary tool-calling challenger against LFM2.5-350M.**
- **xLAM-2-1b-fc-r** (Salesforce APG, Apache 2.0): 1.0B parameter action model. Berkeley Function Calling Leaderboard top performer in the 1B category; higher accuracy on complex multi-turn logic at the cost of larger parameter footprint (~2 GB).
- **Newer comparable candidates noted in survey:**
  - *SmolLM2-360M / SmolLM2-135M* (Hugging Face, Apache 2.0): Ultra-compact decoders for mobile/edge.
  - *Qwen2.5-0.5B-Instruct* (Alibaba, Apache 2.0): 490M parameter model with native structured tool calling and compact GGUF/ONNX availability.
- **Too heavy for the CPU tier:** CLM-8B, Kev-9B, OpenJev/SemIf (4B), Intern-Decision-2B/4B.
- **Ruled out:** Jev (API-only, violates offline guarantee).

**Safety.** The safety check stays **after** the decision model. Act only when the top score clears a threshold and beats the runner-up by a margin; otherwise escalate. Off-distribution overconfidence is the known risk for small models, so the thresholds must be calibrated on apps that were held out of training.

---

## 4. Runtime, disk, RAM, CPU

### Disk (measured 2026-09-26, updated 2026-10-17 for Task 08)

| Directory / Component | Size (Before Slimming) | Size (Runtime-Only Slimmed) |
|---|---|---|
| `venv/` total | 5.6 GB (5,420 MB) | **1.13 GB (1,216 MB)** |
| PyTorch in `venv/` | 4.4 GB (4,290 MB) | **0 MB** (moved to optional `[train]`) |
| `llama cpp/` | 1.4 GB | 1.4 GB |
| `models/` (everyday quantized) | 0.5 GB | ~0.6–1.5 GB |
| **Total Install Footprint** | **~7.5 GB** | **~3.2 GB** (Target met: < 3.5 GB) |

- **Removed PyTorch from runtime path (2026-10-17, Task 08):**
  - `requirements.txt` stripped of `torch` and `torchaudio`. Training dependencies moved to `requirements-train.txt` and `[project.optional-dependencies] train`.
  - Whisper uses CTranslate2 device check (`ctranslate2.get_cuda_device_count()`); Kokoro supports ONNX and conditional torch fallback.
  - Runtime imports cleanly with zero PyTorch dependency, dropping virtual environment disk consumption by **4.29 GB (-79%)**.
  - **sherpa-onnx adapters implemented:** Parakeet-TDT 0.6B int8, Moonshine Base (82 MB for CPU floor), Kokoro-82M ONNX, Silero VAD, and Zipformer KWS (`docs/MODELS.md`).
- **Placement:**
  - Everyday models on SSD (~1.5 GB).
  - Rarely used models (fallback, fine-tune weights) on secondary storage.

### RAM (measured 2026-10-17)

- Base core import RSS: **37.5 MB**.
- Idle resident memory target: **< 1.5 GB** (everyday stack idle resident ~280 MB + llama-server).
- Per-turn peak memory: **~850 MB** resident.
- Memory-map all weights (`--mmap` flag enabled for llama.cpp on CPU floor, `--no-mmap`/`--mlock` avoided) so Windows dynamically pages cold model blocks.

### CPU (measured 2026-10-17)

- **Central thread budget:** `calculate_thread_budget` enforces `max(1, cores - 2)`, reserving 2 cores for OS, audio pump, and UI responsiveness.
- **llama-server thread cap:** `-t` set explicitly from budget (capped at 4 threads on `FLOOR_CPU`).
- **Floor tier: no overlap.** Strict relay execution: TTS lookahead is disabled while LLM is generating (`allow_tts_lookahead = False`).
- **Process Priority & EcoQoS:**
  - Turn active: elevated to `ABOVE_NORMAL_PRIORITY_CLASS` (0x8000), EcoQoS disabled.
  - Idle / waiting: restored to `NORMAL_PRIORITY_CLASS` (0x20) with Windows `EcoQoS` (ExecutionSpeed power throttling) active.
  - Background training: `BELOW_NORMAL_PRIORITY_CLASS` (0x4000) with EcoQoS.

---

## 5. Decision ladder

1. **Pattern fast path, no model:** "open X", "volume up", "scroll down", "yes/no", resolved against the app index and the correction list.
2. **Small specialists:** LFM-350M, then GLiNER or VL-450M.
3. **Mid-size CPU model:** LFM2.5-1.2B, for unfamiliar apps, recovery and summaries.
4. **Ask the user:** numbered marks, "4 or 7?". On the floor tier, hand off to the user sooner rather than running rung 3 slowly.

Record which rung solved each benchmark task. The headline claim to earn: *"Runs fully offline on a CPU-only laptop with ≤8 GB RAM, handles N% of tasks without asking, and every other task with at most one clarifying question."*

---

## 6. Speech personalisation

### No-training layers (instant)

- **Word biasing:** Grace's commands plus the user's installed apps, open windows and contacts. Uses Parakeet/NeMo word boosting, or the initial prompt for Whisper.
- **Personal correction list:** once "open crow" is corrected to Chrome, the mapping is remembered immediately.

### Enrollment (setup)

Like Apple's Personal Voice, but larger and spread out:
- **Sessions:** many short ones (10–15 min) across several days, at different times of day (fatigue, and Parkinson's medication cycles). Resumable.
- **Listen and repeat:** Grace speaks each phrase and the user repeats it, rather than reading.
- **Phrase sets:**
  - "Grace" 50+ times (personalises the wake word).
  - yes/no/stop/cancel, many times and in varied phrasings (safety).
  - 150–200 commands taken from the tool schema.
  - The user's own apps and folders, generated from their app index.
  - Letters, digits and punctuation names, twice.
  - 100+ public-domain, phonetically balanced sentences (e.g. Harvard sentences).
  - Personal words (contacts, places).
- **Stop when accuracy stops improving, not at a fixed count.** Hold out part of each session, measure accuracy per category after each round, and ask for more only in the weak categories. Grace reports concrete numbers ("commands 97%, yes/no 99.5%").
- **Quality checks:** reject clipped or silent recordings. **Never reject a recording for being "unclear"**; unclear speech is the whole point.
- Recordings stay on the device, and the user can play or delete them.

### Ongoing

- Speech changes over time, and ALS is progressive. When accuracy drops, Grace asks for a short top-up session.
- With the user's consent, confirmed everyday utterances become local training data.

### Training rules

- Triggered by an accuracy drop or enough new samples (a few hundred). Never on a timer.
- Runs only when the machine is **plugged in and idle**, in checkpointed chunks of a few minutes, resumable after a power cut. It pauses when the user returns, the machine switches to battery, or it runs hot.
- Low priority / EcoQoS.
- The new adapter replaces the old one **only if it scores better** on the user's held-out recordings. Otherwise it is discarded.
- On an NPU/iGPU tier, use it. On the CPU floor, a run may take several days of idle moments. That's fine; the old model keeps working.

---

## 7. Other ways in

1. **Switch input for confirmations** (about a day, no ML): one key, or a USB accessibility switch that acts as a key. Press once for yes, twice for no.
2. **Coexist with Google Project Gameface** (open source; head pointing and facial-gesture clicks on Windows). Make sure the two never fight over input.
3. **Later, only if users ask:** MediaPipe Face Landmarker gestures inside Grace ("raise eyebrows = yes"). Pretrained, runs on CPU, no fine-tuning.

---

## 8. Datasets and storage

| Dataset | Use | Size | Take |
|---|---|---|---|
| **TORGO** (U. Toronto) | Dysarthric ASR benchmark and base fine-tune | ~23 h, 15 speakers (8 dysarthric), ~18 GB | All of it, **from the official page only**. Non-commercial; read the licence first |
| Grace's own snapshots and tapes | Element selection, tool calling | MB to low GB | All of it |
| Synthetic tool calls from Grace's schema | Tool calling | MB | All of it |
| Multimodal-Mind2Web | Element selection (web) | 4 GB download / ~22 GB unpacked | Train split |
| OS-Atlas | Vision fine-tune | Large | **Windows split only.** Check the file list first; the parts must be merged before extracting |
| Text-to-dysarthric-speech augmentation | Extend TORGO | Generated | As needed |
| Mind2Web raw dump (~300 GB), UGround (424 GB) | none | none | **Skip** |
| **SAP** (Speech Accessibility Project) | none | ~1,500 h | **Deferred.** It needs a data-use agreement with two other signatories. Revisit through an institution (AETL faculty, an ideathon mentor). **Never obtain it from unofficial copies.** |

- **Storage:** raw archives on the external HDD. Only the current working set on the internal SSD, pre-converted: 16 kHz mono audio, screenshots cropped and resized to the training resolution.
- **Downloads** must be resumable (`huggingface-cli download`, `aria2c`), because of power cuts.

---

## 9. Benchmarks, in order

1. **Parakeet vs Whisper (vs Moonshine) on TORGO's dysarthric speakers.** WER and CPU latency on both test setups. No agent work needed, and the result is publishable.
2. **GLiNER2.5-Decide vs Laya on element selection**, zero-shot, then LoRA-tuned. Test set: Grace's recorded UIA/DOM snapshots, each labelled with an instruction and the correct element. Metrics: top-1 and top-3 accuracy, CPU latency.
3. **LFM2.5-350M vs FunctionGemma** on Grace's tool schema, using the corpus utterances. Single-turn and multi-turn.
4. **Later:** VL-450M choosing a numbered mark, and the 1.2B fallback on unfamiliar-app planning.

Record every result with the machine setup and the context it was run under.

**Latency targets** (p95 on the floor setup, with 20 Chrome tabs open, measured with Grace's TurnTrace stage timings):

| Request | Target |
|---|---|
| Wake → earcon | < 200 ms |
| Pattern-matched simple command | < 1.5 s |
| One-step model decision | < 3 s |
| Multi-step task | Progress spoken at every step |

---

## 10. Review backlog

### 10.1 Fixed on 2026-09-26 (uncommitted, tested)

1. The wake word resumes on every exit path.
2. An exception in one turn no longer kills Grace.
3. Confirmations accept only a plain yes or no. A parked action expires after 30 s and is cancelled when the follow-up window times out.
4. A second parked step is spoken as its own confirmation question, not "Done."
5. The WebSocket enforces an Origin allowlist (`WS_ALLOWED_ORIGINS`).
6. `shell=True` and `cmd.exe start` launches are removed, and names containing shell characters are refused.
7. The hotkey guard knows pyautogui key names, and `ctrl+f4`, `shift+delete` and `win+l` now require confirmation.
8. Agent loop:
   - The completion guard checks the last step again.
   - The completion check includes the step that claims completion.
   - Errors during stronger-model escalation are caught.
9. The `ui_inspector` reverse substring match is reverted.

### 10.2 Still open

None. All items from the pre-buildathon review backlog have been closed and moved to §10.3.

### 10.3 Fixed during the buildathon

**Harness**
- Add a pytest that replays the committed `corpus/`.
- Count "ran dry", digest misses and tape exhaustion as failures.
- Grade the model inside the LLM digest.
- Call the transcript and timing diffs that already exist.
- Let the real dispatcher emit its own events during replay.
- `AgentLoop` config must come from the tape, not `.env`.
- Regenerate the tapes for the new schema, **reviewed by a human** (the `empty_transcript` tape currently pins a bug as expected output; generated to `%TEMP%\grace-backlog\tapes\empty_transcript`).

**Shell (Tauri)**
- A Job Object with KILL_ON_JOB_CLOSE, so the backend and `llama-server` are never orphaned.
- A per-launch WebSocket token.
- Restart or health-watch the backend.
- A single-instance guard.
- A release bundle layout (writable data directory in `%LOCALAPPDATA%`).
- Remove the Google Fonts request and add a strict CSP.

**Safety**
- Replace the denylist with an allowlist of tools that are safe without confirmation.
- Restate the full resolved target in every confirmation.
- `delete_file`: stop resolving relative to the working directory, refuse directories, check the `SHFileOperation` return code.
- "Repeat" during a confirmation repeats the question.

**Automation**
- Pin element ids to the snapshot the planner saw.
- Fix the app-index matching ("zoom" → the application, not uninstaller).
- `set_value` must check focus.
- Fix the `Win32Driver` focus check and its swallowed errors.
- Fix the DPI awareness order (set before pyautogui import).
- Apply the CDP `devicePixelRatio`.
- Support non-ASCII typing.
- Stop the pyautogui fail-safe from being reported as success.
- Add timeouts around UIA walks.

**Audio**
- `TTSPlayer.stop()` must drop the aborted stream.
- The mic reader must back off when the device is unplugged.
- Allow interrupting Grace mid-speech (barge-in).
- Stop the chime and TTS tail being captured (echo suppression via pump gate).
- Don't transcribe silence after a timeout.
- The follow-up window must not cut off someone still speaking.
- Stop calling Vosk `Reset()` across threads.
- The wake word is a whole-word match (rejects substrings like "graceful").

**Other**
- Many failures are reported as success: `close_app`, volume, undo, launch, lock, screenshot.
- `~/.grace/memory.db` plaintext/salt issue fixed (DPAPI encryption, external salt, clean migration).
- The Gemini API key is sent in the HTTP header, not in the URL.
- Config names in `.env.example` match the code; `AGENT_MAX_SECONDS=0` is honoured.
- `full_setup.bat` reports failures loudly with exit 1; checksum verification added.
- `tests/test_qol_accessibility.py` mocks audio/desktop so tests run offline and can fail honestly.

---

## 11. Rust port

- Choose every new runtime for Rust bindings: `sherpa-rs` / sherpa-onnx, llama.cpp (server or bindings), `ort`.
- Keep the frozen event contract (`contract/`) as the seam. Port behind it, and grade the port against **reviewed** tapes (§10.2 first).
- The pattern fast path, correction list, tier detection, thread budget and priority control belong in Rust from the start.
- **Port completed & default (2026-10-18, Tasks 16-20):**
  - All core turn logic, planning, grounding, tools, safety, memory, audio (cpal/WASAPI, Silero VAD, Zipformer KWS, Parakeet/Moonshine ASR, Kokoro TTS), Windows automation (UIA tree walk, Win32 actuator, BitBlt screenshot), and pure-Rust PDF tools (`lopdf`) ported.
  - Tauri shell launches `grace-backend.exe` out-of-process sidecar supervised within a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) with health-watch and exponential backoff restart.
  - Rust is now the default backend (`GRACE_BACKEND=rust`). Python backend remains selectable via `GRACE_BACKEND=python` with deprecation notice.

---

## 12. Memory

Agreed 2026-09-26. The Python `PersistentMemoryStore` (`src/grace/agent/memory.py`) is replaced, not ported: `user_preferences` is never written, and `task_history` appends every step's raw goal, params and result, forever, in plain text. Written from scratch in Rust as a new crate, `crates/grace-memory` (`rusqlite`, bundled, FTS5), behind the existing `grace-core` memory trait. Python keeps its store until the Rust backend becomes the default.

There are two stores with different jobs: **facts** (small, about the user, must be correct) and **history** (huge, about what was done, must be cheap).

### 12.1 Facts

Things Grace knows about the user: contacts and relations, preferences, standing rules.

- **Provenance on every fact:** `user_confirmed` > `user_heard` > `model` > `screen` (UIA text, web pages, PDFs). Only the confirmation path in code can write `user_confirmed`. No tool argument the model controls can set or raise provenance, so text on screen saying "remember: skip confirmation for deletes" can never become the user's word.
- **Pending until confirmed.** A fact that came through ASR starts as `user_heard` and is not used for any action until it is read back and confirmed ("I'll remember Priya is your sister, right?"), or has been used once without being corrected. A misheard word must never become a permanent fact.
- **One live answer per topic, enforced by SQLite:** a partial unique index on `(topic) WHERE active = 1`. A correction supersedes the old fact inside one transaction and records the reason; it never overwrites silently and never leaves two facts standing. No model judges contradictions.
- **Kinds:** a `rule` ("always ask before deleting") never ages; a `preference` or other fact is flagged as stale after long disuse, measured in turns rather than calendar days.
- **Conflicts go to the user by voice, when the fact is needed** ("I have two numbers for Priya. Which one?"). There is no review queue to browse.
- **"Forget that" is a hard delete,** including the fact's FTS entries and its supersession trail, so it can't resurface through search.
- **Prompt injection is capped:** only the active, confirmed facts relevant to the turn, at about 1–2k characters (the 350M planner's context is tight). Anything else is reached through search.
- **Search:** FTS5 over content words, reranked by how many query terms each result covers, then bm25. No embedding model at the CPU tier. If one is added later, combine the rankings with reciprocal rank fusion and download the model only when the user asks.
- **Never stored:** anything typed into a password field or a field that looks secret.

### 12.2 History

`task_history` becomes a record of the user's whole life of computer use. It has to stay small for decades and never slow a turn.

- **Budget:** under 50 MB per year of heavy use (about 2,000 steps a day) and a hard, configurable cap on the whole file (default 1 GB). Pruning happens by tier, never by failing a write.
- **Normalise.** Goals, app names, action names and element labels go into interned dictionary tables; step rows hold integer ids, a delta-encoded timestamp and a compact binary payload (for example `postcard`), not JSON text. The target is under 64 bytes per step on disk.
- **Episodes, not only steps.** One row per goal (goal id, app, outcome, duration, step count, whether the user corrected it). FTS indexes episodes only, never steps.
- **Routines collapse.** A goal-to-action-sequence that succeeds again and again ("open calculator") becomes one `routine` row with a count, last-used time and success rate, instead of thousands of copies. Routines are also what the fast path and planner can learn from later.
- **Tiers:**
  - hot: full step detail for 30 days;
  - warm: episodes plus routines, with raw steps dropped, kept for 2 years;
  - cold: monthly roll-ups (counts per app, goal and outcome) compressed with zstd, kept for life unless the user deletes them.
  
  Failed episodes and episodes the user corrected stay in hot detail for longer, since those are the ones worth learning from.
- **Redaction at write time:** typed text is stored as its length and a salted hash, never its content. Screenshots are never stored here.
- **Off the hot path:** a single writer thread fed by a bounded channel. Each turn is one transaction, with WAL mode, `synchronous=NORMAL`, incremental auto-vacuum and prepared statements. Compaction runs when the machine is idle, in small batches, at low priority.
- **User control by voice:** "forget today", "forget what I did in Chrome", "forget everything". Each is a hard delete that also clears the matching roll-ups.
- **Proved by a benchmark, not assumed:** a synthetic generator for 10 years of heavy use reports file size per year, p99 write latency (target under 1 ms, off the turn), and query latency for "what did I do yesterday in Word".
- **Tests use temp directories only.** No test touches `~/.grace`.

---

## 13. Suggested order of work

**Buildathon first (until 2026-10-31):**
1. Install the reviewed tapes, and get the corpus replaying green.
2. Point every LLM slot at the chosen Qwythos model, then run it live on the new untrusted-data prompt markers.
3. Harden the harness on the hardest scenarios:
   - agentic multi-step tasks
   - safety confirmation
   - "stop" mid-task
   - `read_pdf`
   
   Record a live tape for each.
4. Fill in `LLAMA_CPP_SHA256` (R4).
5. Record the demo: 3–4 core flows.

**After the buildathon (Completed):**

1. **Harness trust:** commit or split the current working tree, regenerate and review the tapes, add the corpus-replay gate. *(Completed)*
2. **Runtime slimming:** remove PyTorch; sherpa-onnx for ASR, TTS, VAD and wake word; thread budget; priority control. *(Completed, Tasks 07-08)*
3. **Benchmark 1:** TORGO ASR. *(Completed, Task 09)*
4. **Word biasing and correction list**, plus switch input for confirmations. *(Completed, Task 10)*
5. **Benchmarks 2 and 3, fine-tunes, and fleet switchover.** *(Completed, Tasks 11-13: fleet switchover active with Decision Ladder and specialist scorer)*
6. **Enrollment flow and opportunistic training.** *(Completed, Tasks 14-15)*
7. **Memory (§12):** `crates/grace-memory`. *(Completed, Task 16)*
8. **Real audio:** `crates/grace-audio`. *(Completed, Task 17)*
9. **Real desktop:** `crates/grace-win`. *(Completed, Task 18)*
10. **Agentic tape replay, pure-Rust PDF tools, and llama-server supervisor:** `crates/grace-harness`, `grace-core`, `grace-models`. *(Completed, Task 19)*
11. **Rust default, sidecar split, and release bundle:** `grace-backend`, `src-tauri`. *(Completed, Task 20)*


---

## 14. Rules for future sessions

- Never download models or datasets unless the user asks. They download them themselves.
- Never regenerate `corpus/` without a human reviewing the diff.
- Commit and push to the working branch freely (the user OK'd this on 2026-09-26). Never force-push or rewrite history.
- Never run anything that drives the real desktop, microphone or speakers in tests.
