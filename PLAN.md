# Grace — CPU-first plan

Agreed on 2026-09-26. This file is the handoff between sessions: read it before starting work.

---

## 0. Where things stand

- **Branch:** `migration/phase-0-contract-and-harness`. The Python → Rust/Tauri end-to-end port was already in progress before this plan; everything below assumes it continues.
- **Committed and pushed (2026-09-26):** the user's QoL/accessibility batch plus the nine ship-blocker fixes (`3c7b990`), the 3 s minimum from wake word to idle on a silent activation (`4f108c6`), and phase 2 of the Rust port (`470c51b`; status in `PORT_STATUS.md`).
- **Tests:** Python 841 passed, 1 skipped (`./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`). Rust 200 passed, 3 ignored (`cargo test --workspace`).
- **Corpus:** the committed tapes predate the batch's tool-schema change, so they no longer replay. All 30 were regenerated and the user approved the diff on 2026-09-26: three new tool descriptions in every prompt; `empty_transcript` loses its dead pause and 0 s follow-up window (then held to 3 s); `safety_press_key_confirm` parks Alt+F4 before the planner call. Copying them into `corpus/` is still pending. Tapes for the new features (new tools, confirmation expiry and timeout, mumbled answers, turn-crash recovery) are approved to write but not yet written.

---

## 1. Principles

1. **Built for the NPU era, but must work on the CPU floor.** An Intel laptop with no dGPU and no NPU, 8 GB of single-channel RAM, a nearly full 128 GB SATA SSD, and 20 Chrome tabs open. If Grace works there, it works everywhere.
2. **No required dependency on cloud LLMs or UI-TARS 7B.** They may stay as optional extras only.
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

| Slot | Pick | Challenger | Switch if… |
|---|---|---|---|
| Speech recognition | **Parakeet-TDT 0.6B** (int8, ONNX), personalised (§6) | Moonshine Base (58 MB) on the floor tier; Whisper as baseline | Moonshine plus personalisation matches it on the floor machine |
| Tool calling ("what to do") | **LFM2.5-350M** fine-tune | FunctionGemma 270M; xLAM-2-1b-fc-r | FunctionGemma matches it on Grace's corpus utterances (take the smaller) |
| Element selection (tree exists) | **GLiNER2.5-Decide** (340M) LoRA | Laya (421M), which is fine-tuned for Kerfd anyway | Laya wins after both are tuned on the same Grace snapshots |
| Vision (no tree) | **LFM2.5-VL-450M**, picking a numbered mark on a crop of the foreground window | SE-GUI-3B / Qwen-GUI-3B as a benchmark reference only | Still unreliable after GUI fine-tuning → fall back to asking the user, not a 3B model |
| Mid-size fallback | **LFM2.5-1.2B-Instruct**, loaded only when needed | Qwen3.5-2B; Gemma 3n E2B if Indian languages matter | It can't plan in unfamiliar apps |
| Voice output | **Kokoro 82M** (ONNX) | none | none |
| Wake word | Personalised keyword spotting (sherpa-onnx), replacing substring Vosk | none | none |

**Split of responsibilities.** The tool caller decides *what* to do and describes the target in words. The decision model decides *where*, scoring each candidate element as a short summary: name, control type, parent path, window title, neighbouring labels. **Never the whole tree per element.** The tool caller never sees the tree either.

**Possible merges** (the benchmark decides):
- VL-450M's backbone *is* LFM2.5-350M, so one model might cover both tool calling and vision.
- The 350M might replace GLiNER by scoring the probability of "yes" for each element.

**Safety.** The safety check stays **after** the decision model. Act only when the top score clears a threshold and beats the runner-up by a margin; otherwise escalate. Off-distribution overconfidence is the known risk for small models, so the thresholds must be calibrated on apps that were held out of training.

---

## 4. Runtime, disk, RAM, CPU

### Disk (measured 2026-09-26)

| Directory | Size |
|---|---|
| `venv/` | 5.6 GB, of which **4.4 GB is PyTorch** (the CUDA build) |
| `llama cpp/` | 1.4 GB |
| `models/` | 0.5 GB |

The whole new model stack, quantized, is roughly 2.5–3 GB (estimate).

- **Remove PyTorch from the runtime.** It is used only on the dev machine, for training.
  - **sherpa-onnx:** Parakeet, Moonshine, Kokoro, Silero VAD, keyword spotting. C++ core with Rust bindings, so it fits the port.
  - **llama.cpp:** the LFM models.
  - **ONNX Runtime** (`ort` crate in Rust): GLiNER.
- **Target install:** about 3 GB, down from over 7 GB.
- **Placement:**
  - Everyday models on the SSD (~1.5 GB).
  - Rarely used ones (VL-450M, the fallback) may live on the HDD: 5–10 s to load, acceptable for rare turns.
  - The fallback model is an optional download.

### RAM

- Everyday models stay loaded: under **1.5 GB** resident target.
- Everything else is loaded only when needed.
- Memory-map all weights, so Windows pages out cold ones instead of anything being killed.

### CPU

- **Thread budget:** cap every runtime explicitly (`intra_op_num_threads`, `-t`). The active models together must never exceed the core count, with one or two cores left for Windows, Gameface and the UI. Default settings (every runtime spawning one thread per core) is the real killer.
- **Only one or two models are ever active at once.** A turn is a relay: ASR → tool call → element scoring → action → TTS. Idle cost is near zero: only the wake word, plus Gameface if the user runs it.
- **Floor tier: no overlap.** Turn off TTS lookahead while the LLM is still streaming, and do one thing at a time.
- **Priority:** above-normal during a turn, below-normal when idle (no admin needed). Background training uses Windows EcoQoS.

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

**Harness**
- Add a pytest that replays the committed `corpus/`.
- Count "ran dry", digest misses and tape exhaustion as failures.
- Grade the model inside the LLM digest.
- Call the transcript and timing diffs that already exist.
- Let the real dispatcher emit its own events during replay.
- `AgentLoop` config must come from the tape, not `.env`.
- Regenerate the tapes for the new schema, **reviewed by a human** (the `empty_transcript` tape currently pins a bug as expected output).

**Shell (Tauri)**
- A Job Object with KILL_ON_JOB_CLOSE, so the backend and `llama-server` are never orphaned.
- A per-launch WebSocket token.
- Restart or health-watch the backend.
- A single-instance guard.
- A release bundle layout (writable data directory in `%LOCALAPPDATA%`).
- Remove the Google Fonts request and add a CSP.

**Safety**
- Replace the denylist with an allowlist of tools that are safe without confirmation. Currently these bypass it: clicking "Delete" or "Don't Save", typing into cmd then Enter, `cua_launch`, `open_file`, `undo`.
- Restate the full resolved target in every confirmation.
- `delete_file`: stop resolving relative to the working directory, refuse directories, check the `SHFileOperation` return code.
- "Repeat" during a confirmation must repeat the question.

**Automation**
- Pin element ids to the snapshot the planner saw.
- Fix the app-index matching ("zoom" → the uninstaller).
- `set_value` must check focus.
- Fix the `Win32Driver` focus check and its swallowed errors.
- Fix the DPI awareness order (the pyautogui import wins).
- Apply the CDP `devicePixelRatio`.
- Support non-ASCII typing.
- Stop the pyautogui fail-safe from being reported as success.
- Add timeouts around UIA walks.

**Audio**
- `TTSPlayer.stop()` must drop the aborted stream.
- The mic reader must back off when the device is unplugged.
- Allow interrupting Grace mid-speech.
- Stop the chime and TTS tail being captured (echo).
- Don't transcribe silence after a timeout.
- The follow-up window must not cut off someone still speaking.
- Stop calling Vosk `Reset()` across threads.
- The wake word is a substring match ("graceful"); replaced per §3.

**Other**
- Many failures are reported as success: `close_app`, volume, undo, launch, lock, screenshot.
- `~/.grace/memory.db` stores typed text (including passwords) in plain text, with no retention limit, and tests write to it.
- The Gemini API key is sent in the URL.
- Config names in `.env.example` don't match the code; `AGENT_MAX_SECONDS=0` becomes 180.
- `full_setup.bat` reports success after failures; add checksums to model downloads.
- `tests/test_qol_accessibility.py` plays real audio and calls real UIA, and some of its tests can't fail.

---

## 11. Rust port

- Choose every new runtime for Rust bindings: `sherpa-rs` / sherpa-onnx, llama.cpp (server or bindings), `ort`.
- Keep the frozen event contract (`contract/`) as the seam. Port behind it, and grade the port against **reviewed** tapes (§10.2 first).
- The pattern fast path, correction list, tier detection, thread budget and priority control belong in Rust from the start.

---

## 12. Suggested order of work

1. **Harness trust:** commit or split the current working tree (ask the user first), regenerate and review the tapes, add the corpus-replay gate.
2. **Runtime slimming:** remove PyTorch; sherpa-onnx for ASR, TTS, VAD and wake word; thread budget; priority control.
3. **Benchmark 1:** TORGO ASR.
4. **Word biasing and correction list**, plus switch input for confirmations.
5. **Benchmarks 2 and 3**, then the fine-tunes.
6. **Enrollment flow and opportunistic training.**
7. **Remaining backlog (§10.2)**, in parallel with the Rust port.

---

## 13. Rules for future sessions

- Never download models or datasets unless the user asks. They download them themselves.
- Never regenerate `corpus/` without a human reviewing the diff.
- Commit and push to the working branch freely (the user OK'd this on 2026-09-26). Never force-push or rewrite history.
- Never run anything that drives the real desktop, microphone or speakers in tests.
