# Grace — Offline Voice Accessibility Assistant for Windows

Grace is a fully offline, local Windows voice agent engineered for people with motor dysfunction (such as ALS, cerebral palsy, spinal cord injuries, or severe tremors). While existing tools like Windows Voice Access and Dragon rely on rigid, memorized command grammars ("click button 3", "press enter twice"), Grace understands plain, conversational speech and autonomously plans, executes, and verifies multi-step computer tasks directly on the user's desktop—without sending a single byte of audio, text, or screen captures to the cloud.

---

## 30-Second Quickstart

### Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.11+
- NVIDIA GPU with 8 GB+ VRAM (for the buildathon interim model; e.g. GeForce RTX 4060)
- Optional: Rust toolchain (1.75+) and Node.js 18+ (to build the native Tauri shell)

### Setup
1. Clone the repository:
   ```cmd
   git clone https://github.com/IronManRox228228/Grace.git
   cd Grace
   ```
2. Create and activate a Python virtual environment:
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`:
   ```cmd
   copy .env.example .env
   ```

### Required Model Files
Grace runs entirely offline. Download the required model weights manually and place them on your system (e.g. in `C:\Users\<User>\Downloads\` or a dedicated `models\` directory), then update your `.env` paths:

| Model / Asset | Role | Source / Hugging Face Repo | File Name | Size |
|---|---|---|---|---|
| **Holo-3.1-4B** | Unified LLM (intent, planning, vision grounding, tool calling) | Official `Hcompany/Holo-3.1-4B` is gated — use the ungated quant mirror `prithivMLmods/Holo-3.1-4B-GGUF` | `Holo-3.1-4B.Q4_K_M.gguf` | ~2.9 GB |
| **Holo-3.1-4B mmproj** | Multimodal projector for vision grounding | `prithivMLmods/Holo-3.1-4B-GGUF` | `Holo-3.1-4B.mmproj-q8_0.gguf` | ~350 MB |
| **Whisper Base.en** | Local speech-to-text (ASR) | `Systran/faster-whisper-base.en` | Bundled / cached via `faster-whisper` | ~140 MB |
| **openWakeWord / Vosk** | Offline wake word ("Grace") | Pre-trained openWakeWord models | `grace.onnx` or small Vosk English model | ~45 MB |
| **Kokoro-82M** | Local text-to-speech (TTS) | `hexgrad/Kokoro-82M` | `kokoro-v0_19.onnx` + `voices.json` | ~320 MB |
| **llama-server** | High-performance local inference binary | `ggerganov/llama.cpp` releases | `llama-server.exe` (cuBLAS/CUDA build) | ~50 MB |

Configure `.env` with:
```env
LLAMA_SERVER_EXE=C:\Users\<User>\llama.cpp\llama-server.exe
LLAMA_MODEL_PATH=C:\Users\<User>\Downloads\Holo-3.1-4B.Q4_K_M.gguf
LLAMA_MMPROJ_PATH=C:\Users\<User>\Downloads\Holo-3.1-4B.mmproj-q8_0.gguf
USE_CLOUD_LLM=false
```

### Run

#### Primary Path (Native Rust Backend + Tauri Overlay Shell, Default)
Build and run Grace directly using Cargo:
```cmd
cargo run -p grace-shell
```
The shell automatically starts the high-performance Rust backend sidecar (`grace-backend.exe`) supervised inside a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) with automatic health-watch and recovery.

To run the headless Rust backend sidecar directly without the UI overlay:
```cmd
cargo run -p grace-backend --bin grace-backend
```

#### Legacy Python Backend (Deprecated)
The Python backend remains selectable for one release cycle:
```cmd
set GRACE_BACKEND=python
cargo run -p grace-shell
```
Or run the Python entrypoint directly:
```cmd
python -m grace.main
```

#### Parity Verification Gate
To verify cross-backend parity and run both test suites:
```powershell
powershell -File scripts/parity.ps1
```

Say: *"Grace, open Notepad and write my appointment at 3 PM."*

---

## What Works Today vs. What Doesn't

### What Works Today
- **100% Offline Execution:** Complete end-to-end pipeline (wake word → Whisper ASR → Holo3.1-4B LLM → UI Automation / Actuator → Kokoro TTS) runs locally on the machine with zero cloud connectivity.
- **Dual-Path Capability Router:** Simple requests (app launches, volume, time, undo) route to instantaneous deterministic handlers (<300 ms) without LLM latency; complex instructions route to the ReAct agent loop.
- **Hybrid Grounding:** Inspects native Windows UI Automation (UIA) accessibility control trees first; falls back to vision grounding with Set-of-Marks visual tags for non-standard controls.
- **Voice Stop Reflex:** Immediate interruptibility (<500 ms) via continuous wake-word cancel watch during active planning, actuation, or speech output.
- **Safety Confirmations:** Irreversible or high-risk actions (`delete_file`, `close_app`, `lock_computer`) pause execution and demand clear spoken confirmation.
- **Agent-Written User Memory:** Remembers what the user tells it (e.g. *"remember my usual song is Signal by Home"*, *"remember that I prefer Firefox"*), confirms verbally before saving, injects confirmed facts into intent and planner prompts (bounded to 8 facts, 600 chars, quarantined in untrusted data delimiters), and supports voice recall (*"What do you remember about me?"*) and hard deletion (*"Forget my name"*, *"Forget that I prefer Firefox"*, *"Forget everything"*). Provenance safety guarantees facts come exclusively from spoken words, never screens or documents, and stored facts can never override safety gates.
- **Deterministic Tape Harness:** Every turn can be recorded to structured tapes (`events`, `prompts`, `transcripts`, `actions`, `timings`) and verified via bit-exact replay testing.

### What Doesn't (Honest Limits & Ongoing Work)
- **High-End Hardware Acceleration:** While Grace runs on CPU-floor machines via the `"cpu-fleet"` micromodel architecture (sub-1B models, int8 ONNX, llama_cpp), ultra-low latency (<20 ms per token) benefits from modern NPU or iGPU hardware acceleration.
- **Physical Switch Hardware Integration:** Switch and puff-and-sip inputs are currently mapped through Windows keyboard accessibility emulation rather than direct hardware serial/USB drivers.
- **UAC Prompt Approval (Known Limit):** By Windows operating system design, user-mode accessibility tools like Grace cannot reach or interact with the Windows Secure Desktop. Grace hands off UAC prompts and lock screens to Windows Voice Access.

---

## Accessibility Setup: Passwords & Windows Security Screens

### 1. Hands-Free Windows Security Screens & Voice Access
Grace coexists seamlessly with Windows Voice Access on a shared microphone:
- **Automatic Handoff:** When a UAC prompt, lock screen, or sign-in screen appears, Grace dispatches nothing, speaks a safe handoff line (*"That's a Windows security screen. Voice Access can handle it."*), and shows the exact Voice Access command phrases in the floating pill UI (*"Say 'voice access wake up', then 'click Yes' or 'click No'"*).
- **Grace Goes Deaf on Secure & Locked Screens:** For user privacy, Grace stops listening completely whenever the desktop is locked or secure so that spoken passwords never reach Grace.
- **Lock Screen Setup:** Verify with `grace.exe --check-setup`. To enable Voice Access before sign-in, tick *Settings → Accessibility → Speech → Voice access → "Start voice access before you sign in"*, or let Grace enable it via `grace.exe --enable-voice-access-at-signin` (approved hands-free via UAC).
- **Sign-In:** Windows Hello Face is recommended for hands-free sign-in. With Voice Access enabled before sign-in, saying *"hello"* dismisses the lock screen, allowing the user to speak their password.
- **Best Practice:** Keep Voice Access asleep during normal use (*"voice access sleep"*), and wake it only when security screens appear (*"voice access wake up"*).

### 2. Passwords Without Hands: Saved-Secret Vault
For users unable to type, Grace types saved passwords into verified fields without ever exposing credential values:
- **Caregiver Enrolment:** Passwords are enrolled securely by a caregiver via CLI: `grace.exe --save-secret <name> --site <domain-or-exe>`. Password entry is hidden and double-entered.
- **Zero Model Exposure:** In everything the planner sees, password fields appear only as `[password field, filled]` or `[password field, empty]`. Credentials never enter prompts, logs, or replay tapes.
- **Deterministic Phishing Guard:** When the user says *"type my gmail password"*, `type_saved_secret("gmail")` verifies at the millisecond of typing that the focused element is a password field and the active browser tab matches the registered eTLD+1 domain (rejecting lookalikes like `gmail.com.evil.io`).

---

---

## Safety Architecture in Five Bullets

1. **Explicit Allowlist Confirmations:** Destructive actions (`delete_file`, `close_app`, `lock_computer`, `shutdown`) are blocked by `SafetyGuard` until the user verbally confirms approval. Unclear or mumbled replies trigger clarification rather than assumption.
2. **Untrusted-Data Quarantine:** Desktop content (window titles, web text, file contents) is quarantined within rigid boundary markers (`<<<UNTRUSTED_DATA_START ...>>>`) to neutralize prompt injection attacks.
3. **Instant Voice Stop Reflex:** The microphone stream remains open during task execution. Saying "Grace stop" or "cancel" immediately halts mouse/keyboard actuation and cancels speech within 500 ms.
4. **Airgap Offline Guarantee:** Grace defaults to `USE_CLOUD_LLM=false` and requires no network interface. The local WebSocket server is secured with per-launch cryptographic tokens and strict Origin allowlists.
5. **Memory Encryption & Pinned Binaries:** Long-term facts are stored locally using Windows DPAPI encryption at rest, sensitive tokens are redacted from logs, and external binaries (`llama.cpp`) require SHA-256 integrity verification.

---

## Architecture Diagram

```mermaid
flowchart TD
    subgraph Audio_Pipeline["Audio Input & Speech Pipeline"]
        Mic["Microphone Input"] --> VAD["Silero VAD / Audio Pump"]
        VAD --> WakeWord["Wake Word Detector\n(openWakeWord / Vosk)"]
        WakeWord -- "Wake Detected" --> Whisper["Offline Whisper ASR\n(faster-whisper)"]
    end

    subgraph Core_Routing["Core Routing & State Machine"]
        Whisper --> Router{"Capability Router\n(Fast-Path vs Agent)"}
        Router -- "Simple Intent" --> FastPath["Fast-Path Dispatcher\n(Open App, Media, Volume)"]
        Router -- "Multi-Step Goal" --> AgentLoop["Agentic ReAct Loop"]
    end

    subgraph Agent_Reasoning["Agent Reasoning & Perception"]
        AgentLoop <--> LLM["Local Holo3.1-4B\n(llama-server cuBLAS)"]
        AgentLoop --> UIATree["Windows UIA Tree Inspector"]
        AgentLoop --> VisionGround["Vision Grounder\n(Set-of-Marks)"]
        AgentLoop --> SafetyGate{"SafetyGuard\n(Allowlist Gate)"}
    end

    subgraph Desktop_Execution["Execution & Feedback"]
        SafetyGate -- "Confirmed / Safe" --> Actuator["Actuator\n(Mouse / Keyboard / Window)"]
        SafetyGate -- "Unsafe" --> AskUser["Spoken Confirmation Gate"]
        FastPath --> Actuator
        Actuator --> TTS["Kokoro-82M TTS"]
        TTS --> AudioOut["Audio Output / Earcon Chimes"]
    end

    subgraph Verification_Harness["Verification & Observability"]
        TapeHarness[("Tape Harness\n(JSONL Session Tapes)")] -. "Record / Replay" .-> AgentLoop
        TauriUI["Tauri Shell & React UI"] <== "Authenticated WebSocket\n(grace-events contract)" ==> Core_Routing
    end
```

---

## Completed Milestones & v0.2.0 Roadmap

### Completed Milestones
1. **The Micromodel Fleet (PLAN §3):** Designed and validated the sub-1B fleet architecture running within CPU-floor budgets (`"cpu-fleet"` preset):
   - ASR: Parakeet-TDT 0.6B / Moonshine Base (int8 ONNX via sherpa-onnx)
   - Tool calling: LFM2.5-350M fine-tune / FunctionGemma 270M
   - Element selection: GLiNER2.5-Decide (340M) / Julia-1 (144M ONNX)
   - Vision grounding: LFM2.5-VL-450M Set-of-Marks
   - Decision heads: Contrastive InfoNCE decision projections
2. **Speech Personalisation (PLAN §6):** Acoustic evaluation on the TORGO dysarthria benchmark dataset, word biasing, correction list, and ongoing personal speech adaptation pipeline.
3. **Pure Rust Port & Sidecar Split (PLAN §11):** Transitioned default backend to standalone Rust binary (`grace-backend.exe`) supervised by Tauri in a Windows Job Object with health-watch and recovery.
4. **Floor Latency Benchmarks (PLAN §9):** Hardware throttling kit (`tools/floor/`) and automated floor benchmark runner validating all sub-second latency budgets under severe memory and CPU constraints.

### Upcoming in v0.2.0
- Complete retirement and deletion of deprecated Python serving runtime (`src/grace/main.py`, `src/grace/agent/`, `src/grace/intent/`, `src/grace/tools/`).
- Direct USB/serial driver integration for physical switch accessibility hardware.
- Native MSI installer bundling the Tauri shell and Rust backend sidecar.

---

## Verification & Testing

Grace maintains strict automated regression test suites across Python, Rust, and Frontend:
- **Python Backend:** `1198 passed, 1 skipped` (`pytest`)
- **Rust Crates:** last recorded `289 passed, 3 ignored, 0 failed` (`cargo test --workspace`, per `PORT_STATUS.md`; not re-verifiable on this machine without the Rust toolchain)
- **Frontend Renderer:** `65 passed` (`npm run test`)
- **Preflight Checks:** `14/14 checks passed` (`python scripts/demo_check.py`)
- **Deterministic Replay Gate:** 100% pass on 30/30 recorded corpus tapes with exact event parity.
- **Parity Gate Script:** `powershell -File scripts/parity.ps1` runs both test suites and validates replay parity.
