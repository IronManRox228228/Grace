# Grace — Buildathon Submission Form Answers

This document provides ready-to-paste responses for standard hackathon/buildathon submission forms (e.g. Devpost, Google Forms) for the **Symphony 2026 / Bolna Buildathon**.

---

## 1. Project Information

### Project Name
**Grace**

### Tagline / One-Liner (13 words)
> **Fully offline Windows voice agent enabling hands-free computer control through natural speech.**

---

## 2. Problem Statement
Over 40 million individuals worldwide live with severe motor disabilities resulting from ALS, cerebral palsy, quadriplegia, stroke, or advanced Parkinson's disease. For these users, operating a keyboard and mouse is difficult or impossible.

Existing assistive technologies fail them in critical ways:
1. **Rigid, Exhausting Syntax:** Tools like Windows Voice Access and Dragon require memorizing complex, unnatural command grammars (e.g. *"click button 42"*, *"press tab three times"*). A single misspoken word aborts the entire workflow, imposing intense cognitive fatigue.
2. **Cloud Privacy & Security Risks:** Modern AI assistants (Siri, Alexa, cloud computer-use agents) stream voice recordings, screen captures, and sensitive personal data (emails, banking, medical records) to remote servers—violating user privacy and presenting severe exfiltration risks.
3. **High Latency & Fragility:** Cloud-dependent agents introduce multi-second round-trip latency, making computer control sluggish and unusable during intermittent network connectivity.

---

## 3. Proposed Solution
**Grace** is an offline, autonomous voice accessibility agent that transforms plain, conversational speech into multi-step desktop computer actions directly on a Windows PC.

Instead of demanding strict syntax, Grace understands natural user intentions (e.g. *"Grace, switch to Notepad and copy my notes to the browser"*), autonomously inspects the desktop, plans logical execution steps, verifies the effect of each action, and speaks concise natural language status updates.

Grace operates with a **dual-path architecture**:
- **Fast-Path Dispatcher:** Common system commands (launching applications, media control, volume, undo) execute directly, without an LLM call.
- **ReAct Agent Loop:** Complex goals trigger an autonomous reasoning loop that inspects the Windows UI Automation (UIA) accessibility tree, falls back to Set-of-Marks visual grounding for unstandardized controls, and verifies screen state changes after every step.

---

## 4. Technical Stack & Architecture

- **Speech Recognition (ASR):** Local `faster-whisper` (CTranslate2) with pre-warmed Whisper Base.en models running on-device.
- **Wake Word Detection:** Offline `openWakeWord` and lightweight Vosk keyword spotting with continuous cancel-watch capabilities for voice barge-in.
- **Reasoning & Vision (LLM):** Quantized `Holo-3.1-4B` (GGUF) served locally via `llama.cpp` (`llama-server`) with a multimodal vision projector (`--mmproj`).
- **Desktop Automation:** Windows UI Automation (UIA) tree walker combined with Win32 `SendInput` keyboard and mouse actuation, DPI-aware coordinate scaling, and safe application index caching.
- **Speech Synthesis (TTS):** Local `Kokoro-82M` ONNX text-to-speech providing human-like voice synthesis with zero cloud dependencies.
- **Desktop Shell & UI:** High-contrast, accessible Tauri v2 native overlay shell with React, TypeScript, Tailwind CSS, keyboard focus support, and Windows High Contrast mode compatibility.
- **Rust Core:** Ongoing native port (`crates/grace-core`, `crates/grace-win`, `crates/grace-memory`, `crates/grace-harness`, `crates/grace-backend`) behind a frozen JSON event schema.

---

## 5. What's Novel & Key Differentiators

1. **100% Offline Airgapped Computer Control:** Unlike cloud-tethered OS agents, Grace runs completely locally on the device with zero outbound network calls, ensuring total privacy for personal, financial, and medical documents.
2. **Dual-Path Latency Architecture:** Distinguishes immediate one-shot commands from complex multi-step reasoning, providing conversational responsiveness without sacrificing agentic power.
3. **Accessible Set-of-Marks Disambiguation ("4 or 7?"):** When UI controls are ambiguous or unstandardized in the accessibility tree, Grace overlays high-contrast numbered badges and asks the user to pick by voice or switch input—eliminating frustrating click failures.
4. **Instant Voice Stop Reflex:** The microphone stream remains open during task execution, allowing the user to say *"Stop"* to halt typing, mouse movement, or speech synthesis immediately.
5. **Deterministic Tape Harness:** An hermetic session recorder and replay engine that records multi-step agent interactions to structured tapes, allowing automated regression testing without driving physical hardware.

---

## 6. What Was Built During the Buildathon Window (Oct 15–31, 2026)

*Derived directly from git commit history:*
- **Unified Multimodal Model Integration:** Replaced legacy UI-TARS with unified local Holo3.1-4B covering intent, planning, tool calling, and vision grounding via llama-server (`c080180`, `beefc20`).
- **Autonomous Agent Hardening:** Implemented multi-step ReAct planning loop with single-screen observation memory, post-action verification, and automatic keyboard shortcut escalation (`22cee62`).
- **Safety Architecture Rewrite:** Built an explicit allowlist `SafetyGuard` requiring verbal human confirmation for destructive actions (`delete_file`, `close_app`, `lock_computer`) with strict yes/no parsing (`51180f7`).
- **Airgap & Security Hardening:** Implemented offline-by-default runtime enforcement, per-launch 32-byte WebSocket bearer tokens, Windows DPAPI credential encryption at rest, and regex tape sanitization (`1dfdc72`, `64c9263`).
- **Untrusted Content Quarantine:** Implemented delimited marker quarantine (`<<<UNTRUSTED_DATA_START ...>>>`) to neutralize prompt injection from hostile desktop windows and websites (`64c9263`).
- **Audio & Accessibility Sweep:** Resolved audio barge-in, echo suppression mic gating, silent activation timeouts, and unindexed app handling (`b9878d4`).
- **Scenario Tape Coverage:** Generated and verified 14 new deterministic scenario tapes covering all system tools, failure recovery, and disambiguation paths (`5065bc6`).
- **Demo Polish & Preflight Check:** Built startup warm-up (text-token + vision projector warm-up), instant acknowledgement earcon, GPU tuning configs, per-turn latency reporting (`scripts/latency_report.py`), automated watchdog process recovery, 14-point preflight checklist (`scripts/demo_check.py`), and high-contrast Choice Overlay (`eb9b0fb`).

---

## 7. Team & Links

### Team Members
- **Lead Developer:** `IronManRox228228` (GitHub)
- *[Placeholder: Additional Team Members]*

### Project Links
- **GitHub Repository:** [https://github.com/IronManRox228228/Grace](https://github.com/IronManRox228228/Grace)
- **Demonstration Video:** *[Placeholder: Link to 3-minute video presentation]*
- **Architecture Documentation:** [docs/ARCHITECTURE.md](https://github.com/IronManRox228228/Grace/blob/migration/phase-0-contract-and-harness/docs/ARCHITECTURE.md)
- **Safety Whitepaper:** [docs/SAFETY.md](https://github.com/IronManRox228228/Grace/blob/migration/phase-0-contract-and-harness/docs/SAFETY.md)
- **Live Demo Script:** [docs/demo/SCRIPT.md](https://github.com/IronManRox228228/Grace/blob/migration/phase-0-contract-and-harness/docs/demo/SCRIPT.md)

---

## 8. Open Questions for Buildathon Organizers (Bolna / Symphony)

Before final submission on October 31, the following questions should be clarified with the organizers:
1. **Age Requirement:** Are there any age limitations or legal guardian consent requirements for finalist prize eligibility?
2. **Platform Integration:** Is integration with Bolna's cloud platform/SDK mandatory, or does Grace qualify under the open accessibility track as a 100% offline, privacy-first Windows voice agent?
3. **Submission Deliverables:** What are the exact requirements for the submission package (video length limits, slides, Devpost form fields), and what is the schedule/format for the November 20 in-person finale in Bengaluru?
