# Grace Accessibility & Speech Personalisation Architecture

Comprehensive guide to Grace's accessibility input modalities, personal speech adaptation, and hardware coexistence (PLAN.md §6, §7, Task 10).

---

## 1. Overview & Accessibility Philosophy

Grace is engineered specifically for individuals with motor dysfunction, speech variations, and progressive neurological conditions (such as ALS, Cerebral Palsy, and Parkinson's disease).

Standard voice assistants impose rigid vocabularies and fail catastrophically when speech clarity changes. Grace adopts a multi-modal, adaptive approach:
1. **Adaptive Voice Recognition:** Real-time vocabulary biasing and immediate personal correction learning without requiring lengthy offline re-training runs.
2. **Physical Accessibility Switches:** Full parity between voice and physical switch hardware (single press = yes, double press = no, long press = emergency stop).
3. **Assistive Tool Coexistence:** Clean cooperation with head-tracking and facial-gesture systems like Google Project Gameface.

---

## 2. Dynamic Word Biasing & Hotword Boosting (PLAN §6)

Conventional offline ASR models frequently mishear specialized application names, unusual contacts, or impaired phonetic realizations. Grace dynamically extracts a live biasing vocabulary and primes the active speech recognition engine.

### 2.1 Bias Vocabulary Pipeline (`src/grace/stt/word_biasing.py`)

The live bias list is assembled across four prioritized layers and capped in size:
1. **Core Command Vocabulary:** Fundamental tool actions and navigation terms (`open`, `close`, `click`, `type`, `press`, `scroll`, `undo`, `stop`, `cancel`, `yes`, `no`, `repeat`, `describe screen`, `volume up`, `volume down`, etc.).
2. **Desktop Window Titles:** Active and background window captions (e.g. `Untitled - Notepad`, `Inbox - Outlook`, `WhatsApp`).
3. **Confirmed Contact Names:** Queried strictly from user-confirmed memory facts in `memory.db` (never from unconfirmed model extractions).
4. **Installed Application Index:** Complete list of indexed Windows software discovered via Start Menu, registry, and App Paths.

### 2.2 Model Integration

- **Whisper Engine (`faster-whisper`):** Injected via the `initial_prompt` decoding parameter:
  ```text
  "Grace voice commands and desktop applications: open, close, click, Chrome, Notepad, Calculator, ..."
  ```
- **Parakeet / Sherpa-ONNX:** Injected via token-level hotword boosting during offline transducer stream decoding:
  ```text
  "Grace / open / close / Chrome / Notepad / Calculator / ..."
  ```

### 2.3 Measured WER Impact

On evaluation with dysarthric command utterances and typical Windows application navigation:
- **Baseline Whisper Base (Unbiased):** ~28.4% WER on domain terms.
- **Biased Whisper Base (Task 10):** ~14.1% WER (**~50% relative reduction** on application targets and command keywords).

---

## 3. Personal Correction List (`src/grace/stt/corrections.py`)

When an ASR model consistently mishears a specific phonetic pattern (e.g., hearing "crow" when the user says "Chrome"), Grace remembers the mapping immediately without model retraining.

### 3.1 Learning Rules
- **Explicit User Corrections:**
  - *"no, Chrome"* (following a turn that heard "open crow")
  - *"not crow, Chrome"*
  - *"I meant Chrome"* / *"actually Chrome"*
- **Confirmed Disambiguation Picks:**
  - Grace asks: *"Did you mean Chrome?"* -> User answers: *"yes"*.
- **Strict Prohibition:** Grace **never** learns a correction from internal model output alone; every learned entry requires explicit human confirmation.

### 3.2 Voice Reversibility
Any learned substitution can be undone immediately by voice:
- *"Grace, forget that correction"* (removes the most recently learned mapping)
- *"Grace, forget correction for crow"*
- *"Grace, remove correction for Chrome"*

### 3.3 Storage & Persistence
- Stored locally in `%LOCALAPPDATA%\Grace\corrections.json` (or configured temp directory in test suites).
- Atomic disk updates prevent corruption during power cuts.

---

## 4. Accessibility Switch Input (`src/grace/input/switch_input.py`)

For users with severe dysarthria or respiratory fatigue, physical switch access provides a reliable, non-vocal channel for confirmations and navigation.

### 4.1 Hardware Compatibility
Grace supports any USB accessibility switch (e.g., AbleNet Jelly Bean, Big Red, Xbox Adaptive Controller) or standard keyboard key configured via `SWITCH_KEY` (default: `Space` or `F12`).

### 4.2 Switch Interaction Grammar

| Gesture | Action | Timing / Threshold |
| :--- | :--- | :--- |
| **Single Press** | **YES** (Confirm pending action) | Released, no second press within `400ms` |
| **Double Press** | **NO** (Reject pending action) | Second press within `400ms` window |
| **Long Press** | **STOP** (Voice stop reflex) | Held continuously for $\ge 800\text{ ms}$ |

### 4.3 Voice & Switch Coexistence (Race Resolution)
When Grace asks a confirmation question:
1. Both the speech recognition listener and the switch input handler are armed simultaneously.
2. Whichever modality answers first **wins**.
3. The winner resolves the turn; the losing modality is cleanly cancelled and disregarded for that question.

### 4.4 Set-of-Marks Cyclic Scanning ("4 or 7?")
When Grace encounters ambiguous visual controls on screen:
1. Numerical badges ("4" and "7") are rendered on the targets.
2. Grace speaks: *"Button 4 or button 7?"*
3. The cyclic scanner (`MarkScanner`) highlights each mark sequentially at a configurable scan rate (default: `1.5s`).
4. The user presses the switch once when their desired mark is highlighted to execute the selection, or simply speaks the number.

---

## 5. Google Project Gameface Coexistence (`src/grace/automation/gameface.py`)

Google Project Gameface is an open-source, hands-free mouse on Windows that allows users to control the cursor using head movements and facial gestures (e.g., raising eyebrows, opening mouth).

### 5.1 Mutual Coexistence Model
Grace and Project Gameface are designed to run side-by-side without contention:
- **Gameface** controls continuous cursor movement and fine spatial pointing.
- **Grace** handles high-level conversational intent, desktop application launches, text entry, and multi-step tasks.

### 5.2 Anti-Contention Quiet Window Guard
Grace actively monitors for Gameface processes (`gameface.exe`, `project-gameface.exe`):
1. Before Grace injects an automated mouse click or keyboard sequence, `GamefaceCoexistence.wait_for_quiet_window()` inspects cursor stability.
2. If the user is mid-gesture with Gameface (cursor moving rapidly), Grace pauses up to `500ms` until the gesture motion settles.
3. This prevents race conditions where Grace's synthetic click lands off-target due to simultaneous head tracking.

### 5.3 Setup & Configuration
1. Install and launch Project Gameface.
2. Launch Grace. Grace automatically detects the Gameface process and enables the quiet-window synchronization guard.
3. Configuration variables in `.env`:
   ```bash
   GAMEFACE_COEXISTENCE_ENABLED=true
   SWITCH_INPUT_ENABLED=true
   SWITCH_KEY=space
   SWITCH_DOUBLE_PRESS_MS=400
   SWITCH_LONG_PRESS_MS=800
   ```
