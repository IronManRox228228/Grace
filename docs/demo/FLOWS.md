# Grace — Live Demo Flows & Presenter Script

This document details the core demonstration flows for the **Symphony 2026 / Bolna Buildathon** live demo.
Each flow is designed to showcase Grace's speed, reliability, and accessibility as a **100% offline, local-only Windows voice agent** for people with motor dysfunction.

Every flow has a corresponding deterministic replay tape stored in `%TEMP%\grace-demo\tapes` and verified against the golden contract.

---

## Preflight Before Presenting

Before walking on stage or starting the recording:

1. **Pull the network cable / disable Wi-Fi.** Prove Grace has zero non-loopback connections.
2. **Run the live-model demo gate ("run before every recording"):**
   ```cmd
   python scripts/demo_gate.py
   ```
   Verifies local Holo 3.1-4B visual grounding, planner tool-calling, intent routing (100% of `agent` intents routed), and agentic tape drive execution without stuck loops. All checks must show `PASS`.
3. Run the preflight verification script:
   ```cmd
   python scripts/demo_check.py --demo
   ```
   Ensure all 14 checklist items pass in green.
4. Launch Grace with the Tauri overlay shell:
   ```cmd
   cargo tauri dev
   ```
5. Verify the overlay pill appears at the bottom center of the display showing **`Grace • Ready`** with a green **`Offline`** badge.

---

## Flow 1: Multi-App Task with Verification & Alternative Retry

Demonstrates multi-step reasoning, cross-application focus switching, UI state verification, and automatic escalation when an initial action does not produce the expected state change.

- **Replay Tape:** `%TEMP%\grace-demo\tapes\multi_app_verify_retry`
- **What it Proves:** Grace does not fire-and-forget. It verifies the effect of every action against the UI element graph. When clicking a copy button fails, it reasons about the failure and automatically retries using a keyboard shortcut (`Ctrl+C`), verifies success, and completes the multi-app goal.

### Script
1. **Presenter says:**
   > *"Grace, switch to notepad and copy notes then switch to browser"*
2. **Grace visual & audio response:**
   - Overlay status: **`[ Acting ] Switching to Notepad…`**
   - Grace focuses Notepad.
   - Overlay status: **`[ Acting ] Copying text…`**
   - Click copy button fails verification (`nothing copied`).
   - Overlay status: **`[ Acting ] Using shortcut…`**
   - Grace executes `Ctrl+C` shortcut and confirms clipboard contains copied text.
   - Overlay status: **`[ Acting ] Switching to Browser…`**
   - Grace focuses Microsoft Edge.
   - Spoken response:
     > *"I've copied the notes to the browser."*
3. **If it goes wrong (Recovery line):**
   > *"Grace, stop and switch to browser"*

---

## Flow 2: Safety Confirmation Gate by Voice

Demonstrates human-in-the-loop protection for destructive actions (moving files to the Recycle Bin, terminating processes, or locking the workstation). The action is safely parked on the dispatch boundary until explicitly approved by voice.

- **Replay Tape:** `%TEMP%\grace-demo\tapes\safety_confirm_accepted`
- **What it Proves:** Grace protects user data from accidental voice triggers or hallucinations. High-consequence tools cannot execute without clear, explicit confirmation.

### Script
1. **Presenter says:**
   > *"Grace, get rid of the draft file"*
2. **Grace visual & audio response:**
   - Grace identifies `delete_file` intent for `draft.txt`.
   - SafetyGuard halts execution at the dispatch boundary.
   - Overlay status: **`[ Waiting for your answer ] Are you sure you want me to delete draft.txt?`**
   - Spoken response:
     > *"Are you sure you want me to delete draft.txt?"*
3. **Presenter says:**
   > *"Yes, go ahead"*
4. **Grace visual & audio response:**
   - Overlay status: **`[ Acting ] Moving file to recycle bin…`**
   - `draft.txt` is moved to the Windows Recycle Bin.
   - Spoken response:
     > *"I've moved draft.txt to the Recycle Bin."*
5. **If it goes wrong (Recovery line):**
   > *"No, don't do that"* or *"Never mind"* (Grace safely cancels the parked action and deletes nothing).

---

## Flow 3: Immediate Voice "Stop" Mid-Task (Interruption Reflex)

Demonstrates sub-300ms barge-in and cancellation reflex. At any time during planning, typing, or speech synthesis, a voice "stop" immediately aborts the active pipeline without hanging or leaving stray input on the desktop.

- **Replay Tape:** `%TEMP%\grace-demo\tapes\stop_during_typing`
- **What it Proves:** People with motor dysfunction have complete authority over their PC. Grace is always listening for stop commands, cancelling active typing and clearing partial inputs immediately.

### Script
1. **Presenter says:**
   > *"Grace, type the long query"*
2. **Grace begins executing:**
   - Overlay status: **`[ Acting ] Typing query…`**
   - Grace begins typing text into the focused field.
3. **Presenter interrupts mid-typing:**
   > *"Stop"*
4. **Grace visual & audio response:**
   - Audio feedback: Plays descending cancellation earcon (`FeedbackSounds.play_cancel`).
   - Typing is immediately interrupted; cursor halts.
   - Overlay status: **`[ Stopped ] Ready`**
   - Spoken response:
     > *"Stopped."*
5. **If it goes wrong (Recovery line):**
   > *"Grace, undo"* (Restores previous text state via Ctrl+Z).

---

## Flow 4: Numbered-Marks Visual Fallback ("4 or 7?")

Demonstrates visual disambiguation when multiple similar UI controls match a goal (e.g. multiple download buttons). Grace displays high-contrast numbered badges over the candidates and asks the user to choose by voice or switch input.

- **Replay Tape:** `%TEMP%\grace-demo\tapes\numbered_marks_fallback_choice`
- **What it Proves:** When an accessibility tree is ambiguous or incomplete, Grace uses Set-of-Marks visual prompting to present distinct numbered badges (`[4] Button 4` vs `[7] Button 7`) that the user can answer naturally by voice or key/switch press.

### Script
1. **Presenter says:**
   > *"Grace, download the report"*
2. **Grace visual & audio response:**
   - Grace observes two report download options (PDF vs Excel).
   - High-contrast Choice Overlay mounts on the screen displaying:
     - `[ 4 ] Button 4` (Download Report PDF)
     - `[ 7 ] Button 7` (Download Report Excel)
   - Overlay status: **`[ Waiting for your answer ] Did you mean button 4 or button 7?`**
   - Spoken response:
     > *"Did you mean button 4 or button 7?"*
3. **Presenter says:**
   > *"Button 4"* (or presses switch / key '4')
4. **Grace visual & audio response:**
   - Overlay status: **`[ Acting ] Clicking button 4…`**
   - Button 4 is clicked, downloading the PDF report.
   - Spoken response:
     > *"I've clicked button 4."*
5. **If it goes wrong (Recovery line):**
   > *"Cancel that"* or *"Neither"*

---

## Flow 5: Local Document Reading & Sub-second RAG

Demonstrates local PDF document ingestion, text extraction, sub-second TF-IDF chunk retrieval, and natural spoken synthesis without sending any document data to cloud servers.

- **Replay Tape:** `%TEMP%\grace-demo\tapes\agentic_read_pdf`
- **What it Proves:** Grace performs local RAG on user files in milliseconds with strict offline privacy.

### Script
1. **Presenter says:**
   > *"Grace, read my report pdf to me"*
2. **Grace visual & audio response:**
   - Overlay status: **`[ Acting ] Reading document…`**
   - Local PDF text extracted and chunked.
   - Spoken response:
     > *"The report covers third quarter results."*
3. **If it goes wrong (Recovery line):**
   > *"Stop speaking"*

---

## Tape Review and Installation

All 5 demo flow replay tapes have been deterministically recorded and verified:
- `multi_app_verify_retry`
- `safety_confirm_accepted`
- `stop_during_typing`
- `numbered_marks_fallback_choice`
- `agentic_read_pdf`

To review and install them into the permanent repository corpus:

```cmd
robocopy "%TEMP%\grace-demo\tapes" "corpus" /E
```
