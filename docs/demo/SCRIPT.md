# Grace — 3-Minute Demonstration Video Script

This script provides a shot-by-shot, timed guide for recording the 2.5 to 3 minute demonstration video for the Symphony 2026 / Bolna Buildathon.

---

## Video Overview & Timing Structure

| Section | Timestamp | Content & Focus | Key Visual / Action |
|---|---|---|---|
| **Scene 1: Hook & Mission** | 0:00 – 0:15 | The accessibility problem & who Grace is for | Presenter speaking, showing hands-free desktop |
| **Scene 2: Airgap Proof** | 0:15 – 0:35 | Unplugging network cable / toggling Airplane mode | Hardware camera cut: Ethernet cable pulled / Wi-Fi off |
| **Scene 3: Multi-App Task** | 0:35 – 1:15 | Multi-step reasoning with auto-retry | Notepad to Edge flow with fallback verification |
| **Scene 4: Safety Gate** | 1:15 – 1:45 | Human-in-the-loop voice confirmation | Attempting file deletion → confirmation dialog & voice consent |
| **Scene 5: Voice Stop Reflex** | 1:45 – 2:05 | Instant interruption mid-typing | Interrupting Grace mid-execution with "Stop" |
| **Scene 6: Set-of-Marks ("4 or 7?")** | 2:05 – 2:35 | Visual disambiguation fallback | Ambiguous UI buttons with high-contrast red numbers |
| **Scene 7: Closing & What's Next** | 2:35 – 2:50 | Future roadmap: CPU fleet, TORGO dysarthria, Rust | Summary graphic of the micromodel fleet |

---

## Detailed Scene Breakdown

### Scene 1: The Problem & Opening Hook (0:00 – 0:15)
- **Visual:** Presenter facing camera, hands resting off keyboard and mouse. The monitor in the background displays a clean Windows 11 desktop with the Grace floating overlay showing `Grace • Ready` with a green `Offline` badge.
- **Presenter (Spoken):**
  > "For millions of people living with ALS, cerebral palsy, or severe tremors, using a mouse and keyboard isn't an option. But existing voice tools force users into rigid, exhausting command grammars. Grace changes that: plain, conversational speech that controls your entire Windows PC—one hundred percent offline."
- **On-Screen Caption:**
  > **Grace: 100% Offline Voice Agent for Motor Accessibility**

---

### Scene 2: The "Pull the Network Cable" Moment (0:15 – 0:35)
- **Visual:** Split screen. Left: Physical close-up of presenter reaching behind the laptop and pulling out the Ethernet cable, then turning Wi-Fi off in Windows Action Center. Right: Network icon shows "No Internet Access".
- **Presenter (Spoken):**
  > "Accessibility software should never depend on cloud latency or leak private documents to remote APIs. Right now, I'm pulling the network cable and disconnecting Wi-Fi. Grace runs completely locally on this laptop—from wake word to vision grounding to speech."
- **On-Screen Caption:**
  > **Airgapped Privacy: Zero Cloud Packets • 100% On-Device Inference**

---

### Scene 3: Multi-App Task with Verification & Retry (0:35 – 1:15)
- **Visual:** Screen capture of Windows desktop. Presenter speaks naturally.
- **Presenter (Spoken):**
  > "Grace, switch to notepad and copy notes then switch to browser."
- **Grace Visual & Audio Action:**
  - Overlay changes color to blue: `[ Acting ] Switching to Notepad…`
  - Notepad comes to the foreground.
  - Overlay updates: `[ Acting ] Copying text…`
  - Grace observes that clicking the edit menu copy item failed verification.
  - Overlay updates: `[ Acting ] Using shortcut…`
  - Grace executes `Ctrl+C`, verifies clipboard state, and focuses Microsoft Edge.
  - Spoken response from Grace (Kokoro TTS):
    > *"I've copied the notes to the browser."*
- **Presenter (Spoken Commentary):**
  > "Notice that Grace didn't just fire-and-forget. When clicking the menu failed verification, it reasoned autonomously, switched to a keyboard shortcut, and confirmed the outcome before moving to Edge."
- **On-Screen Caption:**
  > **Autonomous Multi-Step Execution • State Verification & Self-Correction**

---

### Scene 4: Safety Confirmation Gate by Voice (1:15 – 1:45)
- **Visual:** Desktop showing a folder with project files.
- **Presenter (Spoken):**
  > "Grace, get rid of the draft file."
- **Grace Visual & Audio Action:**
  - Grace identifies `delete_file` intent for `draft.txt`.
  - SafetyGuard halts execution before invoking the filesystem.
  - Overlay changes to amber pill: `[ Waiting for your answer ] Are you sure you want me to delete draft.txt?`
  - Spoken response from Grace:
    > *"Are you sure you want me to delete draft.txt?"*
- **Presenter (Spoken):**
  > "Yes, go ahead."
- **Grace Visual & Audio Action:**
  - Overlay changes to green: `[ Acting ] Moving file to recycle bin…`
  - `draft.txt` disappears into the Recycle Bin.
  - Spoken response from Grace:
    > *"I've moved draft.txt to the Recycle Bin."*
- **On-Screen Caption:**
  > **SafetyGuard: Explicit Voice Confirmation for Destructive Actions**

---

### Scene 5: Immediate Voice "Stop" Reflex (1:45 – 2:05)
- **Visual:** Notepad is open and empty.
- **Presenter (Spoken):**
  > "Grace, type the long query."
- **Grace Visual & Audio Action:**
  - Grace focuses Notepad and begins rapidly typing text.
- **Presenter (Spoken Interrupt - Loud and Clear):**
  > "Stop!"
- **Grace Visual & Audio Action:**
  - Descending cancel chime plays immediately.
  - Cursor instantly halts; active typing stops dead.
  - Overlay updates to red/gray: `[ Stopped ] Ready`
  - Spoken response:
    > *"Stopped."*
- **Presenter (Spoken Commentary):**
  > "The microphone remains open and armed during every action. A voice stop halts execution immediately. The user always has total authority."
- **On-Screen Caption:**
  > **Voice Barge-In • Real-Time Voice Interrupt Reflex**

---

### Scene 6: Numbered-Marks Visual Fallback ("4 or 7?") (2:05 – 2:35)
- **Visual:** Browser showing an intranet portal with two similar "Download" buttons that lack distinct accessibility labels in the UIA tree.
- **Presenter (Spoken):**
  > "Grace, download the report."
- **Grace Visual & Audio Action:**
  - Grace detects ambiguous targets in the accessibility tree.
  - High-contrast red numbered tags appear over both buttons: `[4]` on PDF, `[7]` on Excel.
  - Overlay pill displays: `[ Waiting for your answer ] Did you mean button 4 or button 7?`
  - Spoken response from Grace:
    > *"Did you mean button 4 or button 7?"*
- **Presenter (Spoken):**
  > "Button 4."
- **Grace Visual & Audio Action:**
  - High-contrast overlay dismisses.
  - Grace clicks button 4; PDF download begins.
  - Spoken response:
    > *"I've clicked button 4."*
- **On-Screen Caption:**
  > **Set-of-Marks Vision Grounding • Accessible Disambiguation**

---

### Scene 7: Closing & What's Next (2:35 – 2:50)
- **Visual:** Presenter on camera with summary slide highlighting the post-buildathon architecture.
- **Presenter (Spoken):**
  > "What you've seen is our interim buildathon model running locally on this laptop. Our next step is bringing this capability to the CPU floor: a micromodel fleet for budget laptops, speech fine-tuning on the TORGO dysarthria dataset, and a native Rust executable. Grace puts full PC control back into the hands of those who need it most. Thank you."
- **On-Screen Graphic:**
  > **Grace Roadmap: CPU Micromodel Fleet • TORGO Speech Personalisation • Pure Rust Single Binary**
  > **GitHub: github.com/IronManRox228228/Grace**
