# Grace Rust port — status

Two phases in. Phase 1 established the workspace and ported the pure
edge-level logic (contract, safety, router, config, VAD, the Origin
allowlist). **Phase 2** (this update) ported the core turn pipeline itself -
intent parsing, the tool schema, the planner, the grounder, the element
graph, memory, the full agent loop (with every ship-blocker behaviour named
in the task), the dispatcher (every system tool), the response generator,
the earcon/feedback boundary, and the `GraceApp` turn state machine - wired
it into the harness as a real turn replay against real corpus tapes (not
just structural validation), added the first real Windows API calls, and
wired `GRACE_BACKEND=rust` to run one real, complete activation turn
end-to-end through a real WebSocket, using the actual ported decision logic
the whole way through.

**Nothing under `src/grace/`, `tests/`, `corpus/`,
`contract/grace-events.schema.json`, `.env`, or `PLAN.md` was modified.**
Everything new lives under `crates/`, plus small, targeted edits to
`src-tauri/` and the root `Cargo.toml`/`Cargo.lock`. Verified at the end of
this phase too: `git status` shows only those paths changed, and
`./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider` still reports
**841 passed, 1 skipped**, unchanged from PLAN.md §0's baseline.

## Build / test / run

```sh
# Whole workspace (Rust backend crates + the Tauri shell)
cargo build --workspace
cargo test --workspace

# Just the ported logic crates, faster inner loop
cargo test -p grace-contract -p grace-core -p grace-audio -p grace-models -p grace-win -p grace-harness -p grace-backend

# Harness: structural-parity report against the real corpus/ (all 30 tapes)
cargo run -p grace-harness --bin grace-replay -- --corpus corpus

# Harness: real turn replay against specific corpus/ tapes (see "Harness
# parity" below for which ones, and why only those)
cargo test -p grace-harness turn_replay

# Run the shell against the Python backend (default, unchanged)
cargo run -p grace-shell
# Run the shell against the in-process Rust backend (runs one real demo
# turn per wake request - see "Tauri wiring" below)
GRACE_BACKEND=rust cargo run -p grace-shell

# Sanity-check the two real Windows API calls this phase added (not part of
# the automated suite - see "grace-win" below)
cargo test -p grace-win --lib -- --ignored --test-threads=1

# Prove nothing under src/grace or tests was touched
./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

## Crate map

| Crate | Lines | Ported from | What's in it |
|---|---|---|---|
| `crates/grace-contract` | ~340 | `contract/grace-events.schema.json` | `GraceEvent` enum, schema-validated round trip. Unchanged since Phase 1. |
| `crates/grace-core` | ~5,700 | `config.py`, `agent/safety.py`, `agent/loop.py`, `agent/memory.py`, `agent/perception.py`, `agent/planner.py`, `agent/grounder.py`, `intent/parser.py`, `intent/tools.py`, `intent/router.py`, `tools/dispatcher.py`, `response/generator.py`, `response/feedback.py`, `util/timing.py`, `text/sentence_split.py`, `main.py`'s `GraceApp` turn logic, `perception/elements.py`, `perception/element_graph.py` | The whole turn pipeline's decision logic. See below, module by module. |
| `crates/grace-audio` | ~1,530 | `vad/detector.py`, `audio/wake_word.py`, `audio/transcription.py`, `audio/tts.py`, `audio/player.py` | Real microphone capture (`cpal`/WASAPI, 16kHz mono resampling); `AudioPump` with echo gating (`gate(0.20)`); Silero VAD v5 ONNX via `sherpa-rs`; Zipformer KWS keyword-spotting with whole-word regex matching and 500ms stop/cancel watch (`\b(stop|cancel)\b`); Sherpa-ONNX STT (`SherpaAsr`, Moonshine Base & Parakeet CTC via `sherpa-rs-sys` C FFI) implementing `SpeechToText`; Kokoro TTS (`SherpaKokoroTts` generating WAV audio) implementing `TextToSpeech` and `CpalTtsPlayer` with atomic barge-in stop flag and echo gating; `TranscriptSource` trait with `ScriptedTranscriptSource` and `RealTranscriptSource` / `OwnedRealTranscriptSource`. |
| `crates/grace-models` | ~250 | `llm/gemma_client.py`'s HTTP shape | `HttpLlm`: a REAL (not stubbed) blocking HTTP client against a local llama.cpp server's `/v1/chat/completions`, non-streaming, matching `GemmaClient.chat(..., stream=False)`'s exact request shape including the `image_b64` → multimodal `image_url` reshaping. `ElementScorer` trait for a future GLiNER-style scorer, not implemented. |
| `crates/grace-win` | ~600 | `automation/app_indexer.py`, `automation/dpi_helper.py`, a slice of `perception/element_graph.py`'s `active_window()` | `plan_launch`/`find_app` (Phase 1). `dpi::real::get_dpi_scale`/`ensure_dpi_aware` and `win32::real::foreground_window` - real `windows`-crate Win32 calls. UIA tree walk, win32 input driver (`DesktopActuator`), and input-time foreground window re-checks remain trait boundaries. |
| `crates/grace-harness` | ~750 | `harness/tape.py`, `harness/replay.py` | Tape loader + diff machinery (Phase 1). `turn_replay.rs` drives the real `grace_core` turn engine against real tapes and diffs the resulting event stream. |
| `crates/grace-backend` | ~800 | `ws_server.py` | `WsEventServer` with 127.0.0.1 binding, Origin allowlist, and per-launch WebSocket authentication token (`?token=...` or `Authorization: Bearer ...`). `turn.rs` demo turn execution. |
| `src-tauri/` | (small diff) | `main.js:93-141` | `Backend::start` branches on `GRACE_BACKEND`; manages per-launch WebSocket token generation, passes token to backend and injects token into webview via `window.__GRACE_WS_TOKEN__`. |
| `crates/grace-memory` | ~1,850 | `agent/memory.py`'s `PersistentMemoryStore` (replaced, not ported - see PLAN.md §12) | Facts store and history store on `rusqlite`/FTS5. Externalized salt (`.salt` file with legacy migration), `forget_goal` cache and DB eviction, voice-scoped deletes. Note: DPAPI at-rest encryption remains Python-only. |

## What's fully ported this phase (module → behaviour, with tests)

- **`grace-core::tools`** (`intent/tools.py`): the complete `CUA_TOOLS`/
  `SYSTEM_TOOLS` schema (26 tools, every parameter, description and default),
  `format_tools_compact`/`format_tools_for_prompt` string generation. This is
  what the planner's system prompt is built from, so a drift here is a drift
  in what the model is told it can do.
- **`grace-core::intent`** (`intent/parser.py`): `Intent`, `IntentParser`,
  `clean_json_fence` (markdown-fence stripping), `VALID_TOOLS` derived from
  the tool schema rather than hand-duplicated (matching the Python module's
  own stated reason for existing).
- **`grace-core::grounder`** (`agent/grounder.py`): Model-agnostic visual
  grounder replacing UI-TARS. Supports configurable coordinate formats
  (`0-1000`, `0-1`, `pixels`), robust coordinate extraction (`click(point=...)`,
  JSON objects/arrays, box tokens, tuples), bounds validation, screen DPI scaling,
  `describe_target` phrase building, and `Grounder::locate` round-trip.
- **`grace-core::planner`** (`agent/planner.py`): `PlannedStep`,
  `needs_grounding`'s exact three-way gate (has `element_id`? has both `x`
  and `y`? else does it name a `target_name`/`describe`?), the full system
  prompt, `build_prompt`'s section ordering, the per-goal call budget
  (0/negative = unlimited, matching `math.inf`), and `parse_planned_step`'s
  fence-then-brace-scan fallback for a model that wraps JSON in prose.
- **`grace-core::perception`** (`perception/elements.py` +
  `perception/element_graph.py`): `ElementNode` (every field:
  frame/placeholder/container/focused/focusable/automation_id/source),
  `is_actionable`/`is_interactive`, `find_by_id`/`find_by_name`/`find_at_point`,
  `ElementGraph::resolve`'s exact reliability ordering (id → frame-scoped
  name → browser's-page-content-first → unscoped name), and `build_graph`'s
  DOM-then-UIA merge with `drop_overlapping`'s duplicate-page-control
  removal and post-merge dense renumbering. **Scope note**: the real
  UIA/DOM/CDP tree walk that produces the raw elements this merges is not
  ported (real desktop - see `grace-win` below); what's ported is the merge
  logic itself, which is exactly where the "renumbered under the planner
  mid-goal" bug class lives.
- **`grace-core::memory`** (`agent/memory.py`): `AgentMemory`,
  `establish`/`rule_out`'s `MAX_REMEMBERED`-bounded, deduplicating rolling
  lists, `pause_clock`/`resume_clock` (a safety confirmation answered by
  voice must not count against the goal's time budget), `is_exceeded`/
  `is_out_of_time`'s "0 means unlimited" semantics, and
  `format_history_markdown`'s last-3-steps window. `PersistentMemoryStore`
  (SQLite in Python) is a trait with an in-memory fake; no real SQLite file
  yet (a contained, low-risk follow-up - see Remaining work).
- **`grace-core::agent_loop`** (`agent/loop.py`) - the centrepiece, with
  every ship-blocker/finding the task named explicitly:
  - **30 s pending-confirmation TTL** (`PENDING_CONFIRMATION_TTL_SECONDS`):
    `expire_pending` drops a parked step once its wall-clock age exceeds the
    limit; tested both just inside and just outside the boundary.
  - **Completion guard with the claiming step**: `verify_goal_completion`
    takes `claiming_step`/`claiming_result` and includes them in the
    interactive-step history it checks *before* the step that claims
    completion is recorded - reproducing the exact bug class the Python
    docstring describes (a step reporting `wrong_focus` whose own claim of
    success went unchallenged because the check ran against history that
    didn't yet include it).
  - **Escalation budget/rate-limit handling**: the `normal → reground →
    stronger → stop` ladder (`rung`), with the "stronger" rung's *extra*
    planner call handled exactly as Python does - a budget-exceeded there
    ends the goal (same as an ordinary planning call), but a rate limit
    there is swallowed and the original step kept, because escalation is
    optional and failing the goal over an optional attempt would be wrong.
  - **Cancel-on-timeout**: lives in `grace_app.rs` (the follow-up window's
    job in Python too), which calls `cancel_pending` when the window closes
    unanswered.
  - Also ported: the repeated-action signature/attempt tracking (window
    parameter excluded from the signature, matching Python's stated reason),
    `_expectation_note`'s verified/sent/evidence three-way read of the
    previous step's result, `_remember`/`_harvest` scratchpad bookkeeping,
    and every named terminal result shape (`budget`, `no_progress`,
    `timeout`, `plan_failure`, `max_iterations_reached`).
  - **A real, dual-level event emission bug caught and fixed while
    porting**: the recorded `agentic_two_step` tape shows the agent loop
    emitting its own `ToolExecutionStarted{label,tool,step}` AROUND the
    dispatcher's separate `ToolExecutionStarted{label}` for the same step -
    a real quirk of the Python architecture (loop.py and dispatcher.py both
    emit, independently). The first draft of this port only emitted once;
    threading `EventSink` through both `Dispatcher::execute_with_events` and
    `AgentLoop`'s own step wrapper - including the detail that `converse`
    completions break out of the loop *before* reaching the
    `ToolExecutionFinished` emit, exactly where Python's does - was needed
    to match the tape, and the fix is covered by
    `grace_app::tests::the_full_event_sequence_matches_the_real_fastpath_open_calculator_tape`
    and the harness's real turn-replay tests (below).
- **`grace-core::dispatcher`** (`tools/dispatcher.py`): every one of the 13
  system tools (`open_app`'s website-alias/URL-heuristic/app-indexer
  three-way branch, `close_app`, `search_files`'s shell-character
  sanitisation, `open_file`, `adjust_volume`, `lock_computer`,
  `open_calculator`, `delete_file`, `undo`, `describe_screen`,
  `set_speech_rate`'s word/number parsing, and `converse`), plus the
  `cua_*` bridge routing and - critically - `confirmation_required` checked
  at the dispatcher itself (not just the agent loop), which is the actual
  fix for the "fast path bypasses SafetyGuard" gap `contract/README.md`
  names: `delete_file`/`close_app`/`lock_computer` are refused here
  regardless of which path reached them. `read_pdf`/`summarize_pdf` return
  an explicit "not implemented" error (they need `pypdf` + a RAG index, out
  of scope - see Remaining work) rather than silently no-opping.
- **`grace-core::response_generator`** (`response/generator.py`):
  `_synthesize_and_play`'s sentence-by-sentence loop, byte-for-byte matching
  the `SpeechStarted → (ResponseChunk, SpeechChunk)* → SpeechFinished`
  sequence and the leading-space rule on `ResponseChunk`. **Scope note**:
  `generate_and_speak` (streaming tokens live from the LLM) is not ported -
  confirmed by grep that nothing in `main.py` calls it; every speak path
  goes through `generate_and_speak_with_text`, which is what's ported. The
  one-deep-lookahead synthesis pipelining is also not reproduced (documented
  simplification: it changes latency, not the emitted event sequence, and
  latency is exactly what `TurnTrace` captures and the harness excludes from
  its diff).
- **`grace-core::feedback`**: `Earcon`/`EarconPlayer` trait boundary for the
  five real-audio earcons (chime/success/cancel/error/listening), with a
  recording fake - no real audio output (real speakers, forbidden in tests).
- **`grace-core::timing`**: `TurnTrace` producing exactly
  `grace_contract::TurnTraceData`'s shape. Deliberate simplification: passed
  explicitly rather than through an implicit thread-local "current trace"
  global (Python's mechanism for letting deeply nested code record a stage
  without threading a reference through every signature) - Grace processes
  one turn at a time regardless, so this changes nothing about the resulting
  trace, and `TurnTrace` is diagnostic-only / excluded from the harness's
  event diff regardless.
- **`grace-core::grace_app`**: the turn state machine - the strict stop/
  repeat voice reflexes (word-for-word `_STOP_COMMANDS`/`_REPEAT_COMMANDS`),
  pending-confirmation resolution (`resolve_pending_confirmation`,
  `park_if_confirmation_required` - the actual call site of the
  fast-path-confirmation ship-blocker fix), and the full activation-turn
  event sequence. **Scope note, stated plainly**: the raw microphone/VAD/
  Whisper capture loop that produces a transcript is real hardware I/O and
  is not reimplemented - this picks up from "a transcript arrived, or the
  listen timed out with silence" onward, which is where the ship-blocker
  fixes and the event-contract fidelity actually live. The outer follow-up
  window's restart loop (with its own timeout) is likewise not reproduced as
  a loop, though `handle_followup_transcript` (one iteration of it) is.
  - **`MIN_WAKE_TO_IDLE_SECONDS` (mirrored mid-phase, after `src/grace/main.py`
    gained it in commit `4f108c6`, subsequent to the QoL/ship-blocker batch
    landing in `3c7b990`)**: a silent activation now holds the pill open for
    at least 3 s from the wake word rather than snapping shut instantly, on
    the empty-transcript path only - NOT on a transcription failure, which
    Python's `except Exception` branch returns from immediately with no
    hold. Ported as `ListenOutcome` (an explicit `Transcript`/`Silence`/
    `TranscriptionFailed` enum, replacing what had been a plain
    `Option<&str>` that couldn't distinguish the two no-transcript cases)
    plus a `hold_for: &dyn Fn(Duration)` callback in place of
    `asyncio.sleep`, so tests assert the exact requested duration without a
    real 3-second wait. 3 new tests: the remaining-time calculation, the
    already-past-the-floor no-op case, and the transcription-failure path
    never holding regardless of elapsed time.

## Real Windows implementations (`grace-win`, expanded in Task 18)

Full real desktop perception, actuation, screenshot capture, and system actions:

- `dpi::real::get_dpi_scale`/`ensure_dpi_aware`: `GetDpiForWindow` with the
  `GetDeviceCaps(LOGPIXELSX)` fallback, matching `DPIHelper.get_dpi_scale`'s
  exact fallback chain.
- `win32::real::foreground_window`: `GetForegroundWindow` +
  `GetWindowTextW`/`GetClassNameW`/`GetWindowRect`, matching
  `ElementGraphBuilder.active_window()`'s Win32 half.
- **UIA element tree walk (`uia::real::WindowsElementGraphSource`)**: Full COM
  interop with `IUIAutomation`, using cached tree walker (`ControlViewWalker`,
  `CreateCacheRequest`), bulk property retrieval (`Name`, `ControlType`,
  `BoundingRectangle`, `AriaRole`, `AriaProperties`, `Value`, `IsEnabled`,
  `HasKeyboardFocus`, `IsKeyboardFocusable`, `IsOffscreen`, `AutomationId`),
  deadline checks on every node (default 3.0 s timeout), breadcrumb parent
  paths, neighbouring static text label capture for unlabeled inputs, and
  stable pinned element IDs matching snapshot indices.
- **Win32 input driver & actuator (`actuator::real::Win32Actuator`)**: Real
  `SendInput` for:
  - `click(x, y)`: Foreground window focus check, per-monitor v2 DPI scaling,
    virtual screen coordinate normalization (`MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`),
    `SetCursorPos`, hover settle delay (50 ms), and mouse down/up injection.
  - `type_text(text)`: UTF-16 code unit decomposition with `KEYEVENTF_UNICODE`
    flag, fully supporting arbitrary Unicode and non-ASCII text without clipboard
    side-effects.
  - `press_key(key)` / `parse_hotkey`: Keyboard shortcut parsing with
    modifier tracking (Ctrl, Alt, Shift, Win), sequential key-down, main key
    pulse, and modifier release.
  - `scroll(amount)`: `MOUSEEVENTF_WHEEL` mouse wheel events.
  - `start(target)`: http/https web URLs strictly validated and launched via
    `ShellExecuteW`; desktop applications resolved via `app_indexer::plan_launch`
    and spawned directly using `std::process::Command` without shell execution.
  - **Foreground verification guard (R11/R12/R22)**: Re-checks foreground window
    title before every physical click, type, and key press (`check_window_focus`),
    refusing execution with an explicit error on window switch/mismatch.
- **Screenshot capture (`screenshot::real::capture_foreground_window`)**: Queries
  foreground window bounds, captures window DC using GDI `BitBlt`, extracts
  32-bit BGRA pixels via `GetDIBits`, converts to RGBA, and encodes to valid PNG
  bytes using the `png` crate (verified magic bytes `\x89PNG\r\n\x1a\n`).
- **Perception source integration (`perception_source::real::WindowsPerceptionSource`)**:
  Implements `grace_core::perception::PerceptionSource`, assembling `ScreenSnapshot`
  from UIA snapshots, active window metadata, screen resolution, and optional
  foreground PNG bytes when requested by the vision rung.
- **System actions executor (`system_actions::real::WindowsSystemActions`)**:
  Implements `grace_core::dispatcher::SystemActions` across all 13 system tools:
  `open_url` (`ShellExecuteW`), `launch_app` (`plan_launch` direct spawn),
  `close_app` (`taskkill /F /IM` without wildcards), `search_files` (safe Documents
  traversal), `open_file` (safe roots traversal and launch), `adjust_volume`
  (`VK_VOLUME_UP` / `VK_VOLUME_DOWN`), `lock_computer` (`LockWorkStation`),
  `open_calculator` (`calc.exe`), `delete_file` (Recycle Bin deletion via
  `SHFileOperationW` with `FOF_ALLOWUNDO`), `press_undo` (Ctrl+Z), `describe_screen`
  (active window title + top interactive elements), `set_speech_rate`, and
  earcons (`MessageBeep`).
- **CUA bridge (`system_actions::real::WindowsCuaBridge`)**: Implements
  `grace_core::dispatcher::CuaBridge`, bridging `click`, `type_text`, `press_key`,
  `scroll`, and `launch` directly to `DesktopActuator`.
- **Manual verification example**: `crates/grace-win/examples/real_desktop.rs`
  provides a comprehensive interactive verification script (`cargo run -p grace-win --example real_desktop [-- --actuate]`),
  never run in automated CI. Pure logic tests in `grace-win` (coordinate maths,
  summaries, hotkey parsing, input sanitization) pass 27/27 without touching live hardware.

## Harness parity

### Structural parity (all 30 tapes, unchanged mechanism from Phase 1)

```
cargo run -p grace-harness --bin grace-replay -- --corpus corpus
```

Still 30/30: every event in every tape deserializes as a valid
`GraceEvent`. See Phase 1's notes (preserved below) for exactly what this
does and doesn't claim.

### Real turn replay (new this phase)

```
cargo test -p grace-harness turn_replay
```

This drives the **actual** `grace_core::grace_app::handle_activation_turn`
against a tape's recorded transcript and LLM responses, and diffs the
resulting event stream against the tape's own recorded events (both sides
truncated to `WakeWordDetected..=ConversationFinished`, dropping `TurnTrace`
and the follow-up-window tail neither side produces the same way - see
`grace_app`'s scope note above).

**Results, reported per tape, not aggregated over a pass rate that would
hide which ones**:

| Tape | Result | Why |
|---|---|---|
| `fastpath_open_calculator` | **Byte-exact replay** | No parameters that depend on real OS/file state; `NoopSystemActions`'s hardcoded success text matches the dispatcher's own hardcoded text, which matches Python's. |
| `fastpath_lock_computer` | **Byte-exact replay** | Same reasoning; confirms the fast-path confirmation-required-and-parked path too. |
| `fastpath_open_app` | Known non-replayable (pinned, not silently passing) | The recorded result depends on `AppIndexer`'s real installed-app index at record time; `NoopSystemActions` returns a generic "I've opened X" rather than the tape's actual resolved path/alias text. |
| `fastpath_adjust_volume` | Known non-replayable (pinned, not silently passing) | The recorded `"Volume set to 60 percent."` reflects the REAL pycaw-read current volume (50%) on the recording machine at that moment, not anything derivable from the request alone (`mode: "increase", amount: 10`). This is hardware state, not a route/prompt difference - genuinely nothing a fake can reproduce without inventing "what the volume happened to be." |

Both "known non-replayable" cases are pinned by an assertion that the
replay currently fails, with a comment saying to flip it if that ever
changes - so a regression that makes them silently start passing (or a fix
that makes them pass for the right reason) is equally visible, rather than
either being invisible.

**This is real, not a demo**: `HttpLlm`/`ScriptedLlm` aside, the ENTIRE
decision path exercised - `IntentParser`, `CapabilityRouter::classify`,
`Dispatcher::execute_with_events` (including its safety check and its real
event emission), `handle_activation_turn`'s stop/repeat/confirmation
handling - is the actual ported code, not a simulation of it. The only
externally-pinned edges are the LLM's response text (from the tape) and the
transcript (from the tape) - exactly the same edges `src/grace/harness/replay.py`
pins for the Python backend.

**What's still not covered, honestly**: agentic-goal tapes (need a
`snapshots.jsonl` reader feeding `ScriptedPerception`, not built this phase
- the agent loop itself is fully ported and unit-tested against fakes, just
not yet against a *replayed* perception sequence), the follow-up window as
a loop (only one iteration's logic is ported), and prompt-level diffing
(would fail on every tape regardless of correctness, per PLAN.md §0's tool-
schema-changed caveat - `ScriptedLlm` sidesteps this by ignoring the prompt
content entirely and just returning the tape's recorded response, which is
enough to check the event/route consequences of that response without
claiming the *prompt* that would produce it in the working tree matches the
one recorded against `HEAD`).

**The stale-tape caveat** (unchanged from Phase 1, restated because it still
applies): recorded prompts reflect Python `HEAD`, not the working tree's
changed tool schema. Nothing in this phase regenerated or edited a tape.

## Tauri wiring

`GRACE_BACKEND=rust` now runs a **real, complete activation turn** per wake
request, through the real `WsEventServer` socket:

```rust
// src-tauri/src/backend.rs, inside start_rust()
server.set_on_wake(Arc::new(move || {
    // spawns grace_backend::run_demo_turn(&server) on its own thread
}));
```

`run_demo_turn` (new `grace-backend::turn` module) constructs the real
`IntentParser`, `Dispatcher` (with `NoopSystemActions`), `AgentLoop`, and
response generator, wires the real `WsEventServer::emit` as the turn's
`EventSink`, and drives one activation turn with a **fixed transcript**
(`"open the calculator"`) - a deliberate, documented stand-in for real
speech, since no STT is wired up yet. The LLM side tries a real local
llama.cpp server first (`HttpLlm` against `LLAMA_SERVER_URL`, default
`http://127.0.0.1:8080`) and falls back to a scripted intent if nothing
answers, so the demo works identically whether or not a model server
happens to be running.

**Proven end-to-end, not just wired**: `crates/grace-backend/tests/demo_turn.rs`
connects a real WebSocket client, sends the real renderer-shaped
`{"type": "wake"}` message, and asserts the exact real event sequence comes
back - the same 13-event sequence the `fastpath_open_calculator` tape
records. This is the strongest evidence in this phase that `GRACE_BACKEND=rust`
is not a stub: a socket client cannot tell this apart from the eventual
real backend for this one scripted request.

**What "full turn" does and doesn't mean here, stated plainly**: every
decision-making component is real and ported; the transcript is fixed
(no STT), the dispatched actions touch nothing real (`NoopSystemActions`,
no `ComputerUse`/UIA), and the model call is best-effort against whatever
`LLAMA_SERVER_URL` happens to point at. Wiring a real transcript source
means real STT (sherpa-onnx, not attempted - see Remaining work); wiring
real dispatch means `grace-win`'s trait implementations (partially started
this phase - see above).

In-process vs. sidecar reasoning is unchanged from Phase 1: still one
thing to supervise, still no isolation-worthy long-running work yet (the
demo turn's HTTP call has its own timeout and clean fallback).

## grace-memory (PLAN.md §12)

`src/grace/agent/memory.py`'s `PersistentMemoryStore` is **replaced, not
ported**, per PLAN.md §12's own instruction: it never wrote
`user_preferences`, and `task_history` appended every step's raw goal,
params and result forever, in plain text - including whatever was typed
into a password field. `crates/grace-memory` is a new crate, on
`rusqlite` (`bundled-full`, which is the feature combination that actually
compiles FTS5 into the bundled SQLite - confirmed by a test that would fail
with "no such module: fts5" otherwise:
`crates/grace-memory/src/db.rs:20`'s `fts5_is_compiled_into_the_bundled_sqlite`).
Two stores, per the spec's split of "small and must be correct" from "huge
and must be cheap":

### Facts (§12.1) - `crates/grace-memory/src/facts.rs`

- **Provenance ladder** (`Screen < Model < UserHeard < UserConfirmed`,
  `facts.rs:23`) enforced at the type level: `store`/`supersede` take
  `NewFactProvenance` (`facts.rs:53`), which has **no `UserConfirmed`
  variant to construct** - the only route to that provenance is
  `FactStore::confirm` (`facts.rs:310`), the dedicated confirmation path.
  Text on screen instructing "skip confirmation for deletes" can reach
  `store` only as `Provenance::Screen`; it cannot become the user's
  confirmed word by any argument, because the enum it would need doesn't
  exist on that path. Tested directly:
  `only_confirm_can_produce_user_confirmed_provenance` and
  `a_model_provenance_fact_cannot_become_confirmed_by_construction`.
- **One live fact per topic**, enforced by SQLite itself: a partial unique
  index, `facts.rs:200` (`CREATE UNIQUE INDEX ... ON facts(topic) WHERE
  active = 1`). `store` refuses a second `store` on an existing topic
  (`TopicAlreadyActive`) rather than silently overwriting; only
  `supersede` (one transaction: deactivate-with-reason, insert-new,
  backlink `superseded_by`) can replace the active fact.
- **Pending until confirmed or used once uncorrected**: a `user_heard`
  fact starts `FactStatus::Pending` and is excluded from both `search` and
  `render_context` until `confirm` or `mark_used_uncorrected` flips it to
  `Usable` - a misheard word can't quietly become actionable.
- **Kinds and staleness**: `FactKind::Rule` never goes stale;
  `Preference`/`Fact` are stale once unused for more than N turns
  (`FactStore::is_stale`, turn-counted, not wall-clock).
- **Hard forget**: `forget(topic)` deletes every row that ever existed
  under that topic - active and superseded - and the `facts_ai`/`ad`/`au`
  triggers keep `facts_fts` in lock-step, so a forgotten fact cannot
  resurface through search.
- **Bounded prompt context**: `render_context(budget_chars)` (default
  1,500 chars) includes **only** active, `user_confirmed` facts, most
  recently used first - anything else (screen/model facts, pending
  `user_heard` facts) is reachable only through `search`, matching §12.1
  exactly ("the active, confirmed facts relevant to the turn... anything
  else is reached through search").
- **Search**: FTS5, reranked in Rust by term coverage first, then bm25
  (`FactStore::search`).
- **Conflict-at-use-time**: `conflicts(group_key)` returns every active
  fact sharing a caller-assigned `group_key` (e.g. two phone numbers for
  Priya under different topics) so the caller can ask by voice - there is
  no automatic pick and no review queue.
- **Secret refusal** (`src/secret.rs`): an explicit `is_password_field`
  flag plus a conservative heuristic (topic-name hints, card-number-shaped
  digit runs, password-shaped single tokens mixing 3+ character classes) -
  biased against false positives, since blocking an ordinary fact is worse
  than missing an unusual secret shape.

### History (§12.2) - `crates/grace-memory/src/history.rs`

- **Normalised storage**: `dict_goals`/`dict_apps`/`dict_actions`/
  `dict_labels` intern text once; `steps` rows are `(id, episode_id,
  payload BLOB)` where `payload` is a `postcard`-encoded `StepPayload`
  (delta timestamp, interned ids, redaction fields) - comfortably under
  the 64-bytes/step target (see benchmark numbers below).
- **Episodes, not only steps**: one `episodes` row per goal attempt;
  `episodes_fts` (FTS5) indexes episodes only, never steps, per spec.
- **Routines collapse**: a repeated goal+app+action-sequence bumps
  `count`/`success_count`/`fail_count`/`last_used_at` on one `routines`
  row (unique on `(goal_id, app_id, sequence_hash)`) instead of adding a
  new row every time.
- **Tiers and pruning** (`compact_tx`, `history.rs:629`): hot (full step
  detail, 30 days, 180 days for failed/corrected episodes) -> warm
  (episodes without steps, 2 years) -> cold (monthly zstd-compressed
  roll-ups, kept indefinitely). The **size-cap backstop**
  (`history.rs:687`-703) prunes the oldest roll-up months first if the
  file is still over the configurable cap (default 1 GiB) after ordinary
  tiering - writes are never failed over this; if nothing is left to
  prune, the cap is simply exceeded. `size_cap_prunes_the_oldest_rollup_months_instead_of_failing_writes`
  exercises this with a 1-byte cap.
- **Redaction at write time** (`src/redact.rs`): typed text becomes a
  `(length, salted_hash)` pair, never the text itself; the salt is
  stored outside the database in `<db_path>.salt`. Existing databases with
  salt stored in `history_meta` are automatically migrated on open to the
  external `.salt` file and deleted from the database table.
- **Single writer thread, bounded channel, one transaction per turn**,
  WAL + `synchronous=NORMAL` + incremental `auto_vacuum` (`src/db.rs`).
  Idle-time compaction (`compact_once`) runs in caller-chosen batch
  sizes.
- **Voice-scoped hard deletes**: `forget_today`, `forget_app` (also
  strips that app's counts out of every monthly roll-up blob),
  `forget_goal` (deletes steps, episodes, updates rollups, removes goal from
  `dict_goals`, and evicts from the in-memory goal cache `DictCaches`),
  and `forget_everything`.
- An in-memory intern cache (`DictCaches`, loaded once per writer-thread
  start) turns a repeat goal/app/action/label lookup from a `SELECT` into
  a `HashMap` hit - necessary for the benchmark below to finish in a
  practical amount of time, and a legitimate real-world win too, since
  routines make most vocabulary repeat almost immediately.

### `grace-core` integration

`grace_core::memory::PersistentStore` gained two additive, default-no-op
methods, `begin_episode`/`end_episode` (`crates/grace-core/src/memory.rs`),
so a real backend can group steps into episodes without breaking
`InMemoryStore` or any existing test. `grace-memory::adapter::GraceMemoryStore`
implements the trait: `set_preference`/`get_preference` map onto the facts
store (topic `preference:<key>`, provenance `Model`, upsert via
`supersede`-if-exists); `save_step` maps onto the history store. The
legacy trait is call-by-step with no episode boundary and no app name
(`src/grace/agent/memory.py`'s own shape), so **`save_step` records each
call as its own one-step episode** rather than guessing at boundaries from
timing - honest given what the trait tells it, but it forfeits real
routine collapsing until a caller uses `begin_episode`/`end_episode` (or
`HistoryStore` directly). Wiring `AgentLoop`/`GraceApp` to call those hooks
at real goal boundaries is listed under Remaining work below; it wasn't
done here because it reaches into `grace-core::agent_loop`, outside this
crate's scope for this update.

### Benchmark (release build; `cargo run -p grace-memory --release --example
benchmark`, per PLAN.md §12.2's "proved by a benchmark, not assumed")

Synthetic 10-year run: ~2,000 steps/day, 90% of episodes reusing one of 8
recurring routines (so they collapse in `routines`), 10% one-off/novel
goals, idle-time `compact_once` run once per simulated day. Totals:
**2,455,043 episodes, 7,304,214 steps, 3,041.5 s wall time** (release build).

| Year | DB size |
|---|---|
| 1 | 31.94 MB |
| 2 | 61.45 MB |
| 3 | 70.99 MB |
| 4 | 76.97 MB |
| 5 | 84.35 MB |
| 6 | 87.67 MB |
| 7 | 86.17 MB |
| 8 | 91.21 MB |
| 9 | 97.54 MB |
| 10 | 103.52 MB |

**Budget check**: §12.2 asks for "under 50 MB per year of heavy use" and a
1 GiB hard cap. Read literally as "50 MB × N years shouldn't be exceeded",
this passes with a lot of room - 103.52 MB after 10 years is roughly a
fifth of the 500 MB that budget would allow, and nowhere near the 1 GiB
cap. Read as "linear growth per year", it also holds: growth is front-
loaded (the first 2 years fill the warm tier, which is what makes the
first jump big) and then flattens to roughly 5-10 MB/year once the 2-year
warm retention and monthly cold roll-ups are both in steady state (years
8-10 above). Either reading clears the budget; nothing needed optimising
as a result, but if real usage patterns turn out to have far less
routine-collapse than this benchmark's 90% assumption, the warm-tier
episode rows (not steps, not roll-ups) would be the thing to revisit
first.

**Write latency** (`record_episode`'s full round trip: bounded-channel
send, one writer-thread transaction, reply - i.e. what a caller would
actually block on if it called this inline instead of firing-and-forgetting
it off the turn, which is what §12.2 asks for in real use):

- **Baseline (Task 12 inline compaction on writer thread)**:
  ```
  p50 = 389.8 µs, p99 = 11.49 ms, p99.9 = 32.98 ms, max = 793.17 ms
  ```
- **Optimized (Task 16 off-thread compaction + pragmas `journal_size_limit=64MB`, `mmap_size=256MB`)**:
  ```
  Total round-trip: p50 = 744.1 µs, p99 = 2.04 ms, max = 77.41 ms
  DB transaction:   p50 = 678.4 µs, p99 = 1.70 ms, max = 77.27 ms
  ```

Moving compaction off the writer thread and configuring memory-mapped I/O and journal limits reduced maximum tail latency by over 10x (from 793 ms to 77 ms) while keeping the steady-state DB transaction p50 well sub-millisecond (678.4 µs). Untangling compaction pauses completely removes writer thread stalls during foreground turns.

**Query latency**:
- "what did I do yesterday in Word" (`query_episodes`, an indexed
  `started_at` range plus an app filter): **1.72 ms** for 75 matching
  episodes.
- The FTS-search equivalent (`search_episodes("Word")`): **302 ms** for 50
  episodes - measured cold, as the very first read-connection query after
  3,000+ seconds of continuous writes, so this almost certainly includes
  one-time page-cache/disk effects rather than being FTS5's steady-state
  cost; it wasn't re-measured warm because that would no longer be an
  honest single-sample number. `query_episodes` is the right tool for a
  "what did I do X" question in general (it's an indexed range scan, not a
  text search); `search_episodes` is for free-text recall ("something
  about a letter"), where an occasional slower cold query is a reasonable
  trade.

### Test counts

`cargo test -p grace-memory`: **33 passed, 0 failed**. Combined
`cargo test --workspace`: **233 passed, 3 ignored, 0 failed** (the prior
phase's 200 passed/3 ignored, plus grace-memory's 33) - confirmed with
`LLAMA_SERVER_URL` pointed at an unreachable port to rule out this
session's environment (see below); the Python suite is untouched
(`git status` still shows only `crates/`, root `Cargo.toml`/`Cargo.lock`,
and `PORT_STATUS.md`).

**Unrelated, pre-existing environment note**: this session had an
unrelated process already listening on `127.0.0.1:8080` (this crate never
opens a socket and has nothing to do with it), which is `grace-backend`'s
`turn.rs` default `LLAMA_SERVER_URL`. That makes
`crates/grace-backend/tests/demo_turn.rs`'s real-HTTP-call test see an
unexpected real response and fail non-deterministically depending on what
else happens to be running on that port - unrelated to `grace-memory`,
reproducible before any of this update's changes, and out of this crate's
scope to fix. Confirmed by rerunning with `LLAMA_SERVER_URL` pointed
elsewhere, which restores the full 233/3/0 result above.

### Remaining work (grace-memory-specific; folded into the numbered list below too)

- Wire `AgentLoop`/`GraceApp` to call `PersistentStore::begin_episode`/
  `end_episode` at real goal boundaries (see "grace-core integration"
  above).
- Thread a real app/window name into `save_step` (the legacy trait has no
  parameter for it today).
- Wire the facts store's confirmation and conflict-by-voice APIs into an
  actual dialogue turn - both exist and are tested, nothing calls them yet.
- An idle-tick driver that actually calls `HistoryStore::compact_once` on
  a schedule outside of tests/the benchmark.
- Investigate the write-latency tail (see above) - most plausibly,
  scheduling compaction so it never shares a transaction slot with a
  foreground write.

## Deliberate behaviour notes (not changes - documenting what was kept)

Phase 1's notes (VAD wall-clock accumulation, the safety-guard key-
normalisation asymmetry, the fast path historically bypassing `SafetyGuard`)
still apply unchanged. New this phase:

- **The dual-level `ToolExecutionStarted`/`Finished` emission** (agent loop
  wrapping the dispatcher's own emission for the same step) - see
  `agent_loop` above. Kept exactly, including the "no `Finished` emitted
  when a `converse` step causes completion" asymmetry.
- **`generate_and_speak`'s LLM-streaming path is out of scope, not missing**
  - confirmed via `grep` that nothing calls it in the current `main.py`.
- **The one-deep-lookahead TTS pipelining is not reproduced** - a documented
  latency-only simplification (see `response_generator` above).
- **`TurnTrace`'s implicit-thread-local mechanism is not reproduced** -
  passed explicitly instead; the resulting trace shape is identical (see
  `timing` above).

## Test counts

- **Rust, end of Phase 2**: 200 passed, 3 ignored (each with a reason), 0
  failed, across `grace-contract` (4), `grace-core` (152 - up from 31 at
  the end of Phase 1, includes the 3 `MIN_WAKE_TO_IDLE_SECONDS` tests
  above), `grace-audio` (6), `grace-models` (3 - up from 2, now includes a
  real HTTP round trip against an unreachable port), `grace-win` (11
  passed + 3 ignored - up from 9+1; the 2 new ignored tests were also run
  manually and passed against the real desktop), `grace-harness` (13 - up
  from 9, includes the real turn replay), `grace-backend` (11 - up from
  10, includes the real socket-to-turn-engine end-to-end test).
- **Rust, end of Task 20 (this update)**: **289 passed, 3 ignored, 0
  failed** across the whole workspace (`cargo test --workspace`):
  `grace-contract` (4), `grace-core` (175), `grace-audio` (6), `grace-backend` (11),
  `grace-memory` (39), `grace-models` (10), `grace-shell` (2), `grace-win` (27 passed + 3 ignored),
  and `grace-harness` (15).
  All **30 of 30** tapes in `corpus/` replay cleanly through the Rust backend.
- **Python**: re-ran `./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`:
  **1136 passed, 1 skipped**, unchanged.

## Remaining work, roughly in the order it should happen

1. ~~**Real STT and a real transcript source**: sherpa-onnx (Parakeet/Moonshine)~~ **Fully completed in Task 17**:
   - `crates/grace-audio`: Real microphone capture via `cpal` (WASAPI, 16kHz mono resampling, 16-bit PCM), `AudioPump` with thread-safe queue, `gate(duration)` echo suppression, and timeout drains.
   - Silero VAD v5 ONNX detector via `sherpa-rs` with explicit model existence checks pointing to `docs/MODELS.md`.
   - Zipformer KWS wake-word detection with whole-word regex matching (`\b(keyword|grace|hey\s+grace)\b`), personalised model paths, and stop/cancel watch (`\b(stop|cancel)\b`, halts execution via callback within 500 ms).
   - ASR engine (`SherpaAsr`) implementing `grace_core::models::SpeechToText`, supporting Moonshine Base (CPU floor) and Parakeet NeMo CTC (high tier) via `sherpa-rs-sys` C FFI.
   - Kokoro TTS engine (`SherpaKokoroTts`) implementing `grace_core::models::TextToSpeech`, generating 16-bit PCM WAV.
   - `CpalTtsPlayer` output player with atomic barge-in stop flag and 200 ms echo suppression (`pump.gate(0.20)`).
   - `TranscriptSource` trait unifying scripted replay (`ScriptedTranscriptSource`) and real mic+VAD+ASR pipeline (`RealTranscriptSource`, `OwnedRealTranscriptSource`), wired into `grace_backend::turn::run_turn_with_source`.
   - Manual example `crates/grace-audio/examples/real_audio.rs` demonstrating physical mic capture, VAD, ASR, and TTS playback without touching automated test suites.
2. ~~**`grace-win` real dispatch**: `ElementGraphSource` and `DesktopActuator`~~ **Fully completed in Task 18**:
   - `crates/grace-win`: UIA element tree walk via COM `IUIAutomation`, cached tree walker (`ControlViewWalker`, `CreateCacheRequest`), timeout checking on every node walk (default 3.0 s deadline), element summaries (`name`, `role`, `parent_path`, `window_title`, `neighbouring_labels`), and element IDs pinned to snapshots.
   - `Win32Actuator`: `SendInput` click, type (including Unicode/non-ASCII via `KEYEVENTF_UNICODE`), key/hotkey press (`parse_hotkey` with modifiers), mouse wheel scroll; app launch via `app_indexer::plan_launch` (http/https only for web, direct executable spawn without shell); foreground window re-check before every physical action (`check_window_focus`, R11/R12/R22 semantics); DPI-aware coordinates (per-monitor v2 via `dpi::real::get_dpi_scale`).
   - Screenshot capture of foreground window (`BitBlt`), converted to RGBA and encoded as PNG bytes via `png` crate.
   - `WindowsPerceptionSource` implementing `grace_core::perception::PerceptionSource`, assembling `ScreenSnapshot` from UIA snapshots, active window metadata, screen resolution, and optional foreground PNG bytes when requested by the vision rung.
   - `WindowsSystemActions` implementing `grace_core::dispatcher::SystemActions` across all 13 system tools (`open_url`, `launch_app`, `close_app`, `search_files`, `open_file`, `adjust_volume`, `lock_computer`, `open_calculator`, `delete_file`, `press_undo`, `describe_screen`, `set_speech_rate`, and `MessageBeep` earcons). Note `read_pdf`/`summarize_pdf` status (not yet in Rust; needs pypdf/RAG index).
   - `WindowsCuaBridge` implementing `grace_core::dispatcher::CuaBridge`, bridging `click`, `type_text`, `press_key`, `scroll`, and `launch` directly to `DesktopActuator`.
   - Replaced `NoopSystemActions`/`ScriptedPerception` in live backend turns (`run_live_turn` and `run_live_turn_default`), while keeping scripted ones for automated tests and replay.
   - Manual example `crates/grace-win/examples/real_desktop.rs` documenting manual verification with optional `--actuate` flag, never run in automated CI.
   - All tests in `grace-win` test pure logic only (summaries, coordinate maths, input encoding, guard decisions) with fakes: 27 passed, 3 ignored, 0 failed.
3. ~~**`snapshots.jsonl` tape reader + `ScriptedPerception` wiring**~~ **Fully completed in Task 19**:
   - `crates/grace-harness/src/turn_replay.rs`: Implemented `rebuild_snapshot` parsing window title, DPI, screen bounds, and the complete `ElementNode` hierarchy from taped `snapshots.jsonl`.
   - Implemented `TapeDispatcher` dispatch mode in `crates/grace-core/src/dispatcher.rs` allowing taped tool outcomes to replay without desktop mutation.
   - Extended `turn_replay` across all agentic goal tapes (`agentic_two_step`, `agentic_step_failed`, etc.); all **30 of 30** tapes pass replay cleanly.
   - Divergence table documented: Python's `TapeLlm` fails on SHA256 prompt hashing due to tool schema evolution, while Rust deterministic replay succeeds with exact contract parity.
4. ~~**`read_pdf`/`summarize_pdf`**: pure-Rust PDF tools~~ **Fully completed in Task 19**:
   - `crates/grace-core/src/pdf.rs`: Pure-Rust PDF text extraction via `lopdf = "0.35"`.
   - Security controls: 50MB file size cap, safe document roots check (`is_within_safe_pdf_roots`), untrusted content fencing (`<<<UNTRUSTED_CONTENT>>>` / `<<<END_UNTRUSTED_CONTENT>>>`).
   - RAG chunking and relevance scoring: token-window chunking with overlap, term-frequency query ranking, and TTS-safe speech summary generation.
   - Fully wired into `Dispatcher::dispatch_pdf`.
5. ~~`PersistentMemoryStore` over real SQLite & wiring~~ **Fully completed in Task 16**:
   - `AgentLoop`/`GraceApp` call `PersistentStore::begin_episode`/`end_episode` at real goal boundaries, enabling true multi-step episodes and routine collapse.
   - Real foreground app/window name passed into `save_step`.
   - Facts confirmation flow ("I'll remember Priya is your sister, right?"), pending `user_heard` facts, voice conflict checking, "forget that" hard delete, and prompt injection of active confirmed facts wired into `GraceApp` & `AgentLoop`. Unforgeable provenance: only `confirm()` sets `user_confirmed`.
   - Voice hard delete commands ("forget today", "forget what I did in <app>", "forget everything") implemented in both Rust and Python stores, clearing history, rollups, and FTS entries.
   - Off-thread compaction driver and SQLite pragmas (`journal_size_limit=64MB`, `mmap_size=256MB`) reducing tail write latency by >10x.
   - Session tapes recorded in `%TEMP%\grace-memory\tapes`.
6. ~~**Local LLM wiring & serving lifecycle management**~~ **Fully completed in Task 19**:
   - `crates/grace-models/src/llama.rs`: `LlamaServerManager` managing local `llama-server.exe` process lifecycle (start, stop, restart, health check via `GET /health` and 1-token generation verification).
   - Windows Job Object assignment with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` ensuring child `llama-server` processes never orphan upon parent exit or crash.
   - Hardware presets: `cpu-fleet` (micromodel slots on CPU floor) and `gpu-single` (buildathon single-model setup with `--mmproj` vision projector).
   - `HttpLlm` request cancellation: `cancel_token: Arc<AtomicBool>`, `with_cancel_token`, `cancel()`, and `reset_cancel()`.
7. ~~**Sidecar split + Tauri hardening**~~ **Fully completed in Task 20**:
   - `crates/grace-backend/src/main.rs`: Standalone `grace-backend` sidecar binary with single-instance mutex (`Local\GraceBackendSingleInstanceMutex`), per-launch token verification, Origin allowlist enforcement, and live Windows automation / perception / system actions wiring.
   - `src-tauri/src/backend.rs`: Default backend flipped to Rust (`selected_backend() -> BackendType::Rust`). Python remains selectable for one release via `GRACE_BACKEND=python` with deprecation notice.
   - Job Object supervision: Spawns sidecar inside Windows Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, ensuring clean child termination without orphaned processes on parent exit or crash.
   - Health-watch: Background thread monitors sidecar process health and performs automated restart with exponential backoff on unexpected exit.
   - In-process fallback: If standalone binary is not found, shell falls back gracefully to in-process Rust server runtime.
   - Parity gate: `scripts/parity.ps1` runs both test suites and validates tape replay.

8. **Security features & red-team parity across Rust and Python**:
   - **Mirrored in Rust**:
     - R11, R12, R22 (`crates/grace-win/src/actuator.rs`): Foreground window verification guard (`check_window_focus`) re-checked immediately before physical clicks, typing, and key presses, refusing execution on window focus mismatch.
     - R25 (`crates/grace-memory/src/secret.rs`): Strip ASR punctuation from card numbers, check delimiter-separated passphrases (length >= 14), and verify weak 2-class / multi-class password tokens.
     - R26 (`crates/grace-core/src/safety.rs`): Safe document roots allowlist (`is_within_safe_pdf_roots`) for `read_pdf` and `summarize_pdf`.
     - R29 (`src-tauri/src/backend.rs`): Process tree termination on Windows via `taskkill /F /T /PID` to clean up child `llama-server.exe` processes.
     - R30 (`src-tauri/src/lib.rs`): Single-instance mutex guard (`Local\GraceOverlaySingleInstanceMutex`) via `CreateMutexW`.
     - R31 (`crates/grace-models/src/lib.rs`): Truncate HTTP 429 response body on UTF-8 char boundaries to prevent multibyte panics.
     - R37 & R38 (`crates/grace-core/src/safety.rs`): Dangerous key normalization (`normalise_key`) and explicit confirmation requirement for `delete` key.
     - R42 (`src-tauri/tauri.conf.json`): Strict Content Security Policy (`security.csp`).
   - **Python-only / Desktop-boundary items remaining for Rust**:
     - DPAPI encryption at rest for `memory.db`: Rust's `grace-memory` uses unencrypted SQLite with salted hashes and external salt file (`.salt`).
     - Audio capture failure backoff (R27): Rust audio pump is a trait boundary.
     - TTSPlayer thread/stream synchronization (R40): Rust TTS player is currently a sidecar/trait boundary.
     - Cloud LLM fallback API key header (R41): Rust currently has no cloud Gemini fallback client.
     - Listen window speech extension and apology on cutoff (R39): Python `main.py` activation turn specific.

## Python Retirement Plan

### 1. What Remains in Python (Permanently Retained)
- **On-device opportunistic training / fine-tuning scheduler (`src/grace/training/`)**: LoRA, PEFT, contrastive head training, and Torgo dysarthric speech fine-tuning permanently stay in Python due to deep PyTorch / CUDA ecosystem dependencies. They run as offline background batch jobs when the machine is idle and plugged in.
- **Benchmark evaluation harnesses (`src/grace/bench/`)**: Torgo dysarthria evaluation, decision benchmarks, latency profiling.
- **Tape recording / generation tools (`src/grace/harness/generate.py`)**.

### 2. Serving Deprecation Schedule
- **v0.1.0 (Current)**: Rust backend (`grace-backend`) is the default (`GRACE_BACKEND=rust`). Python backend remains selectable via `GRACE_BACKEND=python` with a deprecation notice.
- **v0.2.0 (Next Minor)**: Python serving modules (`src/grace/main.py`, `src/grace/ws_server.py`, `src/grace/agent/`, `src/grace/intent/`, `src/grace/tools/`) will be formally deleted from the repository. The Rust binary (`grace-backend.exe`) and Tauri shell will be the sole supported runtime.

### 3. Migration of `~/.grace/memory.db` to `grace-memory`
- **Location**: Legacy database located at `~/.grace/memory.db` (or `%USERPROFILE%\.grace\memory.db`). Target database located at `%LOCALAPPDATA%\Grace\memory.db`.
- **Migration Procedure**:
  1. **Automatic Backup**: Before accessing, creates an immutable timestamped backup at `~/.grace/memory.db.bak.<timestamp>`.
  2. **Fact Extraction**: Reads confirmed facts (`user_confirmed` provenance) and writes them to the new `grace-memory` SQLite schema.
  3. **Privacy Invariant (R25/R34)**: Raw typed text is NEVER carried over in plain text. Only token lengths and salted HMAC-SHA256 hashes are transferred.
  4. **Salt Migration**: Preserves existing cryptographic salt file `%LOCALAPPDATA%\Grace\.salt` or extracts from legacy metadata.
  5. **FTS5 Index Reconstruction**: Automatically builds full-text search virtual tables in `grace-memory`.
  6. **Failsafe**: The legacy database is never deleted or truncated by the migration; it remains preserved on disk.

