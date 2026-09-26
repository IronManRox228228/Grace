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
| `crates/grace-core` | ~5,700 | `config.py`, `agent/safety.py`, `agent/loop.py`, `agent/memory.py`, `agent/perception.py`, `agent/planner.py`, `agent/grounder.py`, `agent/ui_tars_parser.py`, `intent/parser.py`, `intent/tools.py`, `intent/router.py`, `tools/dispatcher.py`, `response/generator.py`, `response/feedback.py`, `util/timing.py`, `text/sentence_split.py`, `main.py`'s `GraceApp` turn logic, `perception/elements.py`, `perception/element_graph.py` | The whole turn pipeline's decision logic. See below, module by module. |
| `crates/grace-audio` | ~250 | `vad/detector.py` | Unchanged since Phase 1: `VadDetector` fully ported; capture/pump/wake-word are trait boundaries. |
| `crates/grace-models` | ~250 | `llm/gemma_client.py`'s HTTP shape | `HttpLlm`: a REAL (not stubbed) blocking HTTP client against a local llama.cpp server's `/v1/chat/completions`, non-streaming, matching `GemmaClient.chat(..., stream=False)`'s exact request shape including the `image_b64` → multimodal `image_url` reshaping. `ElementScorer` trait for a future GLiNER-style scorer, not implemented. |
| `crates/grace-win` | ~600 | `automation/app_indexer.py`, `automation/dpi_helper.py`, a slice of `perception/element_graph.py`'s `active_window()` | `plan_launch`/`find_app` (Phase 1). **New this phase**: `dpi::real::get_dpi_scale`/`ensure_dpi_aware` and `win32::real::foreground_window` - real `windows`-crate Win32 calls, manually verified against the real desktop (see below), plus `dpi::scale_coords` (pure, unit-tested). UIA tree walk, the win32 input driver, and computer-use actions remain trait boundaries. |
| `crates/grace-harness` | ~750 | `harness/tape.py`, `harness/replay.py` | Tape loader + diff machinery (Phase 1). **New this phase**: `turn_replay.rs` drives the real `grace_core` turn engine against real tapes and diffs the resulting event stream - see "Harness parity" below. |
| `crates/grace-backend` | ~750 | `ws_server.py`'s Origin allowlist | `WsEventServer` (Phase 1). **New this phase**: `turn.rs`'s `run_demo_turn` wires the real turn engine to the real socket - see "Tauri wiring" below. |
| `src-tauri/` | (small diff) | `main.js:93-141` | `Backend::start` branches on `GRACE_BACKEND`; the Rust path now runs a real turn per wake request instead of emitting one static `Idle`. |

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
- **`grace-core::ui_tars_parser`** (`agent/ui_tars_parser.py`): every one of
  UI-TARS's Thought/Action patterns - box-token and named clicks (with the
  click-kind preserved: single/double/right/hover), scroll, drag, hotkey
  (including the `"ctrl a"` → `"ctrl+a"` space-to-plus join), type, wait,
  and `finished()`/`stop()` completion. 12 tests.
- **`grace-core::planner`** (`agent/planner.py`): `PlannedStep`,
  `needs_grounding`'s exact three-way gate (has `element_id`? has both `x`
  and `y`? else does it name a `target_name`/`describe`?), the full system
  prompt, `build_prompt`'s section ordering, the per-goal call budget
  (0/negative = unlimited, matching `math.inf`), and `parse_planned_step`'s
  fence-then-brace-scan fallback for a model that wraps JSON in prose.
- **`grace-core::grounder`** (`agent/grounder.py`): `scale_to_screen`
  (including the "coordinates exceed the image, treat as 0-1000 normalised"
  overflow branch), `describe_target`'s chrome/page/role/name phrase
  building, and `Grounder::locate`'s UI-TARS-native prompt/parse round trip.
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

## Real Windows implementations (`grace-win`, new this phase)

Two real `windows`-crate calls, chosen because they're safe (no COM, no
input injection) and self-contained:

- `dpi::real::get_dpi_scale`/`ensure_dpi_aware`: `GetDpiForWindow` with the
  `GetDeviceCaps(LOGPIXELSX)` fallback, matching `DPIHelper.get_dpi_scale`'s
  exact fallback chain.
- `win32::real::foreground_window`: `GetForegroundWindow` +
  `GetWindowTextW`/`GetClassNameW`/`GetWindowRect`, matching
  `ElementGraphBuilder.active_window()`'s Win32 half (not the UIA tree walk
  that follows it).

Both are `unsafe` FFI and cannot be part of the automated suite (the task's
hard constraint), so each has a documented `#[ignore]`d placeholder test.
**Both were run manually against this session's real desktop** (`cargo test
-p grace-win --lib -- --ignored --test-threads=1`) and returned real,
sane values (a DPI scale > 0, a non-`None` foreground window) - this is
working code, verified once, not merely code that compiles under `cfg(windows)`.

The full UIA tree walk (`IUIAutomation` COM interop), the win32 input driver
(`SendInput`), and DPI-awareness-at-process-startup ordering remain trait
boundaries only - COM interop in particular is enough additional surface
area (apartment threading, interface lifetimes) that attempting it in the
time remaining this phase risked a half-working implementation, which is
worse than an honest stub.

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

- **Rust**: **200 passed, 3 ignored (each with a reason), 0 failed**, across
  `grace-contract` (4), `grace-core` (152 - up from 31 at the end of Phase 1,
  includes the 3 `MIN_WAKE_TO_IDLE_SECONDS` tests above), `grace-audio` (6),
  `grace-models` (3 - up from 2, now includes a real HTTP round trip against
  an unreachable port), `grace-win` (11 passed + 3 ignored - up from 9+1;
  the 2 new ignored tests were also run manually and passed against the
  real desktop), `grace-harness` (13 - up from 9, includes the real turn
  replay), `grace-backend` (11 - up from 10, includes the real
  socket-to-turn-engine end-to-end test). Run with `cargo test --workspace`.
- **Python**: re-ran `./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`
  after the `MIN_WAKE_TO_IDLE_SECONDS` mirror too: **841 passed, 1 skipped** -
  still exactly PLAN.md §0's baseline, and `git status` confirms only
  `crates/`, `src-tauri/`, root `Cargo.toml`/`Cargo.lock`, and
  `PORT_STATUS.md` are touched by this work (the Python source of truth
  changed independently, in commits `3c7b990`/`4f108c6`, not by this port).

## Remaining work, roughly in the order it should happen

1. **Real STT and a real transcript source**: sherpa-onnx (Parakeet/
   Moonshine per PLAN §3) is the biggest single unlock - it's what turns
   `run_demo_turn`'s fixed transcript into a real one and makes
   `grace-audio`'s capture/pump traits worth implementing for real.
2. **`grace-win` real dispatch**: `ElementGraphSource` (the UIA tree walk -
   COM interop, `IUIAutomation`) and `DesktopActuator` (`SendInput` for
   click/type/key/scroll, plus the app-launch call `app_indexer::plan_launch`
   already decides on). This unblocks `NoopSystemActions`/`ScriptedPerception`
   being replaced with real ones in both the demo turn and the harness.
3. **`snapshots.jsonl` tape reader + `ScriptedPerception` wiring**: extends
   `turn_replay.rs` to agentic-goal tapes (`agentic_two_step`,
   `agentic_step_failed`, etc.) - the agent loop itself is fully ready for
   this; only the perception-replay plumbing is missing.
4. **`read_pdf`/`summarize_pdf`**: needs `pypdf`-equivalent extraction (a
   Rust PDF text crate) and the TF-IDF-ish RAG chunking `grace.rag` does;
   contained, no architectural risk.
5. **`PersistentMemoryStore` over real SQLite** (`rusqlite`, bundled): low
   risk, not on the critical path for anything the harness grades.
6. **Local LLM wiring for the escalation ladder's "stronger model" rung and
   the planner/grounder generally**: `HttpLlm` already speaks the right
   protocol; this is really item 1/2's model-serving prerequisite (a
   running `llama-server.exe` or equivalent) rather than new Rust code.
7. **Sidecar split + Tauri hardening**: once (1)-(2) make the Rust backend
   do real, potentially-slow work, split `grace-backend` into its own
   process with the Job Object/KILL_ON_JOB_CLOSE, per-launch WebSocket
   token, restart/health-watch, single-instance guard, and CSP items from
   PLAN §10.2's "Shell (Tauri)" list.
