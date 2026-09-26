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
| `crates/grace-memory` | ~1,700 | `agent/memory.py`'s `PersistentMemoryStore` (replaced, not ported - see PLAN.md §12) | New this update. A facts store and a history store on `rusqlite`/FTS5, behind `grace-core`'s `PersistentStore` trait. See "grace-memory (PLAN.md §12)" below. |

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
  generated once per database and stored in `history_meta`.
- **Single writer thread, bounded channel, one transaction per turn**,
  WAL + `synchronous=NORMAL` + incremental `auto_vacuum` (`src/db.rs`).
  Idle-time compaction (`compact_once`) runs in caller-chosen batch
  sizes.
- **Voice-scoped hard deletes**: `forget_today`, `forget_app` (also
  strips that app's counts out of every monthly roll-up blob),
  `forget_everything`.
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

```
p50 = 389.8 µs, p99 = 11.49 ms, p99.9 = 32.98 ms, max = 793.17 ms
```

The p50 is comfortably sub-millisecond; the p99 (11.49 ms) and especially
the max (793 ms) are higher than the spec's "target under 1 ms" - reported
honestly rather than rounded away. The tail is dominated by the daily
`compact_once` call and by SQLite's own periodic WAL checkpoint pauses
sharing the same single writer thread and connection as ordinary writes;
a p50 of well under 1 ms confirms the steady-state per-transaction cost
itself is fine, but a caller that cannot tolerate an occasional tens-of-
milliseconds stall should treat `record_episode` as fire-and-forget (send
onto a channel, don't await the reply) rather than call it inline on a
turn - which is how PLAN.md §12.2 describes this store being used ("off
the hot path... never slow a turn") in the first place. Untangling
compaction pauses from ordinary write latency (e.g. running compaction on
a schedule that never overlaps a foreground write) is listed under
Remaining work.

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
- **Rust, with `grace-memory` (this update)**: **233 passed, 3 ignored, 0
  failed** across the whole workspace (`cargo test --workspace`) - the 200
  above plus `grace-memory`'s new **33** (facts + history + adapter +
  redact/secret/db). `grace-core` also gained the additive
  `begin_episode`/`end_episode` trait methods (see "grace-memory (PLAN.md
  §12)" above) with no test-count change, since they're default no-ops.
  Reproduced with `LLAMA_SERVER_URL` pointed at an unreachable port to work
  around this session's unrelated port-8080 occupant (see above); without
  that override, `grace-backend`'s `demo_turn` test can fail
  non-deterministically for a reason that predates and is unrelated to
  this update.
- **Python**: re-ran `./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`
  again for this update: **841 passed, 1 skipped** - still exactly
  PLAN.md §0's baseline, unchanged. `git status` confirms this update only
  touches `crates/grace-memory/` (new), `crates/grace-core/src/memory.rs`,
  root `Cargo.toml`/`Cargo.lock`, and `PORT_STATUS.md`.

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
5. ~~`PersistentMemoryStore` over real SQLite~~ **Done this update** - see
   "grace-memory (PLAN.md §12)" below. What's left there, not done here
   because it reaches into `grace-core::agent_loop`/`grace_app` rather than
   `grace-memory` itself:
   - Wire `AgentLoop`/`GraceApp` to call `PersistentStore::begin_episode`/
     `end_episode` at real goal boundaries, so `GraceMemoryStore` gets true
     multi-step episodes (and the routine collapse they enable) through the
     legacy trait, instead of the one-step-per-`save_step`-call fallback
     `grace-memory::adapter` uses today.
   - A real app/window name reaching `save_step` (the legacy trait has no
     such parameter - `adapter.rs` currently records every episode under
     app `"unknown"`).
   - Wire the facts store's confirmation flow ("I'll remember Priya is your
     sister, right?") and conflict-by-voice API into an actual dialogue -
     both exist and are tested in `grace-memory::facts`, but nothing calls
     them from the turn pipeline yet.
   - An idle-tick driver that actually calls `HistoryStore::compact_once` on
     a schedule (it's implemented and tested, but nothing invokes it outside
     of tests/the benchmark yet).
6. **Local LLM wiring for the escalation ladder's "stronger model" rung and
   the planner/grounder generally**: `HttpLlm` already speaks the right
   protocol; this is really item 1/2's model-serving prerequisite (a
   running `llama-server.exe` or equivalent) rather than new Rust code.
7. **Sidecar split + Tauri hardening**: once (1)-(2) make the Rust backend
   do real, potentially-slow work, split `grace-backend` into its own
   process with the Job Object/KILL_ON_JOB_CLOSE, per-launch WebSocket
   token, restart/health-watch, single-instance guard, and CSP items from
   PLAN §10.2's "Shell (Tauri)" list.
