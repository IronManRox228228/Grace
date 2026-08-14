# The Grace contract and the migration harness

Everything here exists to make "no regressions" a claim a machine can check,
before any Rust is written. It is Phase 0 of the Rust + Tauri 2 migration.

## The contract

`grace-events.schema.json` is the single frozen definition of the backend →
frontend event stream. It is the one seam between the backend (Python today,
Rust after the migration) and the React renderer.

It is enforced in four places:

| Where | What | When |
|---|---|---|
| `codegen_types.py` | generates `frontend/renderer/src/state/types.ts` | `--check` in CI |
| `graceReducer.ts` | `never` assertion in the default branch | TypeScript build |
| `src/grace/harness/contract.py` | validates every emitted event | every `emit()` |
| `tests/test_contract.py` | every emitted shape, and no dead variants | `pytest` |

**The contract is additive-only**, and only for genuinely new UI features. If a
port cannot reproduce an existing event at the right moment, that is a bug in
the port, not a reason to edit the schema.

Regenerate `types.ts` after any schema change:

```
python contract/codegen_types.py
```

### What was fixed when it was frozen

The hand-written TypeScript union had drifted from what the backend emitted:

- `FollowupListeningStarted` and `TurnTrace` were emitted but not declared, so
  the renderer silently dropped them through its default branch.
- `PartialTranscript` was declared and handled but never emitted. Grace has no
  streaming transcript - Whisper runs once on the whole utterance after the VAD
  closes the turn.
- `ToolExecutionStarted` carried `{label}` from the dispatcher and
  `{label, tool, step}` from the agent loop.

`Error` is the one variant in the schema that Python never emits. It is kept
deliberately: the renderer has always handled it, and the Rust port needs it
for sidecar-unavailable and supervisor-failure paths.

## Recording a corpus

The tape corpus is the regression oracle. It has to be recorded against the
known-good Python implementation, on the real machine, before that
implementation starts being replaced.

```
set GRACE_RECORD_DIR=%CD%\recordings
set GRACE_CONTRACT_STRICT=1
python src\grace\main.py
```

Each run writes one session directory:

```
recordings/<timestamp>/
  meta.json        config snapshot (API keys redacted) + git sha
  events.jsonl     every emitted event, with offsets from session start
  llm.jsonl        every LLM exchange - full prompt in, ordered chunks out
  stt.jsonl        every transcription
  pcm/<digest>.raw the raw int16 PCM behind each transcription
  audio.jsonl      the mic chunk sequence, with arrival offsets
  audio.raw        the chunk payloads
  snapshots.jsonl  every perception snapshot (the ordered element graph)
  dispatch.jsonl   every tool dispatch - intent in, result out
```

`GRACE_RECORD_AUDIO=0` skips the continuous mic capture (~32 KB/s) when a tape
is only wanted for its LLM and dispatch behaviour. Leave it on for anything
that should be able to grade the VAD or the wake word: that audio stops being
recordable the moment the Python audio stack is deleted.

## Generating tapes instead of recording them

The plan asked for "~30 recorded sessions". That was the wrong unit. What a
tape pins down is a *route* - which tools were asked for, in what order, with
what parameters, and which events the frontend saw - and everything that
decides a route is deterministic given a transcript and a set of model replies.
Both can be scripted, so most of the corpus does not need a voice, a screen, or
an evening.

```
python -m grace.harness.generate --out corpus/     # write every scenario
python -m grace.harness.generate --list            # what each one covers
python -m grace.harness.generate --out corpus/ --only safety_confirm_accepted
```

Scenarios live in `src/grace/harness/generate.py`. Each drives a real
`GraceApp` through `_handle_activation` - the same entry point a wake word uses
- with only the external edges scripted, at the same seams `replay.py` uses:
the model at `_stream_gemini_response`, Whisper's model object, each
`Dispatcher._<tool>` leaf, `ComputerUse.perform`, and a paced microphone. The
turn state machine, router, intent parser, agent loop, safety guard, dispatcher
and response generator are all the production ones.

The catalogue covers: plain conversation; every fast-path tool; a failing tool;
agentic goals including a failed step, a rejected completion, an unparseable
plan, and the `open_app` pre-execution path; a safety confirmation accepted,
declined, and answered with an unrelated request; the `cua_press_key` parameter
guard; a rate-limited intent call and a rate-limited planner; an empty
transcript; an unparseable intent; and two follow-up chains.

Generated tapes are stamped `"synthetic": true` in `meta.json`.

### What still has to be recorded by hand

Four things a script cannot stand in for. This is the corpus that needs the
real machine, and it is much smaller than thirty sessions:

- **Real element graphs.** Generated snapshots are hand-written, so they grade
  the planner's *use* of a graph and never the fidelity of the graph itself.
  Phase 4's parity gate needs snapshots walked from real windows - five or six
  goals across a few different applications, recorded by using Grace normally
  with `GRACE_RECORD_DIR` set.
- **Real speech.** The generator's audio is a square wave. Wake-word behaviour
  and transcription in a real room need real utterances, including some with
  background noise. Ten minutes of talking, not thirty sessions.
- **Grounding.** A taped snapshot stores a digest of the screenshot, not the
  pixels, so a step that needs grounding silently skips the grounder. Grounding
  parity needs its own screenshot corpus.
- **Barge-in.** With TTS stubbed, the `tts_player.stop()` a follow-up performs
  produces no event to compare against.

Recording more sessions buys almost nothing at the LLM layer: the cloud model
returns different text for the same prompt each time, and replay simply pins
whatever was recorded.

Commit the JSONL streams. `pcm/` is gitignored and lives alongside the repo.

## Replaying

```
python -m grace.harness.replay --tape recordings/20260812-140301
python -m grace.harness.replay --corpus corpus/ --json report.json
```

Replay re-runs the turn with every external edge pinned to the recording - LLM
responses, transcripts, perception snapshots, tool results, and the mic chunk
sequence - and diffs the result. It exits non-zero on any mismatch.

What is compared, and how strictly:

| Stream | Comparison | Why |
|---|---|---|
| events | byte-exact, in order | this is the contract and the UI state machine |
| dispatch | byte-exact, in order | catches "right outcome, different route" |
| prompts | byte-exact | a stray newline silently changes what the model does |
| transcripts | ≥98% exact match | two speech models will not agree character-for-character |
| timings | p95 + 20% budget | catches a port that serialised something pipelined |

`TurnTrace` is excluded from the event diff: it carries wall-clock durations
that legitimately differ on every run.

Replay reads the config back out of the tape's `meta.json`. A tape recorded
without one is replayed against the ambient environment - so the VAD silence
window and the follow-up timeout are not actually pinned - and says so in its
notes.

A note prefixed `INCONCLUSIVE (VAD)` is not a regression. It means the replay
could not deliver the recorded chunk sequence on schedule, usually CPU
contention, so turn-end timing was never reproduced faithfully enough to grade.
Re-run it on an idle machine.

### Two things replay cannot decide

- **Audio quality.** Kokoro is stubbed out; synthesis is graded only by the
  `SpeechStarted`/`SpeechChunk`/`SpeechFinished` sequence. Prosody is a
  listening test.
- **Grounding.** A taped snapshot stores a digest of the screenshot, not the
  pixels, so a replay can grade the planner (which sees the element graph) but
  not the grounder (which sees the image). Grounding parity is a Phase 4
  concern with its own screenshot corpus.

### A constraint the harness surfaced

`VadDetector` accumulates silence against `time.time()`, not against a sample
count (`src/grace/vad/detector.py:98`). Turn-end therefore depends on how fast
chunks actually arrived, which is why `TapePump` replays them at their recorded
offsets and why a replay takes about as long as the window it reproduces.

This is a live constraint on the Rust port: a wall-clock VAD has to stay
wall-clock, or turn-end timing shifts for every user.

### Two findings the harness turned up in the current backend

Neither is a migration bug. Both are recorded here because the corpus pins them
as current behaviour, and the port has to reproduce them or change them
deliberately.

**The agent loop can spin without bound.** `GemmaClient.generate_text` catches
every exception except `RateLimitError` and returns `None`
(`llm/gemma_client.py:413`). The planner reads that as an empty response, the
agent loop books it as an unparseable plan and retries - and
`AGENT_MAX_ITERATIONS` defaults to `0`, meaning unlimited. So a persistently
failing planner (a bad key, a network partition) retries forever, burning quota
or GPU, with no way for a user who cannot reach a keyboard to stop it. This is
risk #8 in the migration plan, reachable with no porting bug at all. Both the
generator and the replay driver cap the loop for this reason.

**The fast path does not consult `SafetyGuard`.** `close_app`, `lock_computer`
and `delete_file` are all in `CapabilityRouter.FAST_PATH_TOOLS`, and a
single-tool intent goes straight to the dispatcher. Only the agentic path asks
for confirmation. `fastpath_delete_file` and `safety_confirm_accepted` tape both
routes side by side.
