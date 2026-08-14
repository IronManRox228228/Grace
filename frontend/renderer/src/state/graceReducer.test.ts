import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { graceReducer } from './graceReducer';
import { INITIAL_SNAPSHOT, type GraceEvent, type GraceSnapshot } from './types';

// The renderer has never been executed by any test. `frontend/package.json`
// had no test runner at all, so the reducer - the single place where a backend
// event becomes something the user sees - was verified only by someone looking
// at the overlay.
//
// That matters most for the migration. `contract/README.md` states that a
// recorded event stream is replayed through this reducer and the resulting
// snapshot sequence asserted. Nothing performed that check, so an event shape
// the Rust port changes could break the UI with every Python-side test green.
//
// This is that check: real corpus tapes, through the real reducer.

const HERE = dirname(fileURLToPath(import.meta.url));
const CORPUS = join(HERE, '..', '..', '..', '..', 'corpus');

function tapes(): string[] {
  if (!existsSync(CORPUS)) return [];
  return readdirSync(CORPUS, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .filter((entry) => existsSync(join(CORPUS, entry.name, 'events.jsonl')))
    .map((entry) => entry.name);
}

function eventsOf(tape: string): GraceEvent[] {
  const raw = readFileSync(join(CORPUS, tape, 'events.jsonl'), 'utf-8');
  return raw
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line))
    // The tape wraps each event with its own metadata; the payload is the event.
    .map((record) => (record.event ?? record) as GraceEvent);
}

function replay(events: GraceEvent[]): GraceSnapshot[] {
  const snapshots: GraceSnapshot[] = [];
  let snapshot = INITIAL_SNAPSHOT;
  for (const event of events) {
    snapshot = graceReducer(snapshot, event);
    snapshots.push(snapshot);
  }
  return snapshots;
}

describe('the corpus replays through the reducer', () => {
  const names = tapes();

  it('finds the corpus', () => {
    // Without this the suite below would pass by iterating over nothing, which
    // is the same failure mode as having no tests at all.
    expect(names.length).toBeGreaterThan(0);
  });

  it.each(names)('%s produces a valid snapshot at every step', (tape) => {
    const events = eventsOf(tape);
    expect(events.length).toBeGreaterThan(0);

    const states = new Set([
      'idle', 'listening', 'understanding', 'executing',
      'speaking', 'completed', 'error',
    ]);

    for (const snapshot of replay(events)) {
      expect(states.has(snapshot.state)).toBe(true);
      expect(typeof snapshot.userTranscript).toBe('string');
      expect(typeof snapshot.statusLabel).toBe('string');
      expect(typeof snapshot.responseText).toBe('string');
    }
  });

  it.each(names)('%s never reaches the exhaustiveness fallback', (tape) => {
    // The `default` branch has a `never` assertion at compile time, but a
    // backend emitting an event this union does not declare is a *runtime*
    // event, and it lands there silently. That is precisely the drift Phase 0
    // found - FollowupListeningStarted and TurnTrace went two releases
    // unhandled.
    const declared = new Set([
      'Idle', 'WakeWordDetected', 'ListeningStarted', 'FollowupListeningStarted',
      'FinalTranscript', 'ListeningStopped', 'UnderstandingStarted',
      'UnderstandingFinished', 'ToolExecutionStarted', 'ToolExecutionFinished',
      'ResponseChunk', 'SpeechStarted', 'SpeechChunk', 'SpeechFinished',
      'ConversationFinished', 'TurnTrace', 'Error',
    ]);
    const unknown = eventsOf(tape)
      .map((event) => event.type)
      .filter((type) => !declared.has(type));
    expect(unknown).toEqual([]);
  });
});

describe('the reducer itself', () => {
  it('accumulates response text across chunks', () => {
    const events: GraceEvent[] = [
      { type: 'ResponseChunk', text: 'Opening ' },
      { type: 'ResponseChunk', text: 'Notepad.' },
    ];
    expect(replay(events).at(-1)?.responseText).toBe('Opening Notepad.');
  });

  it('clears the previous turn when listening starts', () => {
    // A turn that showed the last turn's answer while listening to the next
    // one is how a voice-only user loses track of what Grace is doing.
    const after = replay([
      { type: 'ResponseChunk', text: 'the old answer' },
      { type: 'ListeningStarted' },
    ]).at(-1);
    expect(after?.responseText).toBe('');
    expect(after?.userTranscript).toBe('');
  });

  it('returns to the initial snapshot on Idle', () => {
    const after = replay([
      { type: 'FinalTranscript', text: 'open notepad' },
      { type: 'Idle' },
    ]).at(-1);
    expect(after).toEqual(INITIAL_SNAPSHOT);
  });

  it('does not mutate the snapshot it is given', () => {
    const before: GraceSnapshot = { ...INITIAL_SNAPSHOT, responseText: 'kept' };
    graceReducer(before, { type: 'ResponseChunk', text: ' and more' });
    expect(before.responseText).toBe('kept');
  });
});
