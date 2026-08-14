import { GraceEvent, GraceSnapshot, INITIAL_SNAPSHOT } from './types';

// Pure function: (snapshot, backend event) -> next snapshot.
// Keeping this pure and dependency-free makes it trivial to swap the
// transport (WebSocket today, Tauri IPC later) without touching a component.
//
// The switch is exhaustive over GraceEvent: the `never` assertion in the
// default branch makes adding a variant to contract/grace-events.schema.json
// a compile error here until it is handled. That is deliberate - the previous
// silent `return snapshot` default is how FollowupListeningStarted and
// TurnTrace went two releases without anyone noticing the union was wrong.
export function graceReducer(snapshot: GraceSnapshot, event: GraceEvent): GraceSnapshot {
  switch (event.type) {
    case 'Idle':
      return INITIAL_SNAPSHOT;

    case 'WakeWordDetected':
      return { ...INITIAL_SNAPSHOT, state: 'listening', statusLabel: 'Listening…' };

    case 'ListeningStarted':
      return { ...snapshot, state: 'listening', statusLabel: 'Listening…', userTranscript: '', responseText: '' };

    // No-op, matching the behavior this event has always had: it was absent
    // from the union and fell through the old silent default. Showing a
    // "Listening…" affordance for the follow-up window is very likely the
    // right UX, but changing it here would make every Phase 0 golden tape
    // disagree with the recorded baseline for a reason unrelated to the port.
    // Tracked as post-migration work.
    case 'FollowupListeningStarted':
      return snapshot;

    case 'FinalTranscript':
      return { ...snapshot, userTranscript: event.text };

    case 'ListeningStopped':
      return snapshot;

    case 'UnderstandingStarted':
      return { ...snapshot, state: 'understanding', statusLabel: event.label };

    case 'UnderstandingFinished':
      return snapshot;

    case 'ToolExecutionStarted':
      return { ...snapshot, state: 'executing', statusLabel: event.label };

    case 'ToolExecutionFinished':
      return snapshot;

    case 'ResponseChunk':
      return {
        ...snapshot,
        state: 'speaking',
        responseText: snapshot.responseText + event.text,
      };

    case 'SpeechStarted':
      return { ...snapshot, state: 'speaking', statusLabel: 'Responding…' };

    case 'SpeechChunk':
      return snapshot;

    case 'SpeechFinished':
      return snapshot;

    case 'ConversationFinished':
      return { ...snapshot, state: 'completed', statusLabel: '' };

    // Diagnostic only - carries no user-visible state.
    case 'TurnTrace':
      return snapshot;

    case 'Error':
      return { ...snapshot, state: 'error', errorMessage: event.message };

    default: {
      const unhandled: never = event;
      void unhandled;
      return snapshot;
    }
  }
}
