import { GraceEvent, GraceSnapshot, INITIAL_SNAPSHOT } from './types';

// Pure function: (snapshot, backend event) -> next snapshot.
// Keeping this pure and dependency-free makes it trivial to swap the
// mock event source for a real backend transport later without
// touching any component.
export function graceReducer(snapshot: GraceSnapshot, event: GraceEvent): GraceSnapshot {
  switch (event.type) {
    case 'Idle':
      return INITIAL_SNAPSHOT;

    case 'WakeWordDetected':
      return { ...INITIAL_SNAPSHOT, state: 'listening', statusLabel: 'Listening…' };

    case 'ListeningStarted':
      return { ...snapshot, state: 'listening', statusLabel: 'Listening…', userTranscript: '', responseText: '' };

    case 'PartialTranscript':
      return { ...snapshot, state: 'listening', userTranscript: event.text };

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

    case 'Error':
      return { ...snapshot, state: 'error', errorMessage: event.message };

    default:
      return snapshot;
  }
}
