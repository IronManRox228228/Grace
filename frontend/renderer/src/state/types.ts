// Mirrors the backend contract in the spec (section "BACKEND CONTRACT").
// The frontend only ever consumes these — it never infers, executes,
// recognizes speech, or synthesizes it.

export type GraceEvent =
  | { type: 'Idle' }
  | { type: 'WakeWordDetected' }
  | { type: 'ListeningStarted' }
  | { type: 'PartialTranscript'; text: string }
  | { type: 'FinalTranscript'; text: string }
  | { type: 'ListeningStopped' }
  | { type: 'UnderstandingStarted'; label: string } // e.g. "Understanding request…"
  | { type: 'UnderstandingFinished' }
  | { type: 'ToolExecutionStarted'; label: string } // e.g. "Opening Microsoft Edge…"
  | { type: 'ToolExecutionFinished' }
  | { type: 'ResponseChunk'; text: string }
  | { type: 'SpeechStarted' }
  | { type: 'SpeechChunk' }
  | { type: 'SpeechFinished' }
  | { type: 'ConversationFinished' }
  | { type: 'Error'; message: string };

export type GraceState =
  | 'idle'
  | 'listening'
  | 'understanding'
  | 'executing'
  | 'speaking'
  | 'completed'
  | 'error';

export interface GraceSnapshot {
  state: GraceState;
  userTranscript: string; // live/partial + final user speech
  statusLabel: string; // "Understanding request…", "Opening Microsoft Edge…"
  responseText: string; // streamed assistant response
  errorMessage?: string;
}

export const INITIAL_SNAPSHOT: GraceSnapshot = {
  state: 'idle',
  userTranscript: '',
  statusLabel: '',
  responseText: '',
};
