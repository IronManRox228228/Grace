// GENERATED FILE - DO NOT EDIT.
// Source: contract/grace-events.schema.json
// Regenerate: python contract/codegen_types.py
//
// The backend (Python today, Rust after the migration) and this file are two
// views of one frozen contract. Adding a variant here without adding it to the
// schema will be reverted by the next codegen run; CI runs --check.

export type GraceEvent =
  | { type: 'Idle' } // Turn is over; the overlay resets to its resting state.
  | { type: 'WakeWordDetected' } // Vosk matched the wake word, or the UI/global-hotkey forced a wake.
  | { type: 'ListeningStarted' } // Mic is open for the primary utterance of a turn.
  | { type: 'FollowupListeningStarted'; timeout: number } // The post-response follow-up window has opened.
  | { type: 'FinalTranscript'; text: string } // Whisper's transcript for the utterance just captured.
  | { type: 'ListeningStopped' } // Mic closed for this utterance.
  | { type: 'UnderstandingStarted'; label: string } // Intent inference began.
  | { type: 'UnderstandingFinished' } // Intent inference ended (success or failure).
  | { type: 'ToolExecutionStarted'; label: string; tool?: string; step?: number } // A tool is about to run.
  | { type: 'ToolExecutionFinished'; tool?: string; status?: string } // The tool finished.
  | { type: 'ResponseChunk'; text: string } // One sentence-shaped slice of the assistant's response, split off the LLM token stream as it arr…
  | { type: 'SpeechStarted' } // TTS playback began for this response.
  | { type: 'SpeechChunk' } // One synthesized sentence was handed to the player.
  | { type: 'SpeechFinished' } // TTS playback drained.
  | { type: 'ConversationFinished' } // The turn's work is done; the follow-up window may still be open.
  | { type: 'TurnTrace'; trace: { label: string, total_ms: number, stages: { name: string, depth: number, ms: number | null, detail: string | null }[], events: Record<string, number> } } // Per-turn latency instrumentation.
  | { type: 'Error'; message: string }; // A turn failed in a way the user must be told about.

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
