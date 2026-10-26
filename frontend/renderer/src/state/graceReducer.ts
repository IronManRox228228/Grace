import { ChatMessage, GraceEvent, GraceSnapshot, INITIAL_SNAPSHOT } from './types';

// Pure function: (snapshot, backend event) -> next snapshot.
// Keeping this pure and dependency-free makes it trivial to swap the
// transport (WebSocket today, Tauri IPC later) without touching a component.
//
// The only status text this ever produces is text the backend sent (the
// UnderstandingStarted / ToolExecutionStarted label). There are no hard-coded
// state labels such as "Listening…" or "Responding…".
//
// The conversation is a list of messages in order - user, assistant, user,
// assistant - and each reply is its own message: a new reply never gets glued
// onto the previous one.

function appendReplyChunk(snapshot: GraceSnapshot, text: string): GraceSnapshot {
  const last = snapshot.messages[snapshot.messages.length - 1];
  if (snapshot.replyOpen && last && last.role === 'assistant') {
    const updated: ChatMessage = { role: 'assistant', text: last.text + text };
    return {
      ...snapshot,
      state: 'speaking',
      responseText: updated.text,
      messages: [...snapshot.messages.slice(0, -1), updated],
    };
  }
  return {
    ...snapshot,
    state: 'speaking',
    responseText: text,
    replyOpen: true,
    messages: [...snapshot.messages, { role: 'assistant', text }],
  };
}

export function graceReducer(snapshot: GraceSnapshot, event: GraceEvent): GraceSnapshot {
  switch (event.type) {
    case 'Idle':
      return INITIAL_SNAPSHOT;

    case 'WakeWordDetected':
      return { ...INITIAL_SNAPSHOT, state: 'listening' };

    // The mic opens for the user's turn or for the answer to a question Grace
    // just asked. The conversation so far stays on screen either way.
    case 'ListeningStarted':
      return { ...snapshot, state: 'listening', statusLabel: '', replyOpen: false };

    case 'FinalTranscript':
      return {
        ...snapshot,
        userTranscript: event.text,
        responseText: '',
        replyOpen: false,
        messages: [...snapshot.messages, { role: 'user', text: event.text }],
      };

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
      return appendReplyChunk(snapshot, event.text);

    // A new spoken reply starts: its chunks form a new message.
    case 'SpeechStarted':
      return { ...snapshot, state: 'speaking', replyOpen: false };

    case 'SpeechChunk':
      return snapshot;

    case 'SpeechFinished':
      return { ...snapshot, replyOpen: false };

    case 'ConversationFinished':
      return { ...snapshot, state: 'completed', statusLabel: '', replyOpen: false };

    // Diagnostic only - carries no user-visible state.
    case 'TurnTrace':
      return snapshot;

    case 'Error':
      return { ...snapshot, state: 'error', errorMessage: event.message };

    default: {
      // The frozen contract still declares FollowupListeningStarted, but the
      // backend no longer emits it and the UI has nothing to show for it.
      // Every other variant must be handled above: adding one to the schema
      // is a compile error here until it is.
      const unhandled: { type: 'FollowupListeningStarted' } = event;
      void unhandled;
      return snapshot;
    }
  }
}
