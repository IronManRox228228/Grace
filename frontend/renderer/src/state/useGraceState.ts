import { useCallback, useEffect, useReducer, useRef } from 'react';
import { graceReducer } from './graceReducer';
import { INITIAL_SNAPSHOT } from './types';
import { createWsClient } from '../backend/wsClient';
import { onWakeRequested } from '../shell';

export function useGraceState() {
  const [snapshot, dispatch] = useReducer(graceReducer, INITIAL_SNAPSHOT);
  const wsClientRef = useRef<ReturnType<typeof createWsClient> | null>(null);

  // Connect to the backend WebSocket once on mount.
  useEffect(() => {
    const controller = new AbortController();
    wsClientRef.current = createWsClient(dispatch, controller.signal);
    return () => {
      controller.abort();
      wsClientRef.current = null;
    };
  }, []);

  // Ctrl+Alt+G (registered by the shell) or clicking the idle pill sends
  // {"type":"wake"} to the backend over the WebSocket, which triggers the same
  // activation pipeline as the real wake word.
  const wake = useCallback(() => {
    if (snapshot.state !== 'idle' && snapshot.state !== 'completed') return;
    wsClientRef.current?.sendWake();
  }, [snapshot.state]);

  // Held in a ref so the subscription itself is created once. Re-subscribing
  // whenever `wake` changes identity would mean tearing down and rebuilding an
  // asynchronously-registered listener on every state transition, and a
  // shortcut pressed in that gap would be dropped.
  const wakeRef = useRef(wake);
  wakeRef.current = wake;

  useEffect(() => onWakeRequested(() => wakeRef.current()), []);

  return { snapshot, wake };
}
