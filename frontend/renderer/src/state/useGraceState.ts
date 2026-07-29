import { useCallback, useEffect, useReducer, useRef } from 'react';
import { graceReducer } from './graceReducer';
import { INITIAL_SNAPSHOT } from './types';
import { createWsClient } from '../backend/wsClient';

declare global {
  interface Window {
    grace?: {
      onSimulateWake: (callback: () => void) => () => void;
      setIgnoreMouseEvents: (ignore: boolean, options?: { forward?: boolean }) => void;
    };
  }
}

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

  // Ctrl+Alt+G (registered in electron/main.js) or clicking the idle
  // pill sends a {"type":"wake"} message to the backend over WebSocket,
  // which triggers the same activation pipeline as the real wake word.
  const wake = useCallback(() => {
    if (snapshot.state !== 'idle' && snapshot.state !== 'completed') return;
    wsClientRef.current?.sendWake();
  }, [snapshot.state]);

  useEffect(() => {
    const unsubscribe = window.grace?.onSimulateWake(() => wake());
    return () => unsubscribe?.();
  }, [wake]);

  const enterInteractive = useCallback(() => {
    window.grace?.setIgnoreMouseEvents(false);
  }, []);
  const leaveInteractive = useCallback(() => {
    window.grace?.setIgnoreMouseEvents(true, { forward: true });
  }, []);

  return { snapshot, wake, enterInteractive, leaveInteractive };
}
