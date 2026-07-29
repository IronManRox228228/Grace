import { GraceEvent } from '../state/types';

// Default backend WebSocket URL.
const DEFAULT_WS_URL = 'ws://127.0.0.1:8765';

/**
 * Persistent WebSocket client that connects to the Grace Python backend.
 *
 * - Receives GraceEvent JSON messages and dispatches them to the React state.
 * - Provides sendWake() for the UI to trigger a backend interaction.
 * - Auto-reconnects on disconnect unless the abort signal fires.
 * - Only acts as a transport — never modifies application state directly.
 */
export function createWsClient(
  dispatch: (event: GraceEvent) => void,
  signal: AbortSignal,
  wsUrl: string = DEFAULT_WS_URL,
) {
  let ws: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  function cleanup() {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (ws) {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onclose = null;
      ws.onerror = null;
      ws.close();
      ws = null;
    }
  }

  function connect() {
    if (signal.aborted) return;

    cleanup();
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[Grace] WebSocket connected');
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const evt = JSON.parse(event.data) as GraceEvent;
        dispatch(evt);
      } catch (e) {
        console.error('[Grace] Invalid WebSocket message', e);
      }
    };

    ws.onclose = () => {
      ws = null;
      if (!signal.aborted) {
        reconnectTimer = setTimeout(connect, 2000);
      }
    };

    ws.onerror = () => {
      // onclose fires right after onerror
    };
  }

  function sendWake() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'wake' }));
    }
  }

  signal.addEventListener('abort', cleanup);

  connect();

  return { sendWake };
}
