import { GraceEvent } from '../state/types';
import { getWsAuthToken } from '../shell';

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

  async function connect() {
    if (signal.aborted) return;

    cleanup();

    let targetUrl = wsUrl;
    try {
      const token = await getWsAuthToken();
      if (token) {
        // If wsUrl is relative or lacks protocol, URL handles ws:// and http://
        const separator = wsUrl.includes('?') ? '&' : '?';
        targetUrl = `${wsUrl}${separator}token=${encodeURIComponent(token)}`;
      }
    } catch {
      // ignore
    }

    if (signal.aborted) return;
    ws = new WebSocket(targetUrl);

    ws.onopen = () => {
      console.log('[Grace] WebSocket connected');
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const evt = JSON.parse(event.data);
        if (!evt || typeof evt !== 'object' || typeof evt.type !== 'string') {
          console.warn('[Grace] Dropping malformed WebSocket frame (R28):', evt);
          return;
        }
        dispatch(evt as GraceEvent);
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
