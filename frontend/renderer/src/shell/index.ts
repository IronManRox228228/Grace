/**
 * The renderer's entire surface onto the desktop shell.
 *
 * This is the successor to `electron/preload.js` and it exists for the same
 * reason: everything the UI cannot do from inside a webview goes through one
 * named module, so the shell can be replaced without auditing components.
 *
 * It is deliberately *not* the backend transport. Grace's events arrive over
 * the WebSocket in `backend/wsClient.ts`; the two calls below are window
 * management and nothing else.
 *
 * Outside the shell - `vite dev` in a plain browser, or a component test -
 * both calls degrade to no-ops rather than throwing, so the UI can be worked
 * on without a Rust build.
 */
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

/** Must match `WAKE_EVENT` in `src-tauri/src/lib.rs`. */
const WAKE_EVENT = 'grace://wake';

const inShell =
  typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/**
 * Fires when the user presses the global wake shortcut (Ctrl+Alt+G).
 *
 * The shell only reports the keypress; waking is still done by sending
 * `{"type":"wake"}` over the WebSocket, so a shortcut and a click on the pill
 * reach the backend by the identical path.
 */
export function onWakeRequested(callback: () => void): () => void {
  if (!inShell) return () => {};

  // Reported rather than swallowed. A rejected `listen` leaves the wake
  // shortcut permanently dead, and without this it fails as an unhandled
  // promise rejection nobody sees.
  const pending = listen(WAKE_EVENT, () => callback()).catch((err: unknown) => {
    console.error('[Grace] could not subscribe to the wake event', err);
    return null;
  });

  return () => {
    void pending.then((unlisten) => unlisten?.());
  };
}

/**
 * Declares the only part of the overlay that should respond to the pointer.
 *
 * The overlay window is far larger than what it draws, and everything outside
 * the region reported here stays click-through so the desktop underneath keeps
 * working. Pass `null` to make the whole overlay click-through again.
 *
 * The shell watches the cursor against this rectangle rather than the renderer
 * watching for `mouseenter`, because a click-through window receives no mouse
 * messages at all - see `src-tauri/src/hover.rs`. Coordinates are CSS pixels
 * relative to the top-left of the viewport, exactly as `getBoundingClientRect`
 * reports them.
 */
export function reportHoverRegion(region: DOMRect | null): void {
  if (!inShell) return;

  const payload = region
    ? { x: region.x, y: region.y, width: region.width, height: region.height }
    : null;

  void invoke('set_hover_region', { region: payload }).catch((err: unknown) => {
    // Worth a line in the console: a failure here leaves the pill permanently
    // untouchable, or the overlay permanently swallowing clicks on the desktop.
    console.error('[Grace] could not report the hover region', err);
  });
}
