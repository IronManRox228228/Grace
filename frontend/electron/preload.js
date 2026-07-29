const { contextBridge, ipcRenderer } = require('electron');

// This is the ONLY surface the renderer talks to the OS/backend through.
// When the real backend exists, replace the `onSimulateWake` listener
// with whatever transport it uses (IPC, WebSocket, named pipe, etc.)
// and forward its events through this same bridge — the renderer's
// state machine already expects the event shapes documented below.
contextBridge.exposeInMainWorld('grace', {
  // Dev-only: triggered by the Ctrl+Alt+G shortcut in main.js to
  // simulate a WakeWordDetected event from the backend.
  onSimulateWake: (callback) => {
    ipcRenderer.on('grace:simulate-wake', callback);
    return () => ipcRenderer.removeListener('grace:simulate-wake', callback);
  },

  // Lets the pill stay click-through when idle/collapsed, and
  // interactive once it expands, so it never blocks the desktop.
  setIgnoreMouseEvents: (ignore, options) => {
    ipcRenderer.send('grace:set-ignore-mouse-events', ignore, options);
  },
});
