# Grace — the overlay

A translucent pill, anchored bottom-centre of the primary display, that expands
into a live conversation and collapses back to idle. It is presentation only:
no AI, no speech recognition, no synthesis. Every component renders a
`GraceSnapshot` built by a pure reducer from the backend's event stream.

## How it fits together

```
Rust shell (src-tauri/)  ──spawns──▶  Python backend (src/grace/)
       │                                        │
       │ Tauri IPC: click-through, wake shortcut │ WebSocket 127.0.0.1:8765
       ▼                                        ▼
                    React renderer (renderer/)
```

Two channels, and the split matters. Application state — everything the pill
displays — arrives **only** over the WebSocket, which is why the backend can be
rewritten in Rust without the renderer changing. Tauri IPC carries the two
things a webview cannot do for itself: toggling click-through, and hearing the
global wake shortcut. Both live in `renderer/src/shell/`, the successor to
Electron's `preload.js` and the only module that knows a shell exists.

## Run it locally (Windows)

```bash
cd frontend
npm install
npm run dev        # vite + the Tauri shell; also starts the Python backend
```

`npm run dev` is `tauri dev`, which starts Vite itself via `beforeDevCommand` —
there is no separate terminal to keep open. It runs from the repo root (hence
the `cd ..` in the script): the Tauri CLI locates an app by looking for
`src-tauri/` beside the working directory and does not search upwards, and
`src-tauri/` is a sibling of `frontend/`, not a child of it. The shell then spawns
`venv/Scripts/python.exe src/grace/main.py`, exactly as `electron/main.js` used
to. If that interpreter is missing the overlay still runs, so the UI can be
worked on against a backend started by hand.

Requires the **MSVC toolchain** (Visual Studio Build Tools with "Desktop
development with C++") and the **WebView2 runtime**, which ships with Windows
11.

## Production build

```bash
npm run build      # bundles the renderer, compiles the shell, writes an NSIS installer
```

Output lands in `target/release/bundle/nsis/`.

## Try it

Press **Ctrl+Alt+G**, or click the idle pill. Both send `{"type":"wake"}` over
the WebSocket, so the backend sees one activation path rather than two.

## Project layout

```
src-tauri/          # the shell (Rust) — sibling of frontend/, at the repo root
  src/lib.rs        # window setup, wake shortcut, the click-through command
  src/window.rs     # Win32: focus suppression, click-through, placement
  src/backend.rs    # Python child process supervision
renderer/
  src/
    shell/          # the ONLY bridge between renderer and OS
    backend/
      wsClient.ts   # the backend transport: reconnecting WebSocket client
    state/
      types.ts          # GraceEvent / GraceSnapshot — generated from the contract
      graceReducer.ts   # pure event → snapshot reducer
      useGraceState.ts  # wires reducer + WebSocket + shell
    components/
      GracePill.tsx         # idle glyph ↔ expanded card, one spring animation
      ListeningIndicator.tsx
      TranscriptPanel.tsx   # live user transcript
      StatusCard.tsx        # "Understanding request…" / "Opening Edge…"
      ActionIndicator.tsx   # subtle tool-execution motion
      ResponseRenderer.tsx  # streamed assistant response
```

`types.ts` is **generated**. It is the TypeScript projection of
`contract/grace-events.schema.json`; edit the schema and re-run
`python contract/codegen_types.py`. See `contract/README.md` for why the event
union is frozen.

## Why the overlay behaves the way it does

Three window properties are load-bearing rather than cosmetic, and they are
implemented natively in `src-tauri/src/window.rs` because Tauri does not model
them:

- **It never takes focus** (`WS_EX_NOACTIVATE`). Grace narrates automation
  while it drives other windows. A pill that activated on click would steal
  focus from the application being driven, and the click it just dispatched
  would land somewhere else.
- **It is click-through except on the pill** (`WS_EX_TRANSPARENT`), so it never
  blocks the desktop it floats over.
- **It stays on top** (`WS_EX_TOPMOST`), because narration the user cannot see
  is no narration at all.

The first and third are declared in `tauri.conf.json` (`focusable: false`,
`alwaysOnTop`) rather than applied by hand, because tao recomputes the window's
entire extended style from its own flag model on every change — styles set
behind its back survive only until the next one and then vanish mid-session.

Click-through is the interesting one. Electron used
`setIgnoreMouseEvents(true, { forward: true })`, where `forward` kept delivering
mouse-*move* messages to the page so it could still notice the pointer arriving
on the pill. `WS_EX_TRANSPARENT` has no equivalent half-measure: it removes the
window from hit-testing entirely, so a click-through webview receives no mouse
messages at all and `onMouseEnter` can never fire. Ported literally that leaves
the pill permanently untouchable, which kills click-to-wake. So the shell
watches the cursor instead: the renderer reports the pill's rectangle
(`shell/useHoverRegion`) and `src-tauri/src/hover.rs` drops click-through only
while the pointer is inside it.

## Known follow-ups

Two things carried over from the Electron shell unchanged, so that the Tauri
swap could be judged on its own:

- **No CSP.** Electron ran without one, and the renderer's `index.html` pulls
  Inter from Google Fonts, so any policy tight enough to be worth having would
  have changed what loads. Left at Tauri's permissive default for now.
- **Inter is fetched at runtime.** For an assistant that advertises being
  offline-first, the font should be vendored into `renderer/public/fonts/`.
  Today a cold start with no network silently falls back to a system face.

## Design tokens

| Role | Value |
|---|---|
| Background | Cream White `#FDFBF7` |
| Primary accent | Periwinkle `#CCCCFF` |
| Secondary accent | Dusty Plum `#705553` |
| Text / anchor | Dark Olive `#3B4430` |
| Optional success | Soft Sage `#A8B89A` |

**Typography:** Inter covers every in-product role. TAN Pearl is a licensed
display face reserved for the wordmark, and is not bundled. To use it, drop the
licensed files in `renderer/public/fonts/` and uncomment the `@font-face` block
in `renderer/src/styles/globals.css`.
