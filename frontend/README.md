# Grace — Frontend Prototype

This is a working Electron + React frontend for **Grace**, built strictly to the
frontend spec: a translucent overlay pill, anchored bottom-center, that expands
into a live conversation and collapses back to idle. It implements the state
machine, motion design, and visual language from the spec exactly — it does
**not** contain any AI, speech recognition, or speech synthesis, per the
"frontend is presentation only" rule in the spec.

## What's real vs. mocked

- **Real:** the Electron overlay window, the full `Idle → Listening →
  Understanding → Executing → Speaking → Completed → Idle` state machine, the
  streaming transcript, the motion/animation system, and the visual language
  (colors, typography roles, glassmorphism).
- **Mocked:** there is no backend yet, so `renderer/src/mock/mockBackend.ts`
  simulates one full interaction (wake → listening → understanding → tool
  execution → streamed response) with realistic timings, and emits the exact
  event shapes documented in `renderer/src/state/types.ts`. When the real
  backend (Gemma 4 E4B / Whisper / Kokoro pipeline) exists, replace that one
  file with a client for whatever transport the backend uses (WebSocket, named
  pipe, IPC) — nothing else in the frontend needs to change, since every
  component only ever consumes a `GraceSnapshot` built by the reducer.

## Try it

Press **Ctrl+Alt+G** (registered as a global shortcut) or click the idle pill
to simulate "Hey Grace" and watch the full interaction play out.

## Run it locally (Windows, Node 18+)

```bash
npm install
npm run dev
```

This starts the Vite dev server and launches Electron pointed at it. The
overlay window appears bottom-center on your primary display.

## Production build

```bash
npm run build   # bundles the renderer to renderer/dist
npm start       # launches Electron against the built renderer
```

To produce an installable `.exe` (via electron-builder):

```bash
npm run dist
```

## Project layout

```
electron/
  main.js       # transparent, frameless, always-on-top overlay window
  preload.js    # the ONLY bridge between renderer and OS/backend
renderer/
  src/
    state/
      types.ts          # GraceEvent / GraceSnapshot — the backend contract
      graceReducer.ts   # pure event → snapshot reducer
      useGraceState.ts  # wires reducer + mock backend + electron IPC
    mock/
      mockBackend.ts    # ← replace this with the real backend client
    components/
      GracePill.tsx         # idle glyph ↔ expanded card, one spring animation
      ListeningIndicator.tsx
      TranscriptPanel.tsx   # live user transcript
      StatusCard.tsx        # "Understanding request…" / "Opening Edge…"
      ActionIndicator.tsx   # subtle tool-execution motion
      ResponseRenderer.tsx  # streamed assistant response
```

## Design tokens (v2 — warm palette, per the updated brief)

| Role | Value |
|---|---|
| Background | Cream White `#FDFBF7` |
| Primary accent | Periwinkle `#CCCCFF` |
| Secondary accent | Dusty Plum `#705553` |
| Text / anchor | Dark Olive `#3B4430` |
| Optional success | Soft Sage `#A8B89A` |

**Typography:** Inter covers every in-product role (transcripts, status,
labels, streamed text) per the spec. TAN Pearl is a licensed display face
reserved for the wordmark/hero only — it isn't bundled here. To use it, drop
the licensed font files in `renderer/public/fonts/` and uncomment the
`@font-face` block in `renderer/src/styles/globals.css`. Nothing in the
current build uses it, since there's no logo/hero screen in this prototype
yet.

## What I'd want to know before going further

1. **Real backend transport** — once the Gemma/Whisper/Kokoro pipeline
   exists, what transport will it use to send events (WebSocket, named pipe,
   IPC over a local socket)? That decides what replaces `mockBackend.ts`.
2. **Logo/hero moment** — the spec mentions TAN Pearl for a landing/welcome
   screen and floral-geometry branding, but Grace has no window chrome or
   app-launch moment by design. Is there a separate "first run" screen this
   should live on, or is the wordmark only for marketing (outside this app)?
