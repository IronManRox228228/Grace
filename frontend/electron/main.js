const { app, BrowserWindow, screen, globalShortcut, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

const isDev = process.env.NODE_ENV === 'development';

// iGPU-optimized 144 Hz refresh rate switches
app.commandLine.appendSwitch('disable-frame-rate-limit');
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');

// Paths relative to this file: frontend/electron/main.js → project root
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const PYTHON_EXE = path.join(PROJECT_ROOT, 'venv', 'Scripts', 'python.exe');
const PYTHON_MAIN = path.join(PROJECT_ROOT, 'src', 'grace', 'main.py');

// Footprint of the overlay window. It stays modest — Grace is a pill,
// not a window — but tall enough to fit the expanded transcript panel.
const WINDOW_WIDTH = 620;
const WINDOW_HEIGHT = 420;
const BOTTOM_MARGIN = 28; // breathing room above the taskbar

let graceWindow = null;
let pythonProcess = null;

function createWindow() {
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width: screenWidth, height: screenHeight } = primaryDisplay.workAreaSize;

  const x = Math.round((screenWidth - WINDOW_WIDTH) / 2);
  const y = screenHeight - WINDOW_HEIGHT - BOTTOM_MARGIN;

  graceWindow = new BrowserWindow({
    width: WINDOW_WIDTH,
    height: WINDOW_HEIGHT,
    x,
    y,
    frame: false,
    transparent: true,
    hasShadow: false,
    resizable: false,
    movable: false,
    fullscreenable: false,
    minimizable: false,
    maximizable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    backgroundColor: '#00000000',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Sit above most things (including fullscreen apps) without stealing focus.
  graceWindow.setAlwaysOnTop(true, 'screen-saver');
  graceWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  if (isDev) {
    graceWindow.loadURL('http://localhost:5173');
    // graceWindow.webContents.openDevTools({ mode: 'detach' });
  } else {
    graceWindow.loadFile(path.join(__dirname, '..', 'renderer', 'dist', 'index.html'));
  }

  graceWindow.on('closed', () => {
    graceWindow = null;
  });
}

// The real backend owns wake-word detection. Until it's wired up, a
// global shortcut simulates "Hey Grace" so the interaction can be
// demoed end-to-end. Swap this for the backend's WakeWordDetected
// event when it's available.
function registerDevWakeShortcut() {
  globalShortcut.register('CommandOrControl+Alt+G', () => {
    if (graceWindow) {
      graceWindow.webContents.send('grace:simulate-wake');
    }
  });
}

// Lets the renderer grow/shrink the click-through area as Grace
// expands and collapses, so the desktop stays interactive around it.
ipcMain.on('grace:set-ignore-mouse-events', (event, ignore, options) => {
  if (graceWindow) {
    graceWindow.setIgnoreMouseEvents(ignore, options);
  }
});

function startPythonBackend() {
  const fs = require('fs');
  if (!fs.existsSync(PYTHON_EXE)) {
    console.warn(`[Grace] Python backend not found at ${PYTHON_EXE} — skipping.`);
    return;
  }

  console.log('[Grace] Starting Python backend...');
  pythonProcess = spawn(PYTHON_EXE, [PYTHON_MAIN], {
    cwd: PROJECT_ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  pythonProcess.stdout.on('data', (data) => {
    for (const line of data.toString().trim().split('\n')) {
      console.log(`[Backend] ${line}`);
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    for (const line of data.toString().trim().split('\n')) {
      console.error(`[Backend] ${line}`);
    }
  });

  pythonProcess.on('exit', (code) => {
    console.log(`[Grace] Python backend exited with code ${code}`);
    pythonProcess = null;
  });

  pythonProcess.on('error', (err) => {
    console.error(`[Grace] Failed to start Python backend: ${err.message}`);
    pythonProcess = null;
  });
}

function stopPythonBackend() {
  if (!pythonProcess) return;
  console.log('[Grace] Stopping Python backend...');
  pythonProcess.kill('SIGTERM');
  // Give it a moment, then force-kill
  setTimeout(() => {
    if (pythonProcess) {
      pythonProcess.kill('SIGKILL');
      pythonProcess = null;
    }
  }, 5000);
}

app.whenReady().then(() => {
  startPythonBackend();
  createWindow();
  registerDevWakeShortcut();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  globalShortcut.unregisterAll();
  if (process.platform !== 'darwin') app.quit();
});

app.on('will-quit', () => {
  stopPythonBackend();
});
