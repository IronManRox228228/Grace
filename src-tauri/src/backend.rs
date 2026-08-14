//! Supervision of the Python backend, ported from `main.js:93-141`.
//!
//! The shell owns the backend's lifetime but nothing else about it: the two
//! processes still meet only at the WebSocket on `127.0.0.1:8765`. That
//! separation is what lets Phase 2 replace the child here with a Rust binary
//! without the renderer noticing.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// `windowsHide: true` - keeps a console window from flashing up behind the
/// overlay every launch.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The repository root, which is the backend's working directory.
///
/// In a dev run the shell executable lives in `target/debug`, so it cannot be
/// used to find the source tree; the manifest directory can, and it is resolved
/// at compile time. An installed build has everything laid out next to the
/// executable instead.
pub fn project_root() -> PathBuf {
    if cfg!(debug_assertions) {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf))
            .unwrap_or_default()
    }
}

/// A running Python backend, or nothing if it could not be started.
pub struct Backend {
    child: Mutex<Option<Child>>,
}

impl Backend {
    /// Starts `venv/Scripts/python.exe src/grace/main.py` from the repo root.
    ///
    /// A missing interpreter is a warning rather than an error, exactly as in
    /// Electron: the overlay is still useful against a backend someone started
    /// by hand, which is how the record/replay harness is driven.
    pub fn start(root: &Path) -> Self {
        let python = root.join("venv").join("Scripts").join("python.exe");
        let entry = root.join("src").join("grace").join("main.py");

        if !python.exists() {
            eprintln!(
                "[Grace] Python backend not found at {} - skipping.",
                python.display()
            );
            return Self {
                child: Mutex::new(None),
            };
        }

        println!("[Grace] Starting Python backend...");
        let mut command = Command::new(&python);
        command
            .arg(&entry)
            .current_dir(root)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        let child = match command.spawn() {
            Ok(mut child) => {
                if let Some(stdout) = child.stdout.take() {
                    relay(stdout, false);
                }
                if let Some(stderr) = child.stderr.take() {
                    relay(stderr, true);
                }
                Some(child)
            }
            Err(err) => {
                eprintln!("[Grace] Failed to start Python backend: {err}");
                None
            }
        };

        Self {
            child: Mutex::new(child),
        }
    }

    /// Terminates the backend. Safe to call more than once.
    pub fn stop(&self) {
        let Ok(mut guard) = self.child.lock() else {
            return;
        };
        let Some(mut child) = guard.take() else {
            return;
        };
        println!("[Grace] Stopping Python backend...");
        // Electron's `kill('SIGTERM')` is `TerminateProcess` on Windows, and so
        // is this - the graceful-then-forceful pair in main.js only ever took
        // the forceful branch. The backend's own cleanup of `llama-server.exe`
        // is unaffected either way; it never ran under Electron.
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// Forwards one of the child's streams to the console, line by line.
fn relay<R: Read + Send + 'static>(stream: R, is_stderr: bool) {
    std::thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            if is_stderr {
                eprintln!("[Backend] {line}");
            } else {
                println!("[Backend] {line}");
            }
        }
    });
}
