//! Supervision of the backend, ported from `main.js:93-141`.
//!
//! The shell owns the backend's lifetime but nothing else about it: every
//! backend still meets the renderer only at the WebSocket on
//! `127.0.0.1:8765`. That separation is what lets this module choose between
//! two backends without the renderer noticing:
//!
//! - **Python** (default): spawns `venv/Scripts/python.exe src/grace/main.py`
//!   as a child process, exactly as before. This is the proven path and
//!   stays the default until the Rust port's parity is established (task
//!   rule: "keep the Python child path available behind a config/feature
//!   flag until parity is proven; don't delete any Python").
//! - **Rust** (opt-in via `GRACE_BACKEND=rust`): runs `grace_backend`'s
//!   `WsEventServer` in-process, on a dedicated Tokio runtime owned by this
//!   struct, instead of spawning a child at all. Every wake request runs one
//!   real activation turn via `grace_backend::run_demo_turn` - see that
//!   function's doc comment for exactly what "real" means today (every
//!   decision component is the ported one; the transcript is a fixed
//!   stand-in for speech, since no STT is wired up yet - PORT_STATUS.md).
//!   In-process rather than a separate sidecar binary because a second OS
//!   process buys crash isolation for work that could hang (model
//!   inference, `llama-server.exe`) and this demo turn's own HTTP call
//!   already has its own timeout and falls back cleanly; splitting it into
//!   a sidecar (with the Job Object/health-watch plumbing PLAN.md §10.2
//!   describes) remains the plan once a real STT/model pipeline makes that
//!   isolation worth its complexity.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use grace_backend::WsEventServer;

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

/// Which backend `Backend::start` runs, read from `GRACE_BACKEND`. Anything
/// other than exactly `"rust"` (case-insensitive) keeps the proven Python
/// path - this is deliberately fail-safe-to-Python, not fail-safe-to-Rust.
fn wants_rust_backend() -> bool {
    std::env::var("GRACE_BACKEND")
        .map(|v| v.eq_ignore_ascii_case("rust"))
        .unwrap_or(false)
}

/// A running backend - the Python child, or the in-process Rust event
/// server - or nothing if neither could be started.
pub struct Backend {
    child: Mutex<Option<Child>>,
    /// Only set on the Rust path. Never read again after `start_rust` builds
    /// it - its only job is to outlive the server's tasks by outliving
    /// `self`, so dropping it early (e.g. if this field didn't exist and the
    /// runtime were local to `start_rust`) would abort every task the server
    /// spawned onto it the moment `start_rust` returned.
    #[allow(dead_code)]
    rust_runtime: Option<tokio::runtime::Runtime>,
}

impl Backend {
    /// Starts the backend named by `GRACE_BACKEND` (`python`, the default, or
    /// `rust`) from the repo root.
    pub fn start(root: &Path) -> Self {
        if wants_rust_backend() {
            return Self::start_rust();
        }
        Self::start_python(root)
    }

    /// Starts `venv/Scripts/python.exe src/grace/main.py` from the repo root.
    ///
    /// A missing interpreter is a warning rather than an error, exactly as in
    /// Electron: the overlay is still useful against a backend someone started
    /// by hand, which is how the record/replay harness is driven.
    fn start_python(root: &Path) -> Self {
        let python = root.join("venv").join("Scripts").join("python.exe");
        let entry = root.join("src").join("grace").join("main.py");

        if !python.exists() {
            eprintln!(
                "[Grace] Python backend not found at {} - skipping.",
                python.display()
            );
            return Self {
                child: Mutex::new(None),
                rust_runtime: None,
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
            rust_runtime: None,
        }
    }

    /// Starts `grace_backend`'s `WsEventServer` in-process on a dedicated
    /// multi-thread Tokio runtime. Experimental: see this module's doc
    /// comment for exactly what does (and does not) work on this path yet.
    fn start_rust() -> Self {
        println!("[Grace] Starting in-process Rust backend (GRACE_BACKEND=rust)...");
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!("[Grace] Failed to start the Rust backend's runtime: {err}");
                return Self {
                    child: Mutex::new(None),
                    rust_runtime: None,
                };
            }
        };

        let host = std::env::var("WS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port: u16 = std::env::var("WS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8765);
        let allowed_origins = std::env::var("WS_ALLOWED_ORIGINS").unwrap_or_default();

        runtime.spawn(async move {
            let server = WsEventServer::new(host, port, &allowed_origins);

            // Every wake request (the global hotkey, or the renderer's idle
            // pill) runs one real activation turn through
            // `grace_backend::run_demo_turn` - see that function's doc
            // comment for exactly what "real" means here (every decision
            // component is real; the transcript is a fixed stand-in for
            // speech, since no STT is wired up yet).
            let wake_server = Arc::clone(&server);
            server
                .set_on_wake(Arc::new(move || {
                    let server = Arc::clone(&wake_server);
                    std::thread::spawn(move || {
                        grace_backend::run_demo_turn(&server);
                    });
                }))
                .await;

            match server.serve().await {
                Ok(addr) => {
                    println!("[Grace] Rust backend listening on {addr}");
                    server.emit(grace_contract::GraceEvent::Idle);
                }
                Err(err) => eprintln!("[Grace] Rust backend failed to bind: {err}"),
            }
            // Keep this task alive for the runtime's lifetime; the listener
            // it spawned inside `serve()` runs on its own task and does not
            // need this one to stay busy, but parking here makes the intent
            // ("this runtime exists to run the server") readable at the call
            // site instead of implicit.
            std::future::pending::<()>().await;
        });

        Self {
            child: Mutex::new(None),
            rust_runtime: Some(runtime),
        }
    }

    /// Terminates the backend. Safe to call more than once.
    pub fn stop(&self) {
        let mut guard = match self.child.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        if let Some(mut child) = guard.take() {
            println!("[Grace] Stopping Python backend...");
            // Electron's `kill('SIGTERM')` is `TerminateProcess` on Windows, and
            // so is this - the graceful-then-forceful pair in main.js only ever
            // took the forceful branch. The backend's own cleanup of
            // `llama-server.exe` is unaffected either way; it never ran under
            // Electron.
            let _ = child.kill();
            let _ = child.wait();
        }
        // The Rust runtime (if any) is dropped along with `self` at process
        // exit, which aborts its tasks - there is no child process to signal
        // on this path, only in-process tasks that die with the shell itself.
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
