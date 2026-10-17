//! Supervision of the backend, ported from `main.js:93-141`.
//!
//! The shell owns the backend's lifetime but nothing else about it: every
//! backend meets the renderer only at the WebSocket on `127.0.0.1:8765`.
//! That separation is what lets this module choose between two backends
//! without the renderer noticing:
//!
//! - **Rust** (default as of Task 20): runs `grace-backend` as an out-of-process
//!   sidecar binary inside a Windows Job Object (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`),
//!   with per-launch random WebSocket token, single-instance guard, and health-watch
//!   supervision with exponential backoff. If `grace-backend.exe` is not found,
//!   it cleanly falls back to the in-process `grace_backend` runtime.
//! - **Python** (deprecated, selectable via `GRACE_BACKEND=python`): spawns
//!   `venv/Scripts/python.exe src/grace/main.py` as a child process. Maintained
//!   for one release with a deprecation notice before retirement.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use grace_backend::WsEventServer;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
pub struct SafeJobHandle(pub windows::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for SafeJobHandle {}
#[cfg(windows)]
unsafe impl Sync for SafeJobHandle {}

#[cfg(windows)]
type JobHandle = Option<SafeJobHandle>;

#[cfg(not(windows))]
type JobHandle = ();

/// `windowsHide: true` - keeps a console window from flashing up behind the
/// overlay every launch.
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// The repository root, which is the backend's working directory.
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

/// Which backend `Backend::start` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Rust,
    Python,
}

/// Selected backend, defaulting to `Rust`.
/// Python remains selectable for one release via `GRACE_BACKEND=python`.
pub fn selected_backend() -> BackendType {
    match std::env::var("GRACE_BACKEND") {
        Ok(v) if v.eq_ignore_ascii_case("python") => {
            eprintln!(
                "[Grace] DEPRECATION: Python backend is deprecated and will be removed in a future release. The default backend is now Rust (GRACE_BACKEND=rust)."
            );
            BackendType::Python
        }
        _ => BackendType::Rust,
    }
}

/// Searches standard paths for the standalone `grace-backend` executable.
pub fn find_rust_backend_exe(root: &Path) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "grace-backend.exe" } else { "grace-backend" };

    // 1. Next to the running executable (release/installed bundle)
    if let Ok(current) = std::env::current_exe() {
        if let Some(parent) = current.parent() {
            let candidate = parent.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // 2. target/debug or target/release relative to project root
    let candidates = [
        root.join("target").join("debug").join(exe_name),
        root.join("target").join("release").join(exe_name),
        root.join("bin").join(exe_name),
    ];

    for candidate in &candidates {
        if candidate.exists() {
            return Some(candidate.clone());
        }
    }

    None
}

/// A running backend - sidecar child process, Python child, or in-process Rust event server.
pub struct Backend {
    child: Arc<Mutex<Option<Child>>>,
    stopping: Arc<AtomicBool>,
    #[allow(dead_code)]
    rust_runtime: Option<tokio::runtime::Runtime>,
    #[cfg(windows)]
    #[allow(dead_code)]
    job_handle: Arc<Mutex<JobHandle>>,
}

/// Generate or read per-launch random WebSocket auth token (R8).
pub fn generate_auth_token() -> String {
    if let Ok(tok) = std::env::var("GRACE_WS_TOKEN") {
        if !tok.is_empty() {
            return tok;
        }
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let pid = std::process::id();
    format!("{:032x}{:08x}", now.as_nanos(), pid)
}

impl Backend {
    /// Starts the backend (Rust default, or Python if requested via `GRACE_BACKEND=python`).
    pub fn start(root: &Path, auth_token: &str) -> Self {
        match selected_backend() {
            BackendType::Python => Self::start_python(root, auth_token),
            BackendType::Rust => Self::start_rust_sidecar(root, auth_token),
        }
    }

    /// Spawns a child process for the standalone `grace-backend` binary.
    fn spawn_rust_child(exe: &Path, auth_token: &str) -> Option<(Child, JobHandle)> {
        println!("[Grace] Spawning Rust backend sidecar: {}...", exe.display());
        let mut command = Command::new(exe);
        command
            .env("GRACE_WS_TOKEN", auth_token)
            .env("WS_HOST", "127.0.0.1")
            .env("WS_PORT", "8765")
            .env("GRACE_LIVE", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let data_dir = PathBuf::from(local_app_data).join("Grace");
            let _ = std::fs::create_dir_all(&data_dir);
            command.env("GRACE_DATA_DIR", data_dir);
        }

        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        match command.spawn() {
            Ok(mut child) => {
                let job = assign_to_job_object(&child);

                if let Some(stdout) = child.stdout.take() {
                    relay(stdout, false);
                }
                if let Some(stderr) = child.stderr.take() {
                    relay(stderr, true);
                }
                Some((child, job))
            }
            Err(err) => {
                eprintln!("[Grace] Failed to spawn Rust backend sidecar: {err}");
                None
            }
        }
    }

    /// Starts the Rust backend: prefers standalone sidecar binary with Job Object
    /// and health watch; falls back to in-process runtime if binary not yet built.
    fn start_rust_sidecar(root: &Path, auth_token: &str) -> Self {
        println!("[Grace] Starting Rust backend (GRACE_BACKEND=rust, default)...");
        let stopping = Arc::new(AtomicBool::new(false));

        if let Some(exe_path) = find_rust_backend_exe(root) {
            if let Some((child, job)) = Self::spawn_rust_child(&exe_path, auth_token) {
                let child_arc = Arc::new(Mutex::new(Some(child)));
                #[cfg(windows)]
                let job_arc = Arc::new(Mutex::new(job));

                // Health watch with restart and exponential backoff
                let watcher_child = Arc::clone(&child_arc);
                let watcher_stopping = Arc::clone(&stopping);
                let exe_clone = exe_path.clone();
                let token_clone = auth_token.to_string();
                #[cfg(windows)]
                let watcher_job = Arc::clone(&job_arc);

                std::thread::spawn(move || {
                    let mut backoff = Duration::from_millis(500);
                    let max_backoff = Duration::from_secs(10);
                    let mut failures = 0;

                    while !watcher_stopping.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(1000));
                        if watcher_stopping.load(Ordering::Relaxed) {
                            break;
                        }

                        let mut guard = match watcher_child.lock() {
                            Ok(g) => g,
                            Err(e) => e.into_inner(),
                        };

                        if let Some(ref mut child) = *guard {
                            match child.try_wait() {
                                Ok(Some(status)) => {
                                    if watcher_stopping.load(Ordering::Relaxed) {
                                        break;
                                    }
                                    eprintln!(
                                        "[Grace] Rust backend sidecar exited unexpectedly ({status}). Restarting in {:?}...",
                                        backoff
                                    );
                                    std::thread::sleep(backoff);
                                    backoff = std::cmp::min(backoff * 2, max_backoff);
                                    failures += 1;
                                    if failures > 5 {
                                        eprintln!(
                                            "[Grace] Rust backend failed {failures} times consecutively. Backing off 30s."
                                        );
                                        std::thread::sleep(Duration::from_secs(30));
                                        failures = 0;
                                    }

                                    if let Some((new_child, new_job)) =
                                        Self::spawn_rust_child(&exe_clone, &token_clone)
                                    {
                                        *child = new_child;
                                        #[cfg(windows)]
                                        {
                                            let mut jg = match watcher_job.lock() {
                                                Ok(g) => g,
                                                Err(e) => e.into_inner(),
                                            };
                                            *jg = new_job;
                                        }
                                    }
                                }
                                Ok(None) => {
                                    // Healthy
                                    failures = 0;
                                    backoff = Duration::from_millis(500);
                                }
                                Err(err) => {
                                    eprintln!("[Grace] Health-watch failed to query sidecar status: {err}");
                                }
                            }
                        }
                    }
                });

                return Self {
                    child: child_arc,
                    stopping,
                    rust_runtime: None,
                    #[cfg(windows)]
                    job_handle: job_arc,
                };
            }
        }

        println!("[Grace] Standalone grace-backend binary not found; falling back to in-process Rust runtime.");
        Self::start_rust_inprocess(auth_token)
    }

    /// Starts `grace_backend`'s `WsEventServer` in-process on a dedicated
    /// multi-thread Tokio runtime.
    fn start_rust_inprocess(auth_token: &str) -> Self {
        println!("[Grace] Starting in-process Rust backend...");
        let stopping = Arc::new(AtomicBool::new(false));
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!("[Grace] Failed to start the Rust backend's runtime: {err}");
                return Self {
                    child: Arc::new(Mutex::new(None)),
                    stopping,
                    rust_runtime: None,
                    #[cfg(windows)]
                    job_handle: Arc::new(Mutex::new(None)),
                };
            }
        };

        let host = std::env::var("WS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        let port: u16 = std::env::var("WS_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8765);
        let allowed_origins = std::env::var("WS_ALLOWED_ORIGINS").unwrap_or_default();
        let token = auth_token.to_string();

        runtime.spawn(async move {
            let server = WsEventServer::new_with_token(host, port, &allowed_origins, Some(token));

            let wake_server = Arc::clone(&server);
            server
                .set_on_wake(Arc::new(move || {
                    let server = Arc::clone(&wake_server);
                    std::thread::spawn(move || {
                        if std::env::var("GRACE_LIVE")
                            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                            .unwrap_or(false)
                        {
                            grace_backend::run_live_turn_default(&server);
                        } else {
                            grace_backend::run_demo_turn(&server);
                        }
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
            std::future::pending::<()>().await;
        });

        Self {
            child: Arc::new(Mutex::new(None)),
            stopping,
            rust_runtime: Some(runtime),
            #[cfg(windows)]
            job_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Starts `venv/Scripts/python.exe src/grace/main.py` from the repo root.
    fn start_python(root: &Path, auth_token: &str) -> Self {
        let python = root.join("venv").join("Scripts").join("python.exe");
        let entry = root.join("src").join("grace").join("main.py");
        let stopping = Arc::new(AtomicBool::new(false));

        if !python.exists() {
            eprintln!(
                "[Grace] Python backend not found at {} - skipping.",
                python.display()
            );
            return Self {
                child: Arc::new(Mutex::new(None)),
                stopping,
                rust_runtime: None,
                #[cfg(windows)]
                job_handle: Arc::new(Mutex::new(None)),
            };
        }

        println!("[Grace] Starting Python backend...");
        let mut command = Command::new(&python);
        command
            .arg(&entry)
            .current_dir(root)
            .env("GRACE_WS_TOKEN", auth_token)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let data_dir = PathBuf::from(local_app_data).join("Grace");
            let _ = std::fs::create_dir_all(&data_dir);
            command.env("GRACE_DATA_DIR", data_dir);
        }

        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);

        let (child, job) = match command.spawn() {
            Ok(mut child) => {
                let job = assign_to_job_object(&child);

                if let Some(stdout) = child.stdout.take() {
                    relay(stdout, false);
                }
                if let Some(stderr) = child.stderr.take() {
                    relay(stderr, true);
                }
                (Some(child), job)
            }
            Err(err) => {
                eprintln!("[Grace] Failed to start Python backend: {err}");
                (None, assign_to_job_object_fallback())
            }
        };

        Self {
            child: Arc::new(Mutex::new(child)),
            stopping,
            rust_runtime: None,
            #[cfg(windows)]
            job_handle: Arc::new(Mutex::new(job)),
        }
    }

    /// Terminates the backend. Safe to call more than once.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Relaxed);
        let mut guard = match self.child.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        if let Some(mut child) = guard.take() {
            println!("[Grace] Stopping backend process...");
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                let pid = child.id();
                // Terminate the entire process tree so child processes like
                // llama-server.exe cannot survive Grace's termination (R29).
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .creation_flags(0x08000000) // CREATE_NO_WINDOW
                    .status();
            }
            let _ = child.kill();
            let _ = child.wait();
        }

        #[cfg(windows)]
        {
            let mut jg = match self.job_handle.lock() {
                Ok(g) => g,
                Err(e) => e.into_inner(),
            };
            if let Some(job) = jg.take() {
                unsafe {
                    let _ = windows::Win32::Foundation::CloseHandle(job.0);
                }
            }
        }
    }
}

#[cfg(windows)]
fn assign_to_job_object(child: &Child) -> JobHandle {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::*;

    unsafe {
        let job = match CreateJobObjectW(None, None) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[Grace] Failed to create job object: {e}");
                return None;
            }
        };

        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let res = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if let Err(e) = res {
            eprintln!("[Grace] Failed to set job object limit: {e}");
            return None;
        }

        let process_handle = HANDLE(child.as_raw_handle() as _);
        if let Err(e) = AssignProcessToJobObject(job, process_handle) {
            eprintln!("[Grace] Failed to assign child process to job object: {e}");
            None
        } else {
            println!("[Grace] Child process assigned to Job Object with KILL_ON_JOB_CLOSE");
            Some(SafeJobHandle(job))
        }
    }
}

#[cfg(not(windows))]
fn assign_to_job_object(_child: &Child) -> JobHandle {
    ()
}

#[cfg(windows)]
fn assign_to_job_object_fallback() -> JobHandle {
    None
}

#[cfg(not(windows))]
fn assign_to_job_object_fallback() -> JobHandle {
    ()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_is_rust() {
        let current = std::env::var("GRACE_BACKEND").ok();
        std::env::remove_var("GRACE_BACKEND");
        assert_eq!(selected_backend(), BackendType::Rust);

        std::env::set_var("GRACE_BACKEND", "rust");
        assert_eq!(selected_backend(), BackendType::Rust);

        std::env::set_var("GRACE_BACKEND", "python");
        assert_eq!(selected_backend(), BackendType::Python);

        if let Some(val) = current {
            std::env::set_var("GRACE_BACKEND", val);
        } else {
            std::env::remove_var("GRACE_BACKEND");
        }
    }

    #[test]
    fn auth_token_generation_returns_non_empty_token() {
        let token = generate_auth_token();
        assert!(!token.is_empty());
        assert!(token.len() >= 32);
    }
}

