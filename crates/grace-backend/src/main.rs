//! Standalone binary entrypoint for the Grace Rust backend sidecar.
//!
//! Started as a child process by the Tauri overlay shell (`src-tauri/src/backend.rs`)
//! inside a Windows Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
//!
//! Features:
//! - Single-instance guard (`Local\GraceBackendSingleInstanceMutex`)
//! - Per-launch random WebSocket authentication token verification
//! - Origin allowlist enforcement
//! - Real Windows automation / perception / system actions when `GRACE_LIVE=1`
//! - Graceful shutdown on SIGINT / SIGTERM / Ctrl+C

use anyhow::Result;
use grace_backend::server::WsEventServer;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(windows)]
struct SingleInstanceGuard(windows::Win32::Foundation::HANDLE);

#[cfg(windows)]
impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(windows)]
fn acquire_single_instance() -> Result<SingleInstanceGuard, ()> {
    use windows::core::w;
    use windows::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};
    use windows::Win32::System::Threading::CreateMutexW;

    unsafe {
        let handle = CreateMutexW(None, true, w!("Local\\GraceBackendSingleInstanceMutex"))
            .map_err(|_| ())?;
        if GetLastError() == ERROR_ALREADY_EXISTS {
            let _ = windows::Win32::Foundation::CloseHandle(handle);
            return Err(());
        }
        Ok(SingleInstanceGuard(handle))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    #[cfg(windows)]
    let _instance_guard = match acquire_single_instance() {
        Ok(guard) => Some(guard),
        Err(_) => {
            eprintln!("[grace-backend] Another instance of grace-backend is already running. Exiting.");
            return Ok(());
        }
    };

    println!("[grace-backend] Starting Grace Rust Backend Sidecar...");

    // Setup writable data directory (%LOCALAPPDATA%\Grace by default)
    let data_dir = match std::env::var("GRACE_DATA_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                PathBuf::from(local_app_data).join("Grace")
            } else {
                PathBuf::from(".grace")
            }
        }
    };
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        eprintln!("[grace-backend] Warning: could not create data dir {}: {e}", data_dir.display());
    } else {
        println!("[grace-backend] Data dir: {}", data_dir.display());
    }

    let host = std::env::var("WS_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("WS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8765);
    let allowed_origins = std::env::var("WS_ALLOWED_ORIGINS").unwrap_or_default();
    let token = std::env::var("GRACE_WS_TOKEN").ok().filter(|s| !s.is_empty());

    let server = WsEventServer::new_with_token(host, port, &allowed_origins, token);

    // Set wake handler
    let wake_server = Arc::clone(&server);
    server
        .set_on_wake(Arc::new(move || {
            let server = Arc::clone(&wake_server);
            std::thread::spawn(move || {
                let live = std::env::var("GRACE_LIVE")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(true);
                if live {
                    grace_backend::run_live_turn_default(&server);
                } else {
                    grace_backend::run_demo_turn(&server);
                }
            });
        }))
        .await;

    let addr = server.serve().await?;
    println!("[grace-backend] Server listening on {addr}");
    server.emit(grace_contract::GraceEvent::Idle);

    // Serve until process termination
    std::future::pending::<()>().await;

    Ok(())
}
