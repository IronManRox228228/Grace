//! Grace's overlay shell.
//!
//! This replaces `frontend/electron/main.js` and does the same three jobs it
//! did: put a transparent, focus-proof pill at the bottom of the primary
//! display, keep the Python backend alive underneath it, and turn `Ctrl+Alt+G`
//! into a wake request.
//!
//! What it deliberately does *not* do is carry application state. Every event
//! the renderer reacts to still arrives over the WebSocket at
//! `127.0.0.1:8765`, so the shell can be swapped, and later the backend behind
//! it can be swapped, without either end learning about the other.

mod backend;
mod hover;
mod window;

use std::sync::Arc;

use tauri::{Emitter, Manager, RunEvent, State};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

use hover::{HoverRegion, Region};

/// Matches the `label` in `tauri.conf.json`.
const WINDOW_LABEL: &str = "grace";

/// Emitted to the renderer when the wake shortcut fires. The renderer answers
/// it the same way it answers a click on the idle pill - by sending `{"type":
/// "wake"}` over the WebSocket - so the backend sees one activation path, not
/// two.
const WAKE_EVENT: &str = "grace://wake";

/// Reports which part of the overlay should respond to the pointer.
///
/// This is what replaces Electron's `onMouseEnter`/`onMouseLeave` pair. The
/// renderer no longer toggles click-through itself, because under
/// `WS_EX_TRANSPARENT` it never learns the pointer arrived; it declares the
/// pill's rectangle instead and the shell watches the cursor. See `hover.rs`.
#[tauri::command]
fn set_hover_region(state: State<'_, Arc<HoverRegion>>, region: Option<Region>) {
    state.set(region);
}

/// Returns the per-launch WebSocket auth token to the renderer (R8).
#[tauri::command]
fn get_ws_auth_token(state: State<'_, String>) -> String {
    state.inner().clone()
}

/// Returns whether Grace is running in strict offline mode (R19).
#[tauri::command]
fn get_offline_mode() -> bool {
    std::env::var("USE_CLOUD_LLM")
        .map(|v| v.to_lowercase() != "true" && v != "1")
        .unwrap_or(true)
}

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
        let handle = CreateMutexW(None, true, w!("Local\\GraceOverlaySingleInstanceMutex"))
            .map_err(|_| ())?;
        if GetLastError() == ERROR_ALREADY_EXISTS {
            let _ = windows::Win32::Foundation::CloseHandle(handle);
            return Err(());
        }
        Ok(SingleInstanceGuard(handle))
    }
}

/// Builds and runs the overlay. Blocks until the app exits.
pub fn run() {
    #[cfg(windows)]
    let _instance_guard = match acquire_single_instance() {
        Ok(guard) => guard,
        Err(_) => {
            eprintln!("[Grace] Another instance of Grace is already running. Exiting.");
            return;
        }
    };

    let auth_token = backend::generate_auth_token();
    // Started before the window so the backend's model loading overlaps with
    // the webview's, as it did under Electron.
    let backend = backend::Backend::start(&backend::project_root(), &auth_token);

    let managed_token = auth_token.clone();
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![set_hover_region, get_ws_auth_token, get_offline_mode])
        .setup(move |app| {
            app.manage(managed_token);
            let window = app
                .get_webview_window(WINDOW_LABEL)
                .ok_or_else(|| format!("no window labelled {WINDOW_LABEL}"))?;

            let region = Arc::new(HoverRegion::default());
            app.manage(Arc::clone(&region));

            // Ordered deliberately. The window is created hidden so that none
            // of this is observable: placing it after it was already on screen
            // shows the pill top-left for a frame or two. Focus suppression and
            // always-on-top are declared in tauri.conf.json rather than applied
            // here, so they hold from creation - see `window.rs` for why doing
            // it imperatively does not survive.
            window::place_bottom_center(&window)?;
            window::set_click_through(&window, true)?;
            window.show()?;

            hover::track(window, region);
            register_wake_shortcut(app.handle())?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("the Grace overlay failed to build")
        .run(move |_app, event| {
            // Electron stopped the backend from `will-quit`. Anything earlier
            // races the window teardown and can leave the child orphaned,
            // holding the microphone and the WebSocket port against the next
            // launch.
            if matches!(event, RunEvent::Exit) {
                backend.stop();
            }
        });
}

fn register_wake_shortcut(app: &tauri::AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let wake = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::KeyG);
    let watched = wake;

    app.plugin(
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(move |app, shortcut, event| {
                // Press only: without this the release fires a second wake, and
                // the backend would start a listening window the user did not
                // ask for.
                if *shortcut == watched && event.state() == ShortcutState::Pressed {
                    // Reported rather than discarded: if this fails the wake
                    // shortcut is dead, and it fails silently otherwise - the
                    // user presses the key and simply nothing happens, with no
                    // way to tell the shortcut from the backend as the culprit.
                    if let Err(err) = app.emit(WAKE_EVENT, ()) {
                        eprintln!("[Grace] could not emit {WAKE_EVENT}: {err}");
                    }
                }
            })
            .build(),
    )?;
    app.global_shortcut().register(wake)?;
    Ok(())
}
