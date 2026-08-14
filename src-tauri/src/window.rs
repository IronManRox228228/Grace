//! The native window behaviour Tauri does not model.
//!
//! Three of the overlay's properties are load-bearing for an assistant that
//! drives *other* applications:
//!
//! - **It must never take focus.** Grace narrates automation while it happens.
//!   A pill that activates on click steals focus from the window being driven,
//!   and the click that was just dispatched lands somewhere else.
//! - **It must be click-through when collapsed**, so it never blocks the
//!   desktop it floats over.
//! - **It must sit above whatever it is narrating.**
//!
//! All three turn out to be expressible in tao's own window-flag model —
//! `focusable: false`, `set_ignore_cursor_events`, and `alwaysOnTop` — and that
//! matters more than it looks. tao keeps a `WindowFlags` bitset as the single
//! source of truth for the window's styles, and on *any* flag change it
//! recomputes both `GWL_STYLE` and `GWL_EXSTYLE` from that bitset alone
//! (`tao::platform_impl::windows::window_state`, `apply_diff`). Extended styles
//! poked in behind its back with `SetWindowLongPtrW` therefore survive only
//! until the next `show()`, `set_position()`, or click-through toggle, at which
//! point they are silently erased. An overlay that loses `WS_EX_NOACTIVATE` the
//! first time the user hovers it is worse than one that never had it, because
//! the failure appears mid-session rather than at startup.
//!
//! So what is left here is the one thing tao genuinely does not expose: the
//! primary monitor's *work area*. When Phase 4 adds `grace-win`, this moves
//! there.

use tauri::{PhysicalPosition, WebviewWindow};

/// Breathing room above the taskbar, in logical pixels (`main.js:22`).
const BOTTOM_MARGIN: f64 = 28.0;

#[cfg(not(windows))]
compile_error!(
    "Grace's overlay is Windows-only: focus suppression and click-through are \
     Win32 extended window styles with no cross-platform equivalent."
);

use std::ffi::c_void;

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::UI::WindowsAndMessaging::{
    SetWindowPos, SystemParametersInfoW, HWND_TOPMOST, SPI_GETWORKAREA, SWP_NOACTIVATE, SWP_NOMOVE,
    SWP_NOSIZE, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS,
};

/// Puts the overlay back into the topmost band.
///
/// `alwaysOnTop` in `tauri.conf.json` gets it there at creation, but staying
/// there is a separate matter from carrying `WS_EX_TOPMOST`. Band membership is
/// only ever established by a `SetWindowPos` with `HWND_TOPMOST`, and tao
/// issues one solely when its always-on-top *flag changes*; every other update
/// it makes passes `SWP_NOZORDER`. So if anything displaces the overlay - a
/// window entering fullscreen, another topmost window activating over it - the
/// style bit still reads as set while the pill sits behind something, and
/// nothing in the normal course of events restores it.
///
/// Cheap enough to call whenever the foreground window changes, which is when
/// displacement happens.
pub fn assert_topmost(window: &WebviewWindow) -> Result<(), String> {
    let raw: *mut c_void = window.hwnd().map_err(|e| e.to_string())?.0;
    unsafe {
        SetWindowPos(
            HWND(raw),
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            // Never activate: taking focus is the one thing this overlay must
            // not do, and re-asserting z-order must not become a way to do it.
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
        .map_err(|e| e.to_string())
    }
}

/// Toggles hit-test transparency.
///
/// `true` means clicks pass through to the desktop, which is the idle state.
/// The renderer flips this off while the pointer is over the expanded pill.
///
/// This is Electron's `setIgnoreMouseEvents(ignore, {forward: true})`. There is
/// no `forward` argument because the underlying primitive has no such
/// distinction: tao sets `WS_EX_TRANSPARENT`, which always forwards hit-testing
/// to the window behind. Electron's flag existed to opt into that behaviour,
/// which is the only behaviour here.
pub fn set_click_through(window: &WebviewWindow, click_through: bool) -> Result<(), String> {
    window
        .set_ignore_cursor_events(click_through)
        .map_err(|e| e.to_string())
}

/// Bottom-centre of the primary display's work area (`main.js:28-32`).
///
/// The work area, not the monitor, so the pill clears the taskbar wherever the
/// taskbar happens to be docked.
pub fn place_bottom_center(window: &WebviewWindow) -> Result<(), String> {
    let size = window.outer_size().map_err(|e| e.to_string())?;
    let scale = window.scale_factor().map_err(|e| e.to_string())?;
    let work = primary_work_area()?;

    let x = work.left + ((work.right - work.left) - size.width as i32) / 2;
    let y = work.bottom - size.height as i32 - (BOTTOM_MARGIN * scale).round() as i32;

    window
        .set_position(PhysicalPosition::new(x, y))
        .map_err(|e| e.to_string())
}

/// The primary monitor's work area in physical pixels.
///
/// Tauri reports monitor bounds but not the work area, and the difference is
/// exactly the taskbar - which is what the pill has to sit above.
fn primary_work_area() -> Result<RECT, String> {
    let mut rect = RECT::default();
    unsafe {
        SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            Some(&mut rect as *mut RECT as *mut c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
        .map_err(|e| format!("could not read the primary work area: {e}"))?;
    }
    Ok(rect)
}
