//! Hover detection for a window the mouse cannot see.
//!
//! Electron collapsed the overlay with
//! `setIgnoreMouseEvents(true, { forward: true })`, and the `forward` half is
//! load-bearing in a way that is easy to miss: it kept *forwarding mouse-move
//! messages to the page* while clicks passed through. That is what let the
//! renderer notice the pointer arriving on the pill (`onMouseEnter`) and ask
//! for the window to become solid again.
//!
//! `WS_EX_TRANSPARENT` - the native primitive Tauri exposes as
//! `set_ignore_cursor_events` - has no such half-measure. It removes the window
//! from hit-testing entirely, so the webview receives no mouse messages at all.
//! Ported literally, the renderer's hover handler could therefore never fire,
//! which meant click-through could never be turned off, which meant clicking
//! the idle pill to wake Grace stopped working altogether. A dead wake path is
//! not a cosmetic difference for a user who cannot reach for a keyboard.
//!
//! So the shell watches the cursor itself. The renderer reports the rectangle
//! it wants to be touchable - the pill, which is a small part of a mostly
//! transparent window - and this module polls the cursor against it, dropping
//! click-through only while the pointer is genuinely over it. The observable
//! behaviour matches Electron's: the pill is solid under the pointer, and every
//! other pixel of the overlay stays click-through.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use tauri::WebviewWindow;
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, GetForegroundWindow};

/// How often the cursor is sampled.
///
/// Electron's forwarding was message-driven and so effectively instant; this is
/// a poll, and the interval is the tradeoff between idle CPU and how long the
/// pill feels dead after the pointer lands on it. At 16 ms the wake-up is
/// imperceptible and the loop still sleeps ~99% of the time.
const POLL_INTERVAL: Duration = Duration::from_millis(16);

/// A rectangle in CSS pixels, relative to the top-left of the webview.
///
/// CSS pixels rather than physical ones because that is the only coordinate
/// space the renderer can report without knowing the scale factor, and the
/// conversion needs the live scale factor anyway to survive a DPI change.
#[derive(Clone, Copy, Deserialize)]
pub struct Region {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

/// The touchable region, shared between the command that sets it and the thread
/// that reads it.
#[derive(Default)]
pub struct HoverRegion(Mutex<Option<Region>>);

impl HoverRegion {
    pub fn set(&self, region: Option<Region>) {
        if let Ok(mut guard) = self.0.lock() {
            *guard = region;
        }
    }

    fn get(&self) -> Option<Region> {
        self.0.lock().ok().and_then(|guard| *guard)
    }
}

/// Starts watching the cursor, toggling click-through as it enters and leaves
/// the reported region.
pub fn track(window: WebviewWindow, region: Arc<HoverRegion>) {
    std::thread::spawn(move || {
        // Mirrors the state applied at startup, so the first toggle is only
        // issued when something actually changes rather than on the first tick.
        let mut interactive = false;
        let mut foreground = HWND::default();

        loop {
            std::thread::sleep(POLL_INTERVAL);

            // A new foreground window is the moment the overlay can be pushed
            // out of the topmost band - an app being launched or going
            // fullscreen, which for Grace is routinely something it just did
            // itself on the user's behalf.
            let current = unsafe { GetForegroundWindow() };
            if current != foreground {
                foreground = current;
                if let Err(err) = crate::window::assert_topmost(&window) {
                    eprintln!("[Grace] could not restore always-on-top: {err}");
                }
            }

            let wanted = cursor_is_over(&window, &region);
            if wanted == interactive {
                continue;
            }
            interactive = wanted;

            if let Err(err) = crate::window::set_click_through(&window, !interactive) {
                eprintln!("[Grace] could not update click-through: {err}");
            }
        }
    });
}

/// Whether the cursor currently sits inside the reported region.
///
/// Any failure to answer reads as "no", which fails safe: the overlay stays
/// click-through and the desktop underneath keeps working. The opposite default
/// would leave an invisible window swallowing clicks with nothing on screen to
/// explain why.
fn cursor_is_over(window: &WebviewWindow, region: &HoverRegion) -> bool {
    let Some(region) = region.get() else {
        return false;
    };
    let (Ok(origin), Ok(scale)) = (window.inner_position(), window.scale_factor()) else {
        return false;
    };

    let mut cursor = POINT::default();
    if unsafe { GetCursorPos(&mut cursor) }.is_err() {
        return false;
    }

    let left = origin.x as f64 + region.x * scale;
    let top = origin.y as f64 + region.y * scale;
    let right = left + region.width * scale;
    let bottom = top + region.height * scale;

    let (x, y) = (cursor.x as f64, cursor.y as f64);
    x >= left && x < right && y >= top && y < bottom
}
