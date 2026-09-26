//! Real, narrow Win32 calls: the foreground-window query
//! `perception/element_graph.py`'s `ElementGraphBuilder.active_window()`
//! makes (via `win32gui.GetForegroundWindow`/`GetWindowText`/`GetClassName`/
//! `GetWindowRect`), ported to the `windows` crate.
//!
//! This is deliberately narrow: it is the safe, side-effect-free quarter of
//! `ElementGraphBuilder` (identify the foreground window) rather than the
//! whole thing (walk its UIA tree) - the tree walk needs COM interop
//! (`IUIAutomation`) this phase did not reach; see PORT_STATUS.md. Exposing
//! this much for real, rather than leaving the entire module a stub, is
//! still useful: it's what `grace_core::perception::ElementGraphBuilder`
//! (once ported/wired) would call to fill in a `WindowRef`, and it's real,
//! callable, `windows`-crate code today.
//!
//! No test here drives a real window - the one test that would is
//! `#[ignore]`d with a reason, per the task's hard constraints.

#[cfg(windows)]
pub mod real {
    use windows::core::PWSTR;
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::UI::WindowsAndMessaging::{
        GetClassNameW, GetForegroundWindow, GetWindowRect, GetWindowTextLengthW, GetWindowTextW,
    };

    /// The foreground window's identity and geometry - `hwnd` as a raw
    /// integer handle (per-run, not stable across launches; callers that
    /// need identity across a session key off `title`/`class_name` instead,
    /// same as the Python `WindowRef`/tape format does).
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ForegroundWindow {
        pub hwnd: isize,
        pub title: String,
        pub class_name: String,
        pub rect: (i32, i32, i32, i32), // left, top, right, bottom
    }

    /// Queries the current foreground window. Returns `None` if there is no
    /// foreground window (nothing focused, or the query itself failed) -
    /// mirrors `active_window()` returning a default/empty `WindowRef` for
    /// `hwnd == 0`.
    pub fn foreground_window() -> Option<ForegroundWindow> {
        // SAFETY: GetForegroundWindow takes no arguments and cannot fail in
        // a way that's unsafe to observe; it may simply return a null HWND,
        // which is checked below.
        let hwnd = unsafe { GetForegroundWindow() };
        if hwnd.is_invalid() {
            return None;
        }

        let title = window_text(hwnd);
        let class_name = class_name(hwnd);
        let rect = window_rect(hwnd).unwrap_or((0, 0, 0, 0));

        Some(ForegroundWindow { hwnd: hwnd.0 as isize, title, class_name, rect })
    }

    fn window_text(hwnd: HWND) -> String {
        // SAFETY: `hwnd` was just obtained from GetForegroundWindow and
        // checked non-null above; GetWindowTextLengthW/GetWindowTextW are
        // ordinary Win32 calls that tolerate a stale/closed handle by
        // returning 0, which the length check and buffer read both handle.
        unsafe {
            let len = GetWindowTextLengthW(hwnd);
            if len <= 0 {
                return String::new();
            }
            let mut buf = vec![0u16; len as usize + 1];
            let written = GetWindowTextW(hwnd, &mut buf);
            if written <= 0 {
                return String::new();
            }
            String::from_utf16_lossy(&buf[..written as usize])
        }
    }

    fn class_name(hwnd: HWND) -> String {
        // SAFETY: same as `window_text` - a fixed-size stack buffer and a
        // Win32 call that writes at most its capacity.
        unsafe {
            let mut buf = [0u16; 256];
            let written = GetClassNameW(hwnd, &mut buf);
            if written <= 0 {
                return String::new();
            }
            String::from_utf16_lossy(&buf[..written as usize])
        }
    }

    fn window_rect(hwnd: HWND) -> Option<(i32, i32, i32, i32)> {
        let mut rect = RECT::default();
        // SAFETY: `rect` is a valid, correctly-sized out-parameter; GetWindowRect
        // fills it in or returns an error, both of which are handled.
        unsafe { GetWindowRect(hwnd, &mut rect) }.ok()?;
        Some((rect.left, rect.top, rect.right, rect.bottom))
    }

    // Silence "unused" for PWSTR import kept for documentation parity with
    // the wide-string APIs used above (GetWindowTextW takes a raw buffer,
    // not PWSTR, in this crate's binding, but the type is worth naming here
    // for anyone extending this module with SetWindowTextW etc.).
    #[allow(unused_imports)]
    use PWSTR as _;
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "queries the real foreground window via GetForegroundWindow/GetWindowTextW/ \
                GetClassNameW/GetWindowRect; needs a real desktop, which the task's hard \
                constraints forbid in an automated test. Run manually to sanity-check \
                win32::real::foreground_window."]
    fn real_foreground_window_placeholder() {
        #[cfg(windows)]
        {
            let window = super::real::foreground_window();
            assert!(window.is_some());
        }
    }
}
