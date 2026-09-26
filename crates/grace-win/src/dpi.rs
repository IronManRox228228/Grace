//! Ported from `src/grace/automation/dpi_helper.py`'s `DPIHelper`: per-monitor
//! DPI awareness and coordinate scaling, so UIA logical coordinates,
//! screenshot physical pixels, and `SendInput` virtual-screen coordinates
//! all map 1:1 under Windows display scaling (125/150/200%).
//!
//! `scale_coords` is pure and tested normally. Everything else is a real
//! Win32 call behind `cfg(windows)` (`GetDpiForWindow`, falling back to
//! `GetDeviceCaps(LOGPIXELSX)`, matching Python's own fallback chain
//! exactly) - these cannot be unit-tested without a real desktop, so the one
//! test that would exercise them for real is `#[ignore]`d with a reason,
//! per the task's hard constraints.

/// Scale a point from one coordinate space to another. If a source
/// dimension is non-positive the coordinate is returned unchanged, avoiding
/// a division by zero exactly as the Python version's `try`/`except` does.
pub fn scale_coords(x: i32, y: i32, src_w: i32, src_h: i32, dst_w: i32, dst_h: i32) -> (i32, i32) {
    if src_w <= 0 || src_h <= 0 {
        return (x, y);
    }
    (
        (x as i64 * dst_w as i64 / src_w as i64) as i32,
        (y as i64 * dst_h as i64 / src_h as i64) as i32,
    )
}

/// The Win32-metric constants Python's `dpi_helper.py` uses directly.
pub const USER_DEFAULT_SCREEN_DPI: f64 = 96.0;

#[cfg(windows)]
pub mod real {
    //! Real Win32 calls. Not covered by an automated test (would need a
    //! real desktop/display); see `dpi::tests::real_dpi_query_placeholder`.
    use super::USER_DEFAULT_SCREEN_DPI;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::Graphics::Gdi::{GetDC, GetDeviceCaps, ReleaseDC, LOGPIXELSX};
    use windows::Win32::UI::HiDpi::{GetDpiForWindow, SetProcessDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2};
    use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

    /// `SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)`, once per
    /// process. Mirrors `DPIHelper.ensure_dpi_aware`'s primary path;
    /// Python's `SetProcessDPIAware()` fallback for pre-1703 Windows isn't
    /// ported since this crate's own `windows-version` floor already implies
    /// a build with `SetProcessDpiAwarenessContext` available.
    pub fn ensure_dpi_aware() -> bool {
        // SAFETY: no preconditions beyond "called from a process that
        // hasn't already set a DPI awareness mode" - calling it twice is
        // harmless (it just fails the second time), and the return value is
        // deliberately ignored for that reason, matching Python's own
        // best-effort semantics.
        unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) }.is_ok()
    }

    /// The per-monitor DPI scale factor (1.0, 1.25, 1.5, 2.0, ...) for
    /// `hwnd`, or the foreground window if `None`. `physical_px = logical_px
    /// * scale`.
    pub fn get_dpi_scale(hwnd: Option<isize>) -> f64 {
        ensure_dpi_aware();

        let hwnd = hwnd
            .map(|h| HWND(h as *mut core::ffi::c_void))
            .unwrap_or_else(|| unsafe { GetForegroundWindow() });

        if !hwnd.is_invalid() {
            let dpi = unsafe { GetDpiForWindow(hwnd) };
            if dpi > 0 {
                return round2(dpi as f64 / USER_DEFAULT_SCREEN_DPI);
            }
        }

        // Fallback: system DPI via GDI GetDeviceCaps(LOGPIXELSX).
        unsafe {
            let hdc = GetDC(None);
            if !hdc.is_invalid() {
                let dpi = GetDeviceCaps(Some(hdc), LOGPIXELSX);
                ReleaseDC(None, hdc);
                if dpi > 0 {
                    return round2(dpi as f64 / USER_DEFAULT_SCREEN_DPI);
                }
            }
        }

        1.0
    }

    fn round2(value: f64) -> f64 {
        (value * 100.0).round() / 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scales_proportionally() {
        assert_eq!(scale_coords(100, 100, 1280, 720, 1920, 1080), (150, 150));
    }

    #[test]
    fn zero_source_dimension_is_a_no_op() {
        assert_eq!(scale_coords(10, 20, 0, 720, 1920, 1080), (10, 20));
        assert_eq!(scale_coords(10, 20, 1280, 0, 1920, 1080), (10, 20));
    }

    #[test]
    fn negative_source_dimension_is_a_no_op() {
        assert_eq!(scale_coords(10, 20, -5, 720, 1920, 1080), (10, 20));
    }

    #[test]
    #[ignore = "queries the real foreground window's DPI via GetDpiForWindow/GetDeviceCaps; \
                needs a real desktop/display, which the task's hard constraints forbid in an \
                automated test. Run manually on a dev machine to sanity-check dpi::real::get_dpi_scale."]
    fn real_dpi_query_placeholder() {
        #[cfg(windows)]
        {
            let scale = real::get_dpi_scale(None);
            assert!(scale > 0.0);
        }
    }
}
