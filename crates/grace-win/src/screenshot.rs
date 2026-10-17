//! Screenshot capture of the foreground window for the vision slot.
//!
//! Captures the target window (or full display fallback) using GDI `BitBlt`,
//! reads pixels via `GetDIBits`, and encodes to PNG bytes using the `png` crate.

use anyhow::Result;

/// Encodes raw RGBA pixels into PNG format bytes.
pub fn encode_rgba_to_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let mut png_bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut png_bytes, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels)?;
    drop(writer);
    Ok(png_bytes)
}

#[cfg(windows)]
pub mod real {
    use super::*;
    use windows::Win32::Foundation::{HWND, RECT};
    use windows::Win32::Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC,
        GetDIBits, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER,
        BI_RGB, DIB_RGB_COLORS, SRCCOPY,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetSystemMetrics, GetWindowRect, SM_CXSCREEN, SM_CYSCREEN,
    };

    /// Captures the specified window (or foreground window if None) as PNG bytes.
    /// Returns `(png_bytes, width, height)`.
    pub fn capture_foreground_window(target_hwnd: Option<isize>) -> Result<(Vec<u8>, u32, u32)> {
        let hwnd = target_hwnd
            .map(|h| HWND(h as *mut core::ffi::c_void))
            .unwrap_or_else(|| unsafe { GetForegroundWindow() });

        let mut rect = RECT::default();
        let has_rect = if !hwnd.is_invalid() {
            unsafe { GetWindowRect(hwnd, &mut rect) }.is_ok()
        } else {
            false
        };

        let (left, top, width, height) = if has_rect && (rect.right > rect.left) && (rect.bottom > rect.top) {
            (
                rect.left,
                rect.top,
                (rect.right - rect.left) as u32,
                (rect.bottom - rect.top) as u32,
            )
        } else {
            let w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
            let h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
            let w = if w <= 0 { 1920 } else { w as u32 };
            let h = if h <= 0 { 1080 } else { h as u32 };
            (0, 0, w, h)
        };

        unsafe {
            let hdc_screen = GetDC(None);
            if hdc_screen.is_invalid() {
                anyhow::bail!("GetDC(None) failed to acquire screen device context");
            }

            let hdc_mem = CreateCompatibleDC(Some(hdc_screen));
            if hdc_mem.is_invalid() {
                ReleaseDC(None, hdc_screen);
                anyhow::bail!("CreateCompatibleDC failed");
            }

            let hbitmap = CreateCompatibleBitmap(hdc_screen, width as i32, height as i32);
            if hbitmap.is_invalid() {
                let _ = DeleteDC(hdc_mem);
                ReleaseDC(None, hdc_screen);
                anyhow::bail!("CreateCompatibleBitmap failed");
            }

            let old_bmp = SelectObject(hdc_mem, hbitmap.into());

            let blt_res = BitBlt(
                hdc_mem,
                0,
                0,
                width as i32,
                height as i32,
                Some(hdc_screen),
                left,
                top,
                SRCCOPY,
            );

            if let Err(e) = blt_res {
                let _ = SelectObject(hdc_mem, old_bmp);
                let _ = DeleteObject(hbitmap.into());
                let _ = DeleteDC(hdc_mem);
                ReleaseDC(None, hdc_screen);
                anyhow::bail!("BitBlt failed: {e}");
            }

            let mut bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width as i32,
                    biHeight: -(height as i32), // negative for top-down DIB
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    biSizeImage: width * height * 4,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: [windows::Win32::Graphics::Gdi::RGBQUAD::default()],
            };

            let mut bgra_pixels = vec![0u8; (width * height * 4) as usize];
            let dib_res = GetDIBits(
                hdc_screen,
                hbitmap,
                0,
                height,
                Some(bgra_pixels.as_mut_ptr() as *mut _),
                &mut bmi,
                DIB_RGB_COLORS,
            );

            // Clean up handles promptly
            let _ = SelectObject(hdc_mem, old_bmp);
            let _ = DeleteObject(hbitmap.into());
            let _ = DeleteDC(hdc_mem);
            ReleaseDC(None, hdc_screen);

            if dib_res == 0 {
                anyhow::bail!("GetDIBits failed to extract image buffer");
            }

            // GDI 32-bit DIB provides BGRA order; convert to RGBA in-place
            for chunk in bgra_pixels.chunks_exact_mut(4) {
                chunk.swap(0, 2); // Swap B and R
            }

            let png_bytes = encode_rgba_to_png(&bgra_pixels, width, height)?;
            Ok((png_bytes, width, height))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_encoder_generates_valid_png_header_and_data() {
        let width = 2;
        let height = 2;
        // 2x2 image with 4 RGBA pixels: Red, Green, Blue, White
        let pixels: Vec<u8> = vec![
            255, 0, 0, 255,   // red
            0, 255, 0, 255,   // green
            0, 0, 255, 255,   // blue
            255, 255, 255, 255, // white
        ];

        let png = encode_rgba_to_png(&pixels, width, height).unwrap();
        assert!(!png.is_empty());
        // Verify standard PNG magic signature \x89PNG\r\n\x1a\n
        assert_eq!(&png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }
}
