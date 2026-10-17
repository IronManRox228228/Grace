//! Native Win32 desktop actuator implementing `DesktopActuator`.
//!
//! Features:
//! - SendInput mouse clicks with DPI-aware scaling and hover-first settling.
//! - SendInput typing supporting full Unicode (including non-ASCII, emojis, foreign scripts)
//!   via `KEYEVENTF_UNICODE`.
//! - Key presses with hotkey combinations (Ctrl, Alt, Shift) and virtual key mapping.
//! - Mouse wheel scrolling via `MOUSEEVENTF_WHEEL`.
//! - App launches via `app_indexer::plan_launch` without shell invocation (http/https only for web).
//! - Foreground window re-check before every physical action (R11/R12/R22 semantics).

use crate::{app_indexer, DesktopActuator};
use std::time::Duration;

/// Delays tuned for focus acquisition and hover state.
pub const HOVER_SETTLE_MS: u64 = 60;

/// Map key string to Win32 Virtual Key code (u16).
pub fn map_key_name(key: &str) -> Option<u16> {
    let k = key.trim().to_lowercase();
    match k.as_str() {
        "enter" | "return" => Some(0x0D),
        "tab" => Some(0x09),
        "backspace" | "back" => Some(0x08),
        "delete" | "del" => Some(0x2E),
        "escape" | "esc" => Some(0x1B),
        "space" | "spacebar" => Some(0x20),
        "up" => Some(0x26),
        "down" => Some(0x28),
        "left" => Some(0x25),
        "right" => Some(0x27),
        "home" => Some(0x24),
        "end" => Some(0x23),
        "pageup" | "pgup" => Some(0x21),
        "pagedown" | "pgdn" => Some(0x22),
        "insert" | "ins" => Some(0x2D),
        "ctrl" | "control" => Some(0x11),
        "alt" | "menu" => Some(0x12),
        "shift" => Some(0x10),
        "win" | "windows" | "super" => Some(0x5B),
        "f1" => Some(0x70),
        "f2" => Some(0x71),
        "f3" => Some(0x72),
        "f4" => Some(0x73),
        "f5" => Some(0x74),
        "f6" => Some(0x75),
        "f7" => Some(0x76),
        "f8" => Some(0x77),
        "f9" => Some(0x78),
        "f10" => Some(0x79),
        "f11" => Some(0x7A),
        "f12" => Some(0x7B),
        s if s.len() == 1 => {
            let ch = s.chars().next().unwrap();
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_uppercase() as u16)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse a hotkey string (e.g. "ctrl+c", "alt+f4", "ctrl+shift+esc") into
/// modifier virtual keys and the main virtual key.
pub fn parse_hotkey(hotkey: &str) -> Option<(Vec<u16>, u16)> {
    let parts: Vec<&str> = hotkey.split('+').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        return None;
    }

    let mut modifiers = Vec::new();
    let main_part = parts.last().unwrap();

    for &mod_part in &parts[..parts.len() - 1] {
        let vk = match mod_part.to_lowercase().as_str() {
            "ctrl" | "control" => 0x11,
            "alt" | "menu" => 0x12,
            "shift" => 0x10,
            "win" | "windows" | "super" => 0x5B,
            _ => return None,
        };
        modifiers.push(vk);
    }

    let main_vk = map_key_name(main_part)?;
    Some((modifiers, main_vk))
}

/// Pure helper: checks foreground window focus against an expected target window (R11/R12/R22).
/// Returns Ok(()) if the focus is valid, or Err with refusal reason.
pub fn check_window_focus(wanted: Option<&str>, actual: Option<&str>, action: &str) -> Result<(), String> {
    let wanted = wanted.unwrap_or("").trim();
    if wanted.is_empty() {
        return Ok(());
    }

    let actual = actual.unwrap_or("").trim();
    if actual.is_empty() {
        return Ok(());
    }

    let a = wanted.to_lowercase();
    let b = actual.to_lowercase();

    if a.contains(&b) || b.contains(&a) {
        Ok(())
    } else {
        Err(format!(
            "You aimed at '{wanted}' but '{actual}' is in front. Activate the window you meant first before {action}."
        ))
    }
}

/// Validates whether a URL is permitted to be launched automatically (http/https only, no shell/UNC/file).
pub fn validate_url_for_launch(url: &str) -> Result<String, String> {
    let trimmed = url.trim();
    let lower = trimmed.to_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err(format!("Refusing to open '{url}': only http/https links are allowed."));
    }
    Ok(trimmed.to_string())
}

#[cfg(windows)]
pub use real::Win32Actuator;

#[cfg(windows)]
pub mod real {
    use super::*;
    use windows::core::HSTRING;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT,
        KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN,
        MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_WHEEL,
        MOUSEINPUT, VIRTUAL_KEY,
    };
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::{
        GetSystemMetrics, SetCursorPos, SHOW_WINDOW_CMD,
        SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
        SW_SHOWNORMAL,
    };

    pub struct Win32Actuator {
        expected_window: Option<String>,
    }

    impl Win32Actuator {
        pub fn new() -> Self {
            Self { expected_window: None }
        }

        fn recheck_foreground(&self, action: &str) -> anyhow::Result<()> {
            let actual = crate::win32::real::foreground_window().map(|f| f.title);
            if let Err(e) = check_window_focus(self.expected_window.as_deref(), actual.as_deref(), action) {
                anyhow::bail!("{e}");
            }
            Ok(())
        }
    }

    impl Default for Win32Actuator {
        fn default() -> Self {
            Self::new()
        }
    }

    impl DesktopActuator for Win32Actuator {
        fn set_target_window(&mut self, title: Option<&str>) {
            self.expected_window = title.map(|s| s.to_string());
        }

        fn click(&mut self, x: i32, y: i32) -> anyhow::Result<()> {
            self.recheck_foreground("click")?;
            crate::dpi::real::ensure_dpi_aware();

            let dpi_scale = crate::dpi::real::get_dpi_scale(None);
            let phys_x = (x as f64 * dpi_scale).round() as i32;
            let phys_y = (y as f64 * dpi_scale).round() as i32;

            unsafe {
                let vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
                let vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
                let vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
                let vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);

                let vw = if vw <= 0 { 1920 } else { vw };
                let vh = if vh <= 0 { 1080 } else { vh };

                let norm_x = ((phys_x - vx) as i64 * 65535 / (vw - 1).max(1) as i64) as i32;
                let norm_y = ((phys_y - vy) as i64 * 65535 / (vh - 1).max(1) as i64) as i32;

                let _ = SetCursorPos(phys_x, phys_y);
                std::thread::sleep(Duration::from_millis(HOVER_SETTLE_MS));

                let flags_base = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;

                let inputs = [
                    INPUT {
                        r#type: INPUT_MOUSE,
                        Anonymous: INPUT_0 {
                            mi: MOUSEINPUT {
                                dx: norm_x,
                                dy: norm_y,
                                mouseData: 0,
                                dwFlags: flags_base | MOUSEEVENTF_MOVE,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                    INPUT {
                        r#type: INPUT_MOUSE,
                        Anonymous: INPUT_0 {
                            mi: MOUSEINPUT {
                                dx: norm_x,
                                dy: norm_y,
                                mouseData: 0,
                                dwFlags: flags_base | MOUSEEVENTF_LEFTDOWN,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                    INPUT {
                        r#type: INPUT_MOUSE,
                        Anonymous: INPUT_0 {
                            mi: MOUSEINPUT {
                                dx: norm_x,
                                dy: norm_y,
                                mouseData: 0,
                                dwFlags: flags_base | MOUSEEVENTF_LEFTUP,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                ];

                let sent = SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
                if sent != 3 {
                    anyhow::bail!("SendInput click failed: sent {sent} of 3 events");
                }
            }

            Ok(())
        }

        fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
            self.recheck_foreground("type_text")?;

            for ch in text.encode_utf16() {
                unsafe {
                    let inputs = [
                        INPUT {
                            r#type: INPUT_KEYBOARD,
                            Anonymous: INPUT_0 {
                                ki: KEYBDINPUT {
                                    wVk: VIRTUAL_KEY(0),
                                    wScan: ch,
                                    dwFlags: KEYEVENTF_UNICODE,
                                    time: 0,
                                    dwExtraInfo: 0,
                                },
                            },
                        },
                        INPUT {
                            r#type: INPUT_KEYBOARD,
                            Anonymous: INPUT_0 {
                                ki: KEYBDINPUT {
                                    wVk: VIRTUAL_KEY(0),
                                    wScan: ch,
                                    dwFlags: KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                                    time: 0,
                                    dwExtraInfo: 0,
                                },
                            },
                        },
                    ];

                    let sent = SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
                    if sent != 2 {
                        anyhow::bail!("SendInput unicode type failed for char code {ch}");
                    }
                }
            }

            Ok(())
        }

        fn press_key(&mut self, key: &str) -> anyhow::Result<()> {
            self.recheck_foreground("press_key")?;

            let (modifiers, main_vk) = parse_hotkey(key)
                .ok_or_else(|| anyhow::anyhow!("Unrecognised key or hotkey: {key}"))?;

            unsafe {
                // 1. Press modifiers down
                for &m in &modifiers {
                    let input = [INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(m),
                                wScan: 0,
                                dwFlags: windows::Win32::UI::Input::KeyboardAndMouse::KEYBD_EVENT_FLAGS(0),
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    }];
                    SendInput(&input, std::mem::size_of::<INPUT>() as i32);
                }

                // 2. Press main key down and up
                let main_inputs = [
                    INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(main_vk),
                                wScan: 0,
                                dwFlags: windows::Win32::UI::Input::KeyboardAndMouse::KEYBD_EVENT_FLAGS(0),
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                    INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(main_vk),
                                wScan: 0,
                                dwFlags: KEYEVENTF_KEYUP,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                ];
                SendInput(&main_inputs, std::mem::size_of::<INPUT>() as i32);

                // 3. Release modifiers in reverse
                for &m in modifiers.iter().rev() {
                    let input = [INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(m),
                                wScan: 0,
                                dwFlags: KEYEVENTF_KEYUP,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    }];
                    SendInput(&input, std::mem::size_of::<INPUT>() as i32);
                }
            }

            Ok(())
        }

        fn scroll(&mut self, amount: i32) -> anyhow::Result<()> {
            self.recheck_foreground("scroll")?;

            let wheel_delta = amount * 120;
            unsafe {
                let input = [INPUT {
                    r#type: INPUT_MOUSE,
                    Anonymous: INPUT_0 {
                        mi: MOUSEINPUT {
                            dx: 0,
                            dy: 0,
                            mouseData: wheel_delta as u32,
                            dwFlags: MOUSEEVENTF_WHEEL,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                }];
                let sent = SendInput(&input, std::mem::size_of::<INPUT>() as i32);
                if sent != 1 {
                    anyhow::bail!("SendInput scroll failed");
                }
            }

            Ok(())
        }

        fn start(&mut self, target: &str) -> anyhow::Result<()> {
            let trimmed = target.trim();
            if trimmed.is_empty() {
                anyhow::bail!("Cannot start empty target");
            }

            // Direct web URL launch
            if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
                let validated = validate_url_for_launch(trimmed)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                unsafe {
                    let op = HSTRING::from("open");
                    let file = HSTRING::from(&validated);
                    let res = ShellExecuteW(
                        None,
                        &op,
                        &file,
                        None,
                        None,
                        SHOW_WINDOW_CMD(SW_SHOWNORMAL.0),
                    );
                    if res.0 as usize <= 32 {
                        anyhow::bail!("ShellExecuteW failed for URL: {validated}");
                    }
                }
                return Ok(());
            }

            // Application or protocol launch
            let empty_apps = std::collections::BTreeMap::new();
            let empty_uwp = std::collections::BTreeMap::new();
            let plan = app_indexer::plan_launch(trimmed, &empty_apps, &empty_uwp);

            match plan {
                app_indexer::LaunchPlan::Refused { reason } => {
                    anyhow::bail!("Launch refused: {reason}");
                }
                app_indexer::LaunchPlan::Protocol(proto) => {
                    unsafe {
                        let op = HSTRING::from("open");
                        let file = HSTRING::from(proto);
                        let res = ShellExecuteW(
                            None,
                            &op,
                            &file,
                            None,
                            None,
                            SHOW_WINDOW_CMD(SW_SHOWNORMAL.0),
                        );
                        if res.0 as usize <= 32 {
                            anyhow::bail!("ShellExecuteW failed for protocol: {proto}");
                        }
                    }
                    Ok(())
                }
                app_indexer::LaunchPlan::Uwp(appid) => {
                    let folder_arg = format!("shell:AppsFolder\\{appid}");
                    std::process::Command::new("explorer.exe")
                        .arg(&folder_arg)
                        .spawn()
                        .map_err(|e| anyhow::anyhow!("Failed to launch UWP app {appid}: {e}"))?;
                    Ok(())
                }
                app_indexer::LaunchPlan::Path(path) | app_indexer::LaunchPlan::RawFallback(path) => {
                    std::process::Command::new(&path)
                        .spawn()
                        .map_err(|e| anyhow::anyhow!("Failed to spawn {path}: {e}"))?;
                    Ok(())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_name_mapping_covers_common_keys() {
        assert_eq!(map_key_name("enter"), Some(0x0D));
        assert_eq!(map_key_name("tab"), Some(0x09));
        assert_eq!(map_key_name("backspace"), Some(0x08));
        assert_eq!(map_key_name("delete"), Some(0x2E));
        assert_eq!(map_key_name("escape"), Some(0x1B));
        assert_eq!(map_key_name("space"), Some(0x20));
        assert_eq!(map_key_name("f5"), Some(0x74));
        assert_eq!(map_key_name("a"), Some(0x41));
        assert_eq!(map_key_name("z"), Some(0x5A));
        assert_eq!(map_key_name("1"), Some(0x31));
        assert_eq!(map_key_name("unknown_key_xyz"), None);
    }

    #[test]
    fn parse_hotkey_extracts_modifiers_and_main_key() {
        let (mods, main) = parse_hotkey("ctrl+c").unwrap();
        assert_eq!(mods, vec![0x11]);
        assert_eq!(main, 0x43); // 'C'

        let (mods, main) = parse_hotkey("ctrl+shift+esc").unwrap();
        assert_eq!(mods, vec![0x11, 0x10]);
        assert_eq!(main, 0x1B); // ESC

        let (mods, main) = parse_hotkey("enter").unwrap();
        assert!(mods.is_empty());
        assert_eq!(main, 0x0D);
    }

    #[test]
    fn window_focus_check_permits_substring_matches_and_empty_targets() {
        // Empty target = anything goes
        assert!(check_window_focus(None, Some("Notepad"), "click").is_ok());
        assert!(check_window_focus(Some(""), Some("Notepad"), "click").is_ok());

        // Matching substring
        assert!(check_window_focus(Some("Notepad"), Some("Untitled - Notepad"), "click").is_ok());
        assert!(check_window_focus(Some("Google Chrome"), Some("Google Chrome"), "click").is_ok());

        // Mismatched focus refused with exact R11/R12/R22 semantics
        let err = check_window_focus(Some("Calculator"), Some("Document - Word"), "click").unwrap_err();
        assert!(err.contains("You aimed at 'Calculator' but 'Document - Word' is in front"));
    }

    #[test]
    fn validate_url_for_launch_strictly_requires_http_or_https() {
        assert!(validate_url_for_launch("https://google.com").is_ok());
        assert!(validate_url_for_launch("http://localhost:8080").is_ok());
        assert!(validate_url_for_launch("file:///C:/passwords.txt").is_err());
        assert!(validate_url_for_launch("\\\\server\\share\\run.exe").is_err());
        assert!(validate_url_for_launch("calc.exe").is_err());
    }
}
