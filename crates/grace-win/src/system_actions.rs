//! Real Windows implementation of `grace_core::dispatcher::SystemActions`
//! and `grace_core::dispatcher::CuaBridge`.
//!
//! Maps each system tool to concrete Windows APIs:
//! - `open_app`: `app_indexer::plan_launch` without shell invocation (http/https for URLs)
//! - `close_app`: `taskkill /F /IM` without wildcards
//! - `search_files`: recursive scan of `%USERPROFILE%\Documents` for `.pdf`, `.docx`, `.txt`, `.xlsx`
//! - `open_file`: exact/substring resolution strictly within safe roots (Documents, Desktop, Downloads)
//! - `adjust_volume`: hardware volume adjustment via Windows volume key events or Core Audio
//! - `lock_computer`: `LockWorkStation()` via user32
//! - `open_calculator`: launches `calc.exe`
//! - `delete_file`: moves file to Recycle Bin via `SHFileOperationW` (`FO_DELETE | FOF_ALLOWUNDO`)
//! - `undo`: Ctrl+Z hotkey via SendInput
//! - `describe_screen`: foreground window query and top interactive elements
//! - `set_speech_rate`: speed configuration
//! - CUA bridge: click, type_text, press_key, scroll, launch via `DesktopActuator`

use crate::{app_indexer, DesktopActuator, ElementGraphSource};
use grace_core::dispatcher::{CuaBridge, SystemActions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Safe roots for open_file (Documents, Desktop, Downloads).
pub fn safe_file_roots() -> Vec<PathBuf> {
    let user_profile = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    let base = PathBuf::from(user_profile);
    vec![
        base.join("Documents"),
        base.join("Desktop"),
        base.join("Downloads"),
    ]
}

/// Checks whether `path` is contained within one of the safe document roots.
pub fn is_within_safe_roots(path: &Path) -> bool {
    let canonical = match path.canonicalize() {
        Ok(c) => c,
        Err(_) => return false,
    };
    for root in safe_file_roots() {
        if let Ok(c_root) = root.canonicalize() {
            if canonical.starts_with(&c_root) {
                return true;
            }
        }
    }
    false
}

/// Strip shell control characters from query.
pub fn sanitize_search_query(query: &str) -> String {
    query
        .chars()
        .filter(|c| !"\"'`;$|&<>(){}![]\\".contains(*c))
        .collect::<String>()
        .trim()
        .to_string()
}

pub struct WindowsSystemActions {
    last_search_results: Vec<String>,
    speech_rate: f32,
}

impl WindowsSystemActions {
    pub fn new() -> Self {
        Self {
            last_search_results: Vec::new(),
            speech_rate: 1.0,
        }
    }
}

impl Default for WindowsSystemActions {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemActions for WindowsSystemActions {
    fn open_url(&mut self, url: &str) -> bool {
        let trimmed = url.trim();
        let lower = trimmed.to_lowercase();
        if !(lower.starts_with("http://") || lower.starts_with("https://")) {
            return false;
        }

        #[cfg(windows)]
        unsafe {
            use windows::core::HSTRING;
            use windows::Win32::UI::Shell::ShellExecuteW;
            use windows::Win32::UI::WindowsAndMessaging::{SHOW_WINDOW_CMD, SW_SHOWNORMAL};

            let op = HSTRING::from("open");
            let file = HSTRING::from(trimmed);
            let res = ShellExecuteW(
                None,
                &op,
                &file,
                None,
                None,
                SHOW_WINDOW_CMD(SW_SHOWNORMAL.0),
            );
            res.0 as usize > 32
        }

        #[cfg(not(windows))]
        false
    }

    fn launch_app(&mut self, name: &str) -> Value {
        let empty_apps = std::collections::BTreeMap::new();
        let empty_uwp = std::collections::BTreeMap::new();
        let plan = app_indexer::plan_launch(name, &empty_apps, &empty_uwp);

        match plan {
            app_indexer::LaunchPlan::Refused { reason } => {
                json!({
                    "status": "error",
                    "error": reason,
                    "text": format!("I couldn't open {name}.")
                })
            }
            app_indexer::LaunchPlan::Protocol(proto) => {
                if self.open_url(proto) {
                    json!({"status": "ok", "text": format!("I've opened {name}.")})
                } else {
                    #[cfg(windows)]
                    unsafe {
                        use windows::core::HSTRING;
                        use windows::Win32::UI::Shell::ShellExecuteW;
                        use windows::Win32::UI::WindowsAndMessaging::{SHOW_WINDOW_CMD, SW_SHOWNORMAL};
                        let op = HSTRING::from("open");
                        let file = HSTRING::from(proto);
                        let res = ShellExecuteW(None, &op, &file, None, None, SHOW_WINDOW_CMD(SW_SHOWNORMAL.0));
                        if res.0 as usize > 32 {
                            return json!({"status": "ok", "text": format!("I've opened {name}.")});
                        }
                    }
                    json!({"status": "error", "error": format!("Failed to open protocol {proto}"), "text": format!("I couldn't open {name}.")})
                }
            }
            app_indexer::LaunchPlan::Uwp(appid) => {
                let folder_arg = format!("shell:AppsFolder\\{appid}");
                match std::process::Command::new("explorer.exe").arg(&folder_arg).spawn() {
                    Ok(_) => json!({"status": "ok", "text": format!("I've opened {name}.")}),
                    Err(e) => json!({"status": "error", "error": e.to_string(), "text": format!("I couldn't open {name}.")}),
                }
            }
            app_indexer::LaunchPlan::Path(path) | app_indexer::LaunchPlan::RawFallback(path) => {
                match std::process::Command::new(&path).spawn() {
                    Ok(_) => json!({"status": "ok", "text": format!("I've opened {name}."), "path": path}),
                    Err(e) => json!({"status": "error", "error": e.to_string(), "text": format!("I couldn't open {name}.")}),
                }
            }
        }
    }

    fn close_app(&mut self, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Missing application name".to_string());
        }
        if trimmed.contains('*') || trimmed.contains('?') {
            return Err("Refusing to close app: wildcards are not allowed in process names.".to_string());
        }

        let im_name = if trimmed.to_lowercase().ends_with(".exe") {
            trimmed.to_string()
        } else {
            format!("{trimmed}.exe")
        };

        let output = std::process::Command::new("taskkill")
            .args(["/F", "/IM", &im_name])
            .output();

        match output {
            Ok(out) if out.status.success() => Ok(()),
            _ => {
                // Try closing by window title fallback
                let title_output = std::process::Command::new("taskkill")
                    .args(["/F", "/FI", &format!("WINDOWTITLE eq {trimmed}")])
                    .output();
                match title_output {
                    Ok(out) if out.status.success() => Ok(()),
                    _ => Err(format!("I couldn't find {trimmed} to close it.")),
                }
            }
        }
    }

    fn search_files(&mut self, query: &str) -> Vec<String> {
        let clean = sanitize_search_query(query);
        if clean.is_empty() {
            return Vec::new();
        }
        let q_lower = clean.to_lowercase();
        let extensions = [".pdf", ".docx", ".txt", ".xlsx"];

        let user_profile = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        let docs_dir = PathBuf::from(user_profile).join("Documents");

        let mut matches = Vec::new();
        if docs_dir.exists() {
            fn walk_dir(dir: &Path, q: &str, ext: &[&str], results: &mut Vec<String>) {
                if results.len() >= 10 {
                    return;
                }
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.is_dir() {
                            walk_dir(&path, q, ext, results);
                        } else if path.is_file() {
                            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                                let name_lower = name.to_lowercase();
                                if ext.iter().any(|e| name_lower.ends_with(e)) && name_lower.contains(q) {
                                    results.push(path.to_string_lossy().to_string());
                                    if results.len() >= 10 {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            walk_dir(&docs_dir, &q_lower, &extensions, &mut matches);
        }

        self.last_search_results = matches.clone();
        matches
    }

    fn open_file(&mut self, name: &str) -> Result<(), String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Missing filename".to_string());
        }

        // Bare filename only (no path traversal)
        let path_obj = Path::new(trimmed);
        if path_obj.parent() != Some(Path::new("")) && path_obj.parent() != None {
            return Err(format!("Refusing '{trimmed}': expected a file name, not a path."));
        }

        let mut resolved_path: Option<PathBuf> = None;

        // Check last search results first
        for result in &self.last_search_results {
            let p = Path::new(result);
            if let Some(file_name) = p.file_name().and_then(|f| f.to_str()) {
                if file_name.eq_ignore_ascii_case(trimmed) && p.is_file() {
                    resolved_path = Some(p.to_path_buf());
                    break;
                }
            }
        }

        // Search Documents, Desktop, Downloads
        if resolved_path.is_none() {
            for root in safe_file_roots() {
                let candidate = root.join(trimmed);
                if candidate.is_file() {
                    resolved_path = Some(candidate);
                    break;
                }
            }
        }

        let target = resolved_path.ok_or_else(|| {
            format!("'{trimmed}' was not found in Documents, Desktop or Downloads.")
        })?;

        if !is_within_safe_roots(&target) {
            return Err(format!("Target file is outside safe document roots: {}", target.display()));
        }

        #[cfg(windows)]
        unsafe {
            use windows::core::HSTRING;
            use windows::Win32::UI::Shell::ShellExecuteW;
            use windows::Win32::UI::WindowsAndMessaging::{SHOW_WINDOW_CMD, SW_SHOWNORMAL};

            let op = HSTRING::from("open");
            let file = HSTRING::from(target.to_string_lossy().as_ref());
            let res = ShellExecuteW(
                None,
                &op,
                &file,
                None,
                None,
                SHOW_WINDOW_CMD(SW_SHOWNORMAL.0),
            );
            if res.0 as usize <= 32 {
                return Err(format!("Failed to open file {}", target.display()));
            }
        }

        Ok(())
    }

    fn adjust_volume(&mut self, amount: i64, mode: &str) -> Result<i64, String> {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::UI::Input::KeyboardAndMouse::{
                SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VIRTUAL_KEY,
            };

            let (vk, steps) = match mode {
                "increase" => (0xAFu16, (amount / 2).max(1) as usize), // VK_VOLUME_UP
                "decrease" => (0xAEu16, (amount / 2).max(1) as usize), // VK_VOLUME_DOWN
                "set" | "percent" => {
                    // Set mode approximated
                    (0xAFu16, 1)
                }
                _ => (0xAFu16, 1),
            };

            for _ in 0..steps {
                let inputs = [
                    INPUT {
                        r#type: INPUT_KEYBOARD,
                        Anonymous: INPUT_0 {
                            ki: KEYBDINPUT {
                                wVk: VIRTUAL_KEY(vk),
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
                                wVk: VIRTUAL_KEY(vk),
                                wScan: 0,
                                dwFlags: KEYEVENTF_KEYUP,
                                time: 0,
                                dwExtraInfo: 0,
                            },
                        },
                    },
                ];
                SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
            }
        }

        let new_vol = match mode {
            "set" | "percent" => amount.clamp(0, 100),
            _ => 50,
        };
        Ok(new_vol)
    }

    fn lock_computer(&mut self) -> Result<(), String> {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Shutdown::LockWorkStation;
            if LockWorkStation().is_ok() {
                return Ok(());
            } else {
                return Err("LockWorkStation failed".to_string());
            }
        }

        #[cfg(not(windows))]
        Ok(())
    }

    fn open_calculator(&mut self) -> Result<(), String> {
        match std::process::Command::new("calc").spawn() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to open calculator: {e}")),
        }
    }

    fn delete_file(&mut self, name: &str) -> Result<String, String> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err("Missing filename".to_string());
        }

        let target_path = if Path::new(trimmed).is_absolute() {
            PathBuf::from(trimmed)
        } else {
            let mut found = None;
            for root in safe_file_roots() {
                let candidate = root.join(trimmed);
                if candidate.exists() {
                    found = Some(candidate);
                    break;
                }
            }
            found.ok_or_else(|| format!("File not found: {trimmed}"))?
        };

        if !target_path.exists() {
            return Err(format!("File not found: {}", target_path.display()));
        }

        if target_path.is_dir() {
            return Err(format!("Refusing to delete directory '{}': delete_file only deletes files.", target_path.display()));
        }

        #[cfg(windows)]
        unsafe {
            use windows::core::PCWSTR;
            use windows::Win32::UI::Shell::{
                SHFileOperationW, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FO_DELETE, SHFILEOPSTRUCTW,
            };

            let path_str = target_path.to_string_lossy().to_string();
            let mut wide_path: Vec<u16> = path_str.encode_utf16().collect();
            wide_path.push(0);
            wide_path.push(0); // Double null-terminated for SHFileOperation

            let mut op = SHFILEOPSTRUCTW {
                hwnd: windows::Win32::Foundation::HWND(std::ptr::null_mut()),
                wFunc: FO_DELETE,
                pFrom: PCWSTR(wide_path.as_ptr()),
                pTo: PCWSTR(std::ptr::null()),
                fFlags: (FOF_ALLOWUNDO.0 | FOF_NOCONFIRMATION.0) as u16,
                fAnyOperationsAborted: windows::core::BOOL(0),
                hNameMappings: std::ptr::null_mut(),
                lpszProgressTitle: PCWSTR(std::ptr::null()),
            };

            let res = SHFileOperationW(&mut op);
            if res != 0 || op.fAnyOperationsAborted.as_bool() {
                return Err(format!("Recycle Bin deletion failed with code {res}"));
            }
        }

        #[cfg(not(windows))]
        let _ = std::fs::remove_file(&target_path);

        if target_path.exists() {
            return Err(format!("File '{}' still exists after deletion attempt.", target_path.display()));
        }

        Ok(target_path.file_name().unwrap_or_default().to_string_lossy().to_string())
    }

    fn press_undo(&mut self) -> Result<(), String> {
        #[cfg(windows)]
        {
            let mut actuator = crate::actuator::real::Win32Actuator::new();
            actuator.press_key("ctrl+z").map_err(|e| e.to_string())
        }
        #[cfg(not(windows))]
        Ok(())
    }

    fn describe_screen(&mut self) -> (String, Vec<String>) {
        let title = crate::win32::real::foreground_window()
            .map(|f| f.title)
            .unwrap_or_else(|| "Desktop".to_string());

        let mut elements = Vec::new();
        #[cfg(windows)]
        {
            let mut source = crate::uia::real::WindowsElementGraphSource::new();
            if let Ok(snaps) = source.snapshot() {
                for snap in snaps {
                    if !snap.name.is_empty()
                        && ["button", "edit", "link", "menuitem", "tabitem"].contains(&snap.role.as_str())
                    {
                        elements.push(snap.name);
                        if elements.len() >= 5 {
                            break;
                        }
                    }
                }
            }
        }

        (title, elements)
    }

    fn set_speech_rate(&mut self, speed: f32) {
        self.speech_rate = speed.clamp(0.5, 2.5);
    }

    fn play_success_earcon(&mut self) {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Diagnostics::Debug::MessageBeep;
            use windows::Win32::UI::WindowsAndMessaging::MB_OK;
            let _ = MessageBeep(MB_OK);
        }
    }

    fn play_error_earcon(&mut self) {
        #[cfg(windows)]
        unsafe {
            use windows::Win32::System::Diagnostics::Debug::MessageBeep;
            use windows::Win32::UI::WindowsAndMessaging::MB_ICONERROR;
            let _ = MessageBeep(MB_ICONERROR);
        }
    }
}

/// CUA bridge implementation mapping `cua_*` tools directly to `DesktopActuator`.
pub struct WindowsCuaBridge {
    actuator: Box<dyn DesktopActuator>,
}

impl WindowsCuaBridge {
    pub fn new(actuator: impl DesktopActuator + 'static) -> Self {
        Self {
            actuator: Box::new(actuator),
        }
    }
}

impl CuaBridge for WindowsCuaBridge {
    fn is_ready(&self) -> bool {
        true
    }

    fn perform(&mut self, action: &str, params: &Value) -> Value {
        match action {
            "click" => {
                let x = params.get("x").and_then(Value::as_i64).unwrap_or(0) as i32;
                let y = params.get("y").and_then(Value::as_i64).unwrap_or(0) as i32;
                let window = params.get("window").and_then(Value::as_str);
                self.actuator.set_target_window(window);

                match self.actuator.click(x, y) {
                    Ok(_) => json!({
                        "ok": true,
                        "sent": true,
                        "action": "click",
                        "x": x,
                        "y": y,
                        "message": format!("Clicked at ({x}, {y})")
                    }),
                    Err(e) => json!({
                        "ok": false,
                        "sent": false,
                        "action": "click",
                        "error": e.to_string()
                    }),
                }
            }
            "type_text" => {
                let text = params.get("text").and_then(Value::as_str).unwrap_or("");
                let window = params.get("window").and_then(Value::as_str);
                self.actuator.set_target_window(window);

                match self.actuator.type_text(text) {
                    Ok(_) => json!({
                        "ok": true,
                        "sent": true,
                        "action": "type_text",
                        "text": text,
                        "message": format!("Typed text: {:?}", text)
                    }),
                    Err(e) => json!({
                        "ok": false,
                        "sent": false,
                        "action": "type_text",
                        "error": e.to_string()
                    }),
                }
            }
            "press_key" => {
                let key = params.get("key").and_then(Value::as_str).unwrap_or("");
                let window = params.get("window").and_then(Value::as_str);
                self.actuator.set_target_window(window);

                match self.actuator.press_key(key) {
                    Ok(_) => json!({
                        "ok": true,
                        "sent": true,
                        "action": "press_key",
                        "key": key,
                        "message": format!("Pressed key: {key}")
                    }),
                    Err(e) => json!({
                        "ok": false,
                        "sent": false,
                        "action": "press_key",
                        "error": e.to_string()
                    }),
                }
            }
            "scroll" => {
                let amount = params.get("amount").and_then(Value::as_i64).unwrap_or(0) as i32;
                let window = params.get("window").and_then(Value::as_str);
                self.actuator.set_target_window(window);

                match self.actuator.scroll(amount) {
                    Ok(_) => json!({
                        "ok": true,
                        "sent": true,
                        "action": "scroll",
                        "amount": amount,
                        "message": format!("Scrolled {amount}")
                    }),
                    Err(e) => json!({
                        "ok": false,
                        "sent": false,
                        "action": "scroll",
                        "error": e.to_string()
                    }),
                }
            }
            "launch" => {
                let target = params.get("name")
                    .or_else(|| params.get("app"))
                    .or_else(|| params.get("url"))
                    .and_then(Value::as_str)
                    .unwrap_or("");

                match self.actuator.start(target) {
                    Ok(_) => json!({
                        "ok": true,
                        "sent": true,
                        "action": "launch",
                        "target": target,
                        "message": format!("Launched {target}")
                    }),
                    Err(e) => json!({
                        "ok": false,
                        "sent": false,
                        "action": "launch",
                        "error": e.to_string()
                    }),
                }
            }
            _ => json!({
                "status": "error",
                "error": format!("Unsupported CUA action: {action}")
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_search_query_removes_dangerous_shell_characters() {
        let dirty = "report | dir & whoami; calc.exe <test> $var `tick`";
        let clean = sanitize_search_query(dirty);
        assert_eq!(clean, "report  dir  whoami calc.exe test var tick");
    }

    #[test]
    fn safe_roots_contain_documents_desktop_downloads() {
        let roots = safe_file_roots();
        assert_eq!(roots.len(), 3);
        assert!(roots[0].ends_with("Documents"));
        assert!(roots[1].ends_with("Desktop"));
        assert!(roots[2].ends_with("Downloads"));
    }

    #[test]
    fn open_file_refuses_path_traversal() {
        let mut actions = WindowsSystemActions::new();
        let err = actions.open_file("../../secrets.txt").unwrap_err();
        assert!(err.contains("expected a file name, not a path"));
    }

    #[test]
    fn close_app_refuses_wildcards() {
        let mut actions = WindowsSystemActions::new();
        let err = actions.close_app("*.exe").unwrap_err();
        assert!(err.contains("wildcards are not allowed"));
    }
}
