//! Ported from `src/grace/agent/safety.py`.
//!
//! Enforces Project Grace's safety spec by intercepting destructive or
//! irreversible actions (file deletion, app closure, system locking) and
//! requiring explicit voice confirmation before execution. Keep this in sync
//! with `src/grace/agent/safety.py` deliberately by hand - see
//! PORT_STATUS.md for the process for re-checking that after any Python-side
//! change (this repo's hard constraints forbid editing the Python source, so
//! there is no way to derive this automatically from it).

use serde_json::Value;
use std::collections::BTreeSet;

/// Actions that ALWAYS require user confirmation before execution.
pub const CONFIRMATION_REQUIRED_TOOLS: [&str; 3] = ["delete_file", "close_app", "lock_computer"];

/// Hotkeys that close or discard work, in normalised form (see
/// `normalise_key`). Mirrors `SafetyGuard.CONFIRMATION_REQUIRED_KEYS`,
/// including the three added in the 2026-09-26 review
/// (`ctrl+f4`, `shift+delete`, `win+l`).
pub const CONFIRMATION_REQUIRED_KEYS: [&str; 7] = [
    "alt+f4",
    "ctrl+w",
    "ctrl+shift+w",
    "ctrl+q",
    "ctrl+f4",
    "shift+delete",
    "win+l",
];

/// X11-style keysyms and pyautogui's own native names, both mapped to their
/// plain canonical form.
fn key_alias(part: &str) -> &str {
    match part {
        "control_l" | "control_r" | "control" => "ctrl",
        "shift_l" | "shift_r" => "shift",
        "alt_l" | "alt_r" => "alt",
        "super_l" | "super_r" | "super" => "win",
        "ctrlleft" | "ctrlright" => "ctrl",
        "altleft" | "altright" => "alt",
        "shiftleft" | "shiftright" => "shift",
        "winleft" | "winright" => "win",
        other => other,
    }
}

/// Lower-case, de-alias, and sort modifiers so orderings compare equal
/// (`"Alt+F4"`, `"altleft+f4"` and `"Control_L+w"` all normalise the same
/// way a hand-typed `"ctrl+w"` would).
pub fn normalise_key(raw: Option<&str>) -> String {
    let raw = match raw {
        Some(r) if !r.is_empty() => r,
        _ => return String::new(),
    };
    let parts: Vec<String> = raw
        .split('+')
        .map(|piece| piece.trim().to_lowercase())
        .filter(|p| !p.is_empty())
        .map(|p| key_alias(&p).to_string())
        .collect();
    if parts.is_empty() {
        return String::new();
    }
    let (last, modifiers) = parts.split_last().unwrap();
    let mut modifiers: Vec<String> = modifiers.to_vec();
    modifiers.sort();
    modifiers.push(last.clone());
    modifiers.join("+")
}

/// Mirrors `(is_safe, confirmation_prompt)` from `SafetyGuard.evaluate`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    pub is_safe: bool,
    pub confirmation_prompt: Option<String>,
}

/// Evaluates proposed agent actions and checks if user voice confirmation is
/// required. `params` is the tool's argument object, as parsed JSON (mirrors
/// the Python `dict[str, Any]`).
pub fn evaluate(action: &str, params: &Value) -> Evaluation {
    if CONFIRMATION_REQUIRED_TOOLS.contains(&action) {
        let prompt = build_confirmation_prompt(action, params);
        return Evaluation {
            is_safe: false,
            confirmation_prompt: Some(prompt),
        };
    }

    if action == "cua_press_key" {
        let key = normalise_key(params.get("key").and_then(Value::as_str));
        if CONFIRMATION_REQUIRED_KEYS.contains(&key.as_str()) {
            return Evaluation {
                is_safe: false,
                confirmation_prompt: Some(
                    "Closing windows can cause loss of unsaved work. Should I proceed?".into(),
                ),
            };
        }
    }

    Evaluation {
        is_safe: true,
        confirmation_prompt: None,
    }
}

fn param_str<'a>(params: &'a Value, key: &str) -> Option<&'a str> {
    params.get(key).and_then(Value::as_str).filter(|s| !s.is_empty())
}

fn build_confirmation_prompt(action: &str, params: &Value) -> String {
    match action {
        "delete_file" => {
            let filename = param_str(params, "name")
                .or_else(|| param_str(params, "path"))
                .unwrap_or("this file");
            format!("Are you sure you want me to delete {filename}?")
        }
        "close_app" => {
            let app_name = param_str(params, "name").unwrap_or("this application");
            format!("Should I close {app_name}? Any unsaved changes may be lost.")
        }
        "lock_computer" => "Should I lock your computer now?".to_string(),
        other => format!("Confirm executing action '{other}'?"),
    }
}

/// Every tool the FAST_PATH is allowed to short-circuit through, minus the
/// ones `evaluate` intercepts, is exactly `CapabilityRouter.FAST_PATH_TOOLS -
/// SafetyGuard.CONFIRMATION_REQUIRED_TOOLS` - kept as a set here so a caller
/// (e.g. the harness or a future allowlist rewrite) can ask "does the fast
/// path currently bypass safety for this tool?" without hand-computing the
/// difference. See `contract/README.md`'s "the fast path does not consult
/// SafetyGuard" finding - that gap is preserved, not fixed, by this port.
pub fn confirmation_required_tools() -> BTreeSet<&'static str> {
    CONFIRMATION_REQUIRED_TOOLS.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn delete_file_always_needs_confirmation() {
        let eval = evaluate("delete_file", &json!({"name": "report.docx"}));
        assert!(!eval.is_safe);
        assert_eq!(
            eval.confirmation_prompt.unwrap(),
            "Are you sure you want me to delete report.docx?"
        );
    }

    #[test]
    fn delete_file_falls_back_to_path_then_generic() {
        let eval = evaluate("delete_file", &json!({"path": "C:/tmp/x.txt"}));
        assert_eq!(
            eval.confirmation_prompt.unwrap(),
            "Are you sure you want me to delete C:/tmp/x.txt?"
        );
        let eval = evaluate("delete_file", &json!({}));
        assert_eq!(
            eval.confirmation_prompt.unwrap(),
            "Are you sure you want me to delete this file?"
        );
    }

    #[test]
    fn close_app_and_lock_computer_need_confirmation() {
        assert!(!evaluate("close_app", &json!({"name": "Notepad"})).is_safe);
        assert!(!evaluate("lock_computer", &json!({})).is_safe);
    }

    #[test]
    fn ordinary_tools_are_safe() {
        let eval = evaluate("open_app", &json!({"name": "Notepad"}));
        assert!(eval.is_safe);
        assert!(eval.confirmation_prompt.is_none());
    }

    #[test]
    fn alt_f4_variants_all_require_confirmation() {
        // Only the trailing token is treated as "the key" (modifiers are the
        // ones sorted); this is a faithful port of that asymmetry, not an
        // improvement on it, so "F4+Alt" (last token "alt") normalises to
        // "f4+alt" and is deliberately NOT in this list.
        for raw in ["alt+f4", "Alt+F4", "altleft+f4", "Alt_L+F4"] {
            let eval = evaluate("cua_press_key", &json!({"key": raw}));
            assert!(!eval.is_safe, "{raw} should have required confirmation");
        }
    }

    #[test]
    fn the_2026_09_26_review_additions_require_confirmation() {
        for raw in ["ctrl+f4", "shift+delete", "win+l", "Control_L+F4", "Shift+Delete"] {
            let eval = evaluate("cua_press_key", &json!({"key": raw}));
            assert!(!eval.is_safe, "{raw} should have required confirmation");
        }
    }

    #[test]
    fn ordinary_keys_are_safe() {
        let eval = evaluate("cua_press_key", &json!({"key": "enter"}));
        assert!(eval.is_safe);
        let eval = evaluate("cua_press_key", &json!({"key": "ctrl+c"}));
        assert!(eval.is_safe);
    }

    #[test]
    fn normalise_key_sorts_modifiers_and_deduplicates_casing() {
        assert_eq!(normalise_key(Some("Alt+F4")), "alt+f4");
        // Only the leading tokens (all but the last) are sorted as
        // "modifiers"; the last token is always kept as the trailing key, so
        // reversing which token comes last changes the result rather than
        // being normalised away. This is the Python behaviour, ported as-is.
        assert_eq!(normalise_key(Some("F4+Alt")), "f4+alt");
        assert_eq!(normalise_key(Some("altleft+f4")), "alt+f4");
        assert_eq!(normalise_key(None), "");
        assert_eq!(normalise_key(Some("")), "");
    }
}
