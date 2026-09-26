//! Ported (the pure, launch-target-resolution half) from
//! `src/grace/automation/app_indexer.py`'s `AppIndexer.launch`/`find_app`.
//!
//! Ship-blocker #6 (PLAN.md §10.1): `shell=True` and `cmd.exe start` launches
//! are removed, and names containing shell characters are refused. That
//! refusal is pure string logic and is ported here in full, independent of
//! the actual Windows process-launch call (`os.startfile`), which lives
//! behind the `ProcessLauncher` trait below so it can be faked in tests.
//!
//! What is NOT ported yet: `AppIndexer`'s startup scan of the Start Menu and
//! `Get-StartApps` (installed-app discovery). This module takes the indexed
//! app tables as plain arguments instead of owning the scan, so the
//! resolution logic - the part with the safety property and the "zoom
//! matches the uninstaller" bug from PLAN.md §10.2 to fix later - is
//! testable without a real filesystem or PowerShell.

use std::collections::BTreeMap;

/// Characters cmd.exe (or a future shell path) would re-parse as its own
/// syntax. `name` can be model-generated text, so a launch request must
/// never carry one through to a process launch.
const UNSAFE_NAME_CHARS: &[char] = &['&', '|', '<', '>', '^', '"', '%'];

pub fn has_unsafe_chars(name: &str) -> bool {
    name.chars().any(|c| UNSAFE_NAME_CHARS.contains(&c))
}

/// Windows protocol URI handlers resolved directly, before any app-index
/// lookup.
pub fn protocol_for(clean_name: &str) -> Option<&'static str> {
    match clean_name {
        "whatsapp" => Some("whatsapp://"),
        "settings" => Some("ms-settings:"),
        "calculator" | "calc" => Some("calc:"),
        _ => None,
    }
}

/// What `find_app` resolves a query to: a plain shortcut/exe path, or a UWP
/// app's AppID (launched via `shell:AppsFolder\<id>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppTarget {
    Path(String),
    Uwp(String),
}

/// Mirrors `AppIndexer.find_app`: exact match first, then substring
/// (either direction) over indexed shortcuts, then the same two passes over
/// UWP apps. `apps` and `uwp_apps` are keyed by the same lowercase name
/// `_apps`/`_uwp_apps` use in Python.
pub fn find_app(
    query: &str,
    apps: &BTreeMap<String, String>,
    uwp_apps: &BTreeMap<String, String>,
) -> Option<AppTarget> {
    let q = query.to_lowercase();
    let q = q.trim();

    if let Some(path) = apps.get(q) {
        return Some(AppTarget::Path(path.clone()));
    }
    if let Some(appid) = uwp_apps.get(q) {
        return Some(AppTarget::Uwp(appid.clone()));
    }
    for (app_name, app_path) in apps {
        if app_name.contains(q) || q.contains(app_name.as_str()) {
            return Some(AppTarget::Path(app_path.clone()));
        }
    }
    for (uwp_name, appid) in uwp_apps {
        if uwp_name.contains(q) || q.contains(uwp_name.as_str()) {
            return Some(AppTarget::Uwp(appid.clone()));
        }
    }
    None
}

/// What `launch()` decided to do, before any process is actually started.
/// Kept distinct from doing it so the decision is unit-testable on its own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchPlan {
    Refused { reason: String },
    Protocol(&'static str),
    Uwp(String),
    Path(String),
    /// Nothing resolved; try the raw cleaned name as a last resort, exactly
    /// as Python's final `os.startfile(clean_name)` does.
    RawFallback(String),
}

/// Mirrors `AppIndexer.launch` up to (not including) the actual
/// `os.startfile`/`explorer.exe` call.
pub fn plan_launch(
    name: &str,
    apps: &BTreeMap<String, String>,
    uwp_apps: &BTreeMap<String, String>,
) -> LaunchPlan {
    if name.is_empty() {
        return LaunchPlan::Refused {
            reason: "Missing application name".to_string(),
        };
    }
    if has_unsafe_chars(name) {
        return LaunchPlan::Refused {
            reason: format!("Refusing to launch '{name}': contains unsafe characters."),
        };
    }

    let clean_name = name.to_lowercase();
    let clean_name = clean_name.trim();

    if let Some(protocol) = protocol_for(clean_name) {
        return LaunchPlan::Protocol(protocol);
    }

    match find_app(clean_name, apps, uwp_apps) {
        Some(AppTarget::Uwp(appid)) => LaunchPlan::Uwp(appid),
        Some(AppTarget::Path(path)) => LaunchPlan::Path(path),
        None => LaunchPlan::RawFallback(clean_name.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apps() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("notepad".to_string(), "C:/Windows/notepad.exe".to_string()),
            ("spotify".to_string(), "C:/Users/x/Spotify.exe".to_string()),
        ])
    }

    #[test]
    fn unsafe_characters_are_refused() {
        for bad in ["chrome & del c:", "notepad | more", "app\"name", "a%b"] {
            assert!(has_unsafe_chars(bad), "{bad}");
        }
        assert!(!has_unsafe_chars("notepad"));
    }

    #[test]
    fn plan_launch_refuses_unsafe_names_before_resolving_anything() {
        let plan = plan_launch("notepad & calc", &apps(), &BTreeMap::new());
        assert!(matches!(plan, LaunchPlan::Refused { .. }));
    }

    #[test]
    fn plan_launch_refuses_empty_names() {
        let plan = plan_launch("", &apps(), &BTreeMap::new());
        assert!(matches!(plan, LaunchPlan::Refused { .. }));
    }

    #[test]
    fn protocol_names_resolve_before_the_app_index() {
        assert_eq!(
            plan_launch("Settings", &apps(), &BTreeMap::new()),
            LaunchPlan::Protocol("ms-settings:")
        );
    }

    #[test]
    fn exact_and_substring_app_matches_resolve_to_a_path() {
        assert_eq!(
            plan_launch("Notepad", &apps(), &BTreeMap::new()),
            LaunchPlan::Path("C:/Windows/notepad.exe".to_string())
        );
        assert_eq!(
            plan_launch("spot", &apps(), &BTreeMap::new()),
            LaunchPlan::Path("C:/Users/x/Spotify.exe".to_string())
        );
    }

    #[test]
    fn unresolved_names_fall_back_to_the_raw_cleaned_name() {
        assert_eq!(
            plan_launch("some random app", &apps(), &BTreeMap::new()),
            LaunchPlan::RawFallback("some random app".to_string())
        );
    }
}
