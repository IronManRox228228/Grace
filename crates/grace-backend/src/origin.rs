//! Ported from `src/grace/ws_server.py`'s Origin allowlist (ship-blocker #5,
//! PLAN.md §10.1: "The WebSocket enforces an Origin allowlist
//! (`WS_ALLOWED_ORIGINS`)"). See
//! `tests/test_ship_blockers.py::TestWebSocketOriginAllowlist` for the
//! Python tests this file's tests mirror.

use std::collections::BTreeSet;

/// Origins the Tauri renderer actually connects from. WebView2 reports the
/// packaged app's own page as one of the tauri.localhost/tauri://localhost
/// forms depending on config; the dev server is the fixed port declared in
/// both `frontend/renderer/vite.config.ts` and `src-tauri/tauri.conf.json`'s
/// `devUrl`. Anything else talking to this socket is not the renderer Grace
/// ships with.
pub fn default_allowed_origins() -> BTreeSet<String> {
    [
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
        "http://localhost:5173",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Splits the comma-separated `WS_ALLOWED_ORIGINS` override.
pub fn parse_extra_origins(raw: &str) -> BTreeSet<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Whether a WebSocket handshake with this Origin header should proceed.
///
/// No Origin header at all is allowed: only browsers send one, so a
/// non-browser local client (a test script, a health check) would otherwise
/// be rejected for doing nothing wrong. A browser-origin connection from
/// anywhere not in the allowlist is what this guards against - without it,
/// any page open in the user's browser could open this socket, read live
/// transcripts, and inject a fake wake event.
pub fn is_origin_allowed(origin: Option<&str>, allowed: &BTreeSet<String>) -> bool {
    match origin {
        None => true,
        Some(o) => allowed.contains(o),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_origin_header_is_allowed() {
        assert!(is_origin_allowed(None, &default_allowed_origins()));
    }

    #[test]
    fn known_renderer_origins_are_allowed() {
        for origin in [
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
            "http://localhost:5173",
        ] {
            assert!(is_origin_allowed(Some(origin), &default_allowed_origins()), "{origin}");
        }
    }

    #[test]
    fn an_arbitrary_web_page_is_rejected() {
        assert!(!is_origin_allowed(Some("http://evil.example"), &default_allowed_origins()));
    }

    #[test]
    fn ws_allowed_origins_env_extends_but_does_not_replace_the_defaults() {
        let extra = parse_extra_origins("http://example.test, http://foo.test");
        let allowed: BTreeSet<String> = default_allowed_origins().union(&extra).cloned().collect();
        assert!(allowed.contains("http://example.test"));
        assert!(allowed.contains("http://foo.test"));
        assert!(allowed.contains("http://localhost:5173"));
    }

    #[test]
    fn empty_override_parses_to_no_extra_origins() {
        assert!(parse_extra_origins("").is_empty());
        assert!(parse_extra_origins("   ").is_empty());
    }
}
