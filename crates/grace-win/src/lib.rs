//! Windows-touching ports: UIA inspector, win32 driver, app indexer, DPI and
//! computer-use actions. Per the task's hard constraints, no test here may
//! drive the real desktop; anything that must touch a live window is behind
//! a trait, with fakes used in tests. See PORT_STATUS.md for exactly what is
//! implemented vs. stubbed in this crate - in short: `app_indexer`'s pure
//! launch-resolution logic (including the ship-blocker #6 shell-character
//! refusal) is fully ported; the UIA element-graph walk, the win32 driver,
//! DPI awareness ordering and the pyautogui-backed computer-use actions are
//! trait boundaries only, not yet implemented against the real Windows API.

pub mod actuator;
pub mod app_indexer;
pub mod dpi;
pub mod perception_source;
pub mod screenshot;
pub mod system_actions;
pub mod uia;
pub mod win32;

pub use actuator::Win32Actuator;
pub use perception_source::WindowsPerceptionSource;
pub use screenshot::encode_rgba_to_png;
#[cfg(windows)]
pub use screenshot::real::capture_foreground_window;
pub use system_actions::{WindowsCuaBridge, WindowsSystemActions};
#[cfg(windows)]
pub use uia::real::WindowsElementGraphSource;

/// A rectangle in screen pixels, as the element graph and computer-use
/// actions exchange coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl Rect {
    pub fn center(&self) -> (i32, i32) {
        ((self.left + self.right) / 2, (self.top + self.bottom) / 2)
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        }
    }
}

/// One element in the accessibility/element graph
/// (`src/grace/agent/perception.py`'s graph builder output). Deliberately a
/// short summary, never the whole subtree - PLAN.md §3's "the decision model
/// scores each candidate element as a short summary... never the whole tree
/// per element."
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ElementSnapshot {
    pub id: String,
    pub name: String,
    pub role: String,
    pub rect: Rect,
    pub parent_path: String,
    pub window_title: String,
    pub neighbouring_labels: Vec<String>,
    pub value: String,
    pub placeholder: String,
    pub container: String,
    pub frame: String,
    pub focused: bool,
    pub focusable: bool,
    pub enabled: bool,
    pub offscreen: bool,
    pub automation_id: String,
}

impl ElementSnapshot {
    pub fn new(id: impl Into<String>, name: impl Into<String>, role: impl Into<String>, rect: Rect) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            role: role.into(),
            rect,
            enabled: true,
            ..Default::default()
        }
    }

    /// Generates a concise summary: "name, control type, parent path, window title, neighbouring labels"
    /// matching PLAN.md §3 / §5 for candidate element scoring.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.name.is_empty() {
            parts.push(format!("name: {:?}", self.name));
        } else if !self.placeholder.is_empty() {
            parts.push(format!("placeholder: {:?}", self.placeholder));
        }
        parts.push(format!("role: {}", self.role));
        if !self.parent_path.is_empty() {
            parts.push(format!("parent: {:?}", self.parent_path));
        }
        if !self.window_title.is_empty() {
            parts.push(format!("window: {:?}", self.window_title));
        }
        if !self.neighbouring_labels.is_empty() {
            parts.push(format!("near: {:?}", self.neighbouring_labels.join(", ")));
        }
        parts.join(", ")
    }

    pub fn to_element_node(&self) -> grace_core::perception::ElementNode {
        let id = self.id.parse::<i64>().unwrap_or(0);
        let (cx, cy) = self.rect.center();
        grace_core::perception::ElementNode {
            id,
            role: self.role.clone(),
            name: self.name.clone(),
            rect: (self.rect.left as i64, self.rect.top as i64, self.rect.right as i64, self.rect.bottom as i64),
            center: (cx as i64, cy as i64),
            value: self.value.clone(),
            placeholder: self.placeholder.clone(),
            frame: if self.frame.is_empty() { grace_core::perception::FRAME_APP.to_string() } else { self.frame.clone() },
            container: self.container.clone(),
            focused: self.focused,
            focusable: self.focusable,
            enabled: self.enabled,
            offscreen: self.offscreen,
            automation_id: self.automation_id.clone(),
            source: grace_core::perception::SOURCE_UIA.to_string(),
        }
    }
}

/// What `PerceptionEngine`'s UIA walk exposes: the current foreground
/// window's element graph. A real implementation walks the UI Automation
/// tree (behind `cfg(windows)`); this trait is the seam a fake snapshot
/// (from a tape, or a hand-built test fixture) implements instead.
pub trait ElementGraphSource: Send {
    fn snapshot(&mut self) -> anyhow::Result<Vec<ElementSnapshot>>;
}

/// A fixed, pre-built `ElementGraphSource` for tests and the harness -
/// mirrors how `replay.py` pins a tape's `snapshots.jsonl` instead of
/// walking a real window.
pub struct FixedElementGraph {
    pub elements: Vec<ElementSnapshot>,
}

impl ElementGraphSource for FixedElementGraph {
    fn snapshot(&mut self) -> anyhow::Result<Vec<ElementSnapshot>> {
        Ok(self.elements.clone())
    }
}

/// What the computer-use dispatcher's leaves do to the real desktop: click,
/// type, scroll, drag, press a key, launch a process. A real implementation
/// wraps pyautogui-equivalent Win32 calls (behind `cfg(windows)`); tests use
/// a fake that records calls instead of moving the real mouse.
pub trait DesktopActuator: Send {
    fn click(&mut self, x: i32, y: i32) -> anyhow::Result<()>;
    fn type_text(&mut self, text: &str) -> anyhow::Result<()>;
    fn press_key(&mut self, key: &str) -> anyhow::Result<()>;
    fn scroll(&mut self, amount: i32) -> anyhow::Result<()>;
    /// Starts a process/protocol/URI the way `os.startfile` would - no
    /// shell, no re-parsing (ship-blocker #6). Implementations must not fall
    /// back to a shell for an unresolved target.
    fn start(&mut self, target: &str) -> anyhow::Result<()>;
    /// Sets expected window title for foreground re-check (R11/R12/R22 semantics).
    fn set_target_window(&mut self, _title: Option<&str>) {}
}

/// A `DesktopActuator` that records every call instead of touching the
/// desktop, for tests that assert "which action was requested" without
/// requiring a real screen.
#[derive(Debug, Default)]
pub struct RecordingActuator {
    pub calls: Vec<String>,
}

impl DesktopActuator for RecordingActuator {
    fn click(&mut self, x: i32, y: i32) -> anyhow::Result<()> {
        self.calls.push(format!("click({x},{y})"));
        Ok(())
    }
    fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        self.calls.push(format!("type_text({text:?})"));
        Ok(())
    }
    fn press_key(&mut self, key: &str) -> anyhow::Result<()> {
        self.calls.push(format!("press_key({key:?})"));
        Ok(())
    }
    fn scroll(&mut self, amount: i32) -> anyhow::Result<()> {
        self.calls.push(format!("scroll({amount})"));
        Ok(())
    }
    fn start(&mut self, target: &str) -> anyhow::Result<()> {
        self.calls.push(format!("start({target:?})"));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_center_matches_python_midpoint_convention() {
        let r = Rect {
            left: 0,
            top: 0,
            right: 100,
            bottom: 50,
        };
        assert_eq!(r.center(), (50, 25));
    }

    #[test]
    fn recording_actuator_records_without_touching_the_desktop() {
        let mut actuator = RecordingActuator::default();
        actuator.click(10, 20).unwrap();
        actuator.type_text("hello").unwrap();
        assert_eq!(actuator.calls, vec!["click(10,20)", "type_text(\"hello\")"]);
    }

    /// A genuinely desktop-driving test would go here, `#[ignore]`d with a
    /// reason - e.g. once a real `cfg(windows)` UIA walk is implemented,
    /// exercising it against an actual foreground window. None exists yet
    /// because no such implementation exists yet (see PORT_STATUS.md); this
    /// test documents where it belongs rather than skipping silently.
    #[test]
    #[ignore = "no real-desktop UIA/win32 implementation exists yet in this port; \
                see PORT_STATUS.md. This is a placeholder for the test that will \
                exercise it once one does, per the task's rule that anything \
                touching a real desktop must be #[ignore]d with a reason."]
    fn real_uia_walk_placeholder() {
        unimplemented!("no real ElementGraphSource impl to test against a live window yet")
    }
}
