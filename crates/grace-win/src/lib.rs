//! Windows-touching ports: UIA inspector, win32 driver, app indexer, DPI and
//! computer-use actions. Per the task's hard constraints, no test here may
//! drive the real desktop; anything that must touch a live window is behind
//! a trait, with fakes used in tests. See PORT_STATUS.md for exactly what is
//! implemented vs. stubbed in this crate - in short: `app_indexer`'s pure
//! launch-resolution logic (including the ship-blocker #6 shell-character
//! refusal) is fully ported; the UIA element-graph walk, the win32 driver,
//! DPI awareness ordering and the pyautogui-backed computer-use actions are
//! trait boundaries only, not yet implemented against the real Windows API.

pub mod app_indexer;
pub mod dpi;
pub mod win32;

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

/// One element in the accessibility/element graph
/// (`src/grace/agent/perception.py`'s graph builder output). Deliberately a
/// short summary, never the whole subtree - PLAN.md §3's "the decision model
/// scores each candidate element as a short summary... never the whole tree
/// per element."
#[derive(Debug, Clone, PartialEq)]
pub struct ElementSnapshot {
    pub id: String,
    pub name: String,
    pub role: String,
    pub rect: Rect,
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
