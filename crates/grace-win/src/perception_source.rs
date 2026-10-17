//! Live Windows perception source implementing `grace_core::perception::PerceptionSource`.
//!
//! Bridges `ElementGraphSource` (UIA tree walk) and `screenshot` (GDI BitBlt)
//! into `ScreenSnapshot` consumed by `AgentLoop` and the planner.

use crate::ElementGraphSource;
use grace_core::perception::{ElementGraph, PerceptionSource, ScreenSnapshot};

pub struct WindowsPerceptionSource {
    graph_source: Box<dyn ElementGraphSource>,
    capture_vision: bool,
}

impl WindowsPerceptionSource {
    pub fn new() -> Self {
        #[cfg(windows)]
        let source: Box<dyn ElementGraphSource> =
            Box::new(crate::uia::real::WindowsElementGraphSource::new());
        #[cfg(not(windows))]
        let source: Box<dyn ElementGraphSource> = Box::new(crate::FixedElementGraph { elements: Vec::new() });

        Self {
            graph_source: source,
            capture_vision: false,
        }
    }

    pub fn with_source(source: Box<dyn ElementGraphSource>) -> Self {
        Self {
            graph_source: source,
            capture_vision: false,
        }
    }

    pub fn with_vision(mut self, enable: bool) -> Self {
        self.capture_vision = enable;
        self
    }

    pub fn enable_vision(&mut self, enable: bool) {
        self.capture_vision = enable;
    }
}

impl Default for WindowsPerceptionSource {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceptionSource for WindowsPerceptionSource {
    fn capture_snapshot(&mut self) -> ScreenSnapshot {
        let snapshots = self.graph_source.snapshot().unwrap_or_default();
        let nodes: Vec<_> = snapshots.into_iter().map(|s| s.to_element_node()).collect();

        #[cfg(windows)]
        let fg = crate::win32::real::foreground_window();
        #[cfg(not(windows))]
        let fg: Option<crate::win32::ForegroundWindowPlaceholder> = None;

        let window_title = fg.as_ref().map(|f| f.title.clone()).unwrap_or_default();
        let class_name = fg.as_ref().map(|f| f.class_name.clone()).unwrap_or_default();
        let is_browser = crate::uia::is_browser_window(&window_title, &class_name);

        let graph = ElementGraph {
            elements: nodes,
            window_title: window_title.clone(),
            is_browser,
            sources: vec!["uia".to_string()],
        };

        #[cfg(windows)]
        let (screen_w, screen_h) = {
            let w = unsafe { windows::Win32::UI::WindowsAndMessaging::GetSystemMetrics(windows::Win32::UI::WindowsAndMessaging::SM_CXSCREEN) };
            let h = unsafe { windows::Win32::UI::WindowsAndMessaging::GetSystemMetrics(windows::Win32::UI::WindowsAndMessaging::SM_CYSCREEN) };
            let w = if w <= 0 { 1920 } else { w as u32 };
            let h = if h <= 0 { 1080 } else { h as u32 };
            (w, h)
        };
        #[cfg(not(windows))]
        let (screen_w, screen_h) = (1920u32, 1080u32);

        let mut png_bytes = None;
        let mut image_width = 0;
        let mut image_height = 0;

        if self.capture_vision {
            #[cfg(windows)]
            {
                if let Ok((bytes, w, h)) = crate::screenshot::real::capture_foreground_window(None) {
                    png_bytes = Some(bytes);
                    image_width = w;
                    image_height = h;
                }
            }
        }

        ScreenSnapshot {
            graph,
            window_title,
            width: screen_w,
            height: screen_h,
            png_bytes,
            image_width,
            image_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementSnapshot, FixedElementGraph, Rect};

    #[test]
    fn perception_source_builds_screen_snapshot_from_elements() {
        let elements = vec![
            ElementSnapshot {
                id: "1".to_string(),
                name: "Submit".to_string(),
                role: "button".to_string(),
                rect: Rect { left: 10, top: 10, right: 60, bottom: 40 },
                enabled: true,
                ..Default::default()
            },
        ];

        let fake_source = Box::new(FixedElementGraph { elements });
        let mut perception = WindowsPerceptionSource::with_source(fake_source);

        let snapshot = perception.capture_snapshot();
        assert_eq!(snapshot.graph.len(), 1);
        assert_eq!(snapshot.graph.elements[0].name, "Submit");
        assert_eq!(snapshot.graph.elements[0].role, "button");
        assert_eq!(snapshot.png_bytes, None);
    }
}
