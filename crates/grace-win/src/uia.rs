//! Windows UI Automation provider and tree walker.
//!
//! Real implementation of `ElementGraphSource` using the `windows` crate's COM
//! bindings for `IUIAutomation`, `IUIAutomationTreeWalker` (ControlViewWalker),
//! and `IUIAutomationCacheRequest`.
//!
//! Features:
//! - Cached property requests: reads name, control type, bounding rectangle,
//!   automation id, class name, focus, offscreen, help text, and aria properties in bulk.
//! - Walk timeouts on every invocation, preventing hangs on unresponsive processes.
//! - Enriched element summaries: name, role/control type, parent path breadcrumbs,
//!   window title, and neighbouring static text labels (PLAN.md §3).
//! - Pinned element IDs for the duration of the snapshot.
//! - Browser (Chrome/Edge/Firefox) vs app document boundary classification (chrome vs page).

use crate::{ElementSnapshot, Rect};
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const DEFAULT_WALK_TIMEOUT: Duration = Duration::from_millis(3000);

pub const MAX_ELEMENTS_DEFAULT: usize = 120;
pub const MAX_ELEMENTS_BROWSER: usize = 250;
pub const MAX_DEPTH_DEFAULT: usize = 12;
pub const MAX_DEPTH_BROWSER: usize = 18;
pub const MIN_SIZE_PX: i32 = 4;
pub const MAX_NAME_LEN: usize = 160;

pub const DOCUMENT_CONTROL_TYPE: i32 = 50030;

pub fn control_type_to_role(control_type: i32) -> &'static str {
    match control_type {
        50000 => "button",
        50001 => "calendar",
        50002 => "checkbox",
        50003 => "combobox",
        50004 => "edit",
        50005 => "link",
        50006 => "image",
        50007 => "listitem",
        50008 => "list",
        50009 => "menu",
        50010 => "menubar",
        50011 => "menuitem",
        50012 => "progressbar",
        50013 => "radiobutton",
        50014 => "scrollbar",
        50015 => "slider",
        50016 => "spinbutton",
        50017 => "statusbar",
        50018 => "tab",
        50019 => "tabitem",
        50020 => "text",
        50021 => "toolbar",
        50022 => "tooltip",
        50023 => "tree",
        50024 => "treeitem",
        50025 => "custom",
        50026 => "group",
        50027 => "thumb",
        50028 => "datagrid",
        50029 => "dataitem",
        50030 => "document",
        50031 => "splitbutton",
        50032 => "window",
        50033 => "pane",
        50034 => "header",
        50035 => "headeritem",
        50036 => "table",
        50037 => "titlebar",
        50038 => "separator",
        50039 => "semantickey",
        50040 => "appbar",
        _ => "control",
    }
}

pub fn is_interactive_control(control_type: i32) -> bool {
    matches!(
        control_type,
        50000
            | 50002
            | 50003
            | 50004
            | 50005
            | 50007
            | 50011
            | 50013
            | 50015
            | 50016
            | 50018
            | 50019
            | 50021
            | 50024
            | 50029
            | 50031
            | 50035
            | 50040
    )
}

pub fn is_container_control(control_type: i32) -> bool {
    matches!(
        control_type,
        50008 | 50009 | 50010 | 50023 | 50026 | 50030 | 50032 | 50033 | 50036 | 50034
    )
}

pub fn is_named_region_control(control_type: i32) -> bool {
    matches!(control_type, 50021 | 50026 | 50030 | 50033 | 50008 | 50009 | 50010)
}

pub fn is_browser_window(title: &str, class_name: &str) -> bool {
    if class_name == "Chrome_WidgetWin_1" || class_name == "MozillaWindowClass" {
        return true;
    }
    let lower = title.to_lowercase();
    for kw in ["edge", "chrome", "firefox", "brave", "opera", "vivaldi"] {
        if lower.contains(kw) {
            return true;
        }
    }
    false
}

pub fn parse_aria_properties(raw: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if raw.is_empty() {
        return map;
    }
    for chunk in raw.split(';') {
        if let Some((k, v)) = chunk.split_once('=') {
            let key = k.trim().to_lowercase();
            if !key.is_empty() {
                map.insert(key, v.trim().to_string());
            }
        }
    }
    map
}

#[cfg(windows)]
pub mod real {
    use super::*;
    use crate::ElementGraphSource;
    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationCacheRequest, IUIAutomationElement,
        IUIAutomationTreeWalker, TreeScope_Element, UIA_AriaPropertiesPropertyId,
        UIA_AriaRolePropertyId, UIA_AutomationIdPropertyId, UIA_BoundingRectanglePropertyId,
        UIA_ClassNamePropertyId, UIA_ControlTypePropertyId, UIA_HasKeyboardFocusPropertyId,
        UIA_HelpTextPropertyId, UIA_IsEnabledPropertyId, UIA_IsKeyboardFocusablePropertyId,
        UIA_IsOffscreenPropertyId, UIA_NamePropertyId,
    };

    pub struct WindowsElementGraphSource {
        target_hwnd: Option<isize>,
        timeout: Duration,
    }

    impl WindowsElementGraphSource {
        pub fn new() -> Self {
            Self {
                target_hwnd: None,
                timeout: DEFAULT_WALK_TIMEOUT,
            }
        }

        pub fn with_hwnd(hwnd: isize) -> Self {
            Self {
                target_hwnd: Some(hwnd),
                timeout: DEFAULT_WALK_TIMEOUT,
            }
        }

        pub fn with_timeout(mut self, timeout: Duration) -> Self {
            self.timeout = timeout;
            self
        }
    }

    impl Default for WindowsElementGraphSource {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ElementGraphSource for WindowsElementGraphSource {
        fn snapshot(&mut self) -> anyhow::Result<Vec<ElementSnapshot>> {
            let fg = crate::win32::real::foreground_window();
            let hwnd_val = match self.target_hwnd {
                Some(h) => h,
                None => match &fg {
                    Some(f) => f.hwnd,
                    None => return Ok(Vec::new()),
                },
            };

            let (window_title, class_name) = match &fg {
                Some(f) if f.hwnd == hwnd_val => (f.title.clone(), f.class_name.clone()),
                _ => (String::new(), String::new()),
            };

            let is_browser = is_browser_window(&window_title, &class_name);
            let cap = if is_browser {
                MAX_ELEMENTS_BROWSER
            } else {
                MAX_ELEMENTS_DEFAULT
            };
            let max_depth = if is_browser {
                MAX_DEPTH_BROWSER
            } else {
                MAX_DEPTH_DEFAULT
            };
            let base_frame = if is_browser { "chrome" } else { "app" };

            let deadline = Instant::now() + self.timeout;

            unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                let uia: IUIAutomation =
                    match CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) {
                        Ok(u) => u,
                        Err(e) => anyhow::bail!("Failed to create IUIAutomation: {e}"),
                    };

                let cache_req = match uia.CreateCacheRequest() {
                    Ok(r) => r,
                    Err(e) => anyhow::bail!("Failed to create CacheRequest: {e}"),
                };

                let _ = cache_req.AddProperty(UIA_NamePropertyId);
                let _ = cache_req.AddProperty(UIA_ControlTypePropertyId);
                let _ = cache_req.AddProperty(UIA_BoundingRectanglePropertyId);
                let _ = cache_req.AddProperty(UIA_AutomationIdPropertyId);
                let _ = cache_req.AddProperty(UIA_ClassNamePropertyId);
                let _ = cache_req.AddProperty(UIA_IsKeyboardFocusablePropertyId);
                let _ = cache_req.AddProperty(UIA_HasKeyboardFocusPropertyId);
                let _ = cache_req.AddProperty(UIA_IsEnabledPropertyId);
                let _ = cache_req.AddProperty(UIA_IsOffscreenPropertyId);
                let _ = cache_req.AddProperty(UIA_HelpTextPropertyId);
                let _ = cache_req.AddProperty(UIA_AriaRolePropertyId);
                let _ = cache_req.AddProperty(UIA_AriaPropertiesPropertyId);
                let _ = cache_req.SetTreeScope(TreeScope_Element);

                let root = match uia.ElementFromHandleBuildCache(
                    HWND(hwnd_val as *mut core::ffi::c_void),
                    &cache_req,
                ) {
                    Ok(r) => r,
                    Err(_e) => {
                        return Ok(Vec::new());
                    }
                };

                let walker = match uia.ControlViewWalker() {
                    Ok(w) => w,
                    Err(e) => anyhow::bail!("Failed to get ControlViewWalker: {e}"),
                };

                let mut elements = Vec::new();
                let mut current_sibling_labels = Vec::new();

                walk_node(
                    &walker,
                    &cache_req,
                    &root,
                    &mut elements,
                    &mut current_sibling_labels,
                    0,
                    max_depth,
                    cap,
                    base_frame,
                    "",
                    &window_title,
                    "",
                    deadline,
                );

                Ok(elements)
            }
        }
    }

    unsafe fn walk_node(
        walker: &IUIAutomationTreeWalker,
        cache_req: &IUIAutomationCacheRequest,
        node: &IUIAutomationElement,
        elements: &mut Vec<ElementSnapshot>,
        sibling_labels: &mut Vec<String>,
        depth: usize,
        max_depth: usize,
        cap: usize,
        frame: &str,
        container: &str,
        window_title: &str,
        parent_path: &str,
        deadline: Instant,
    ) {
        if elements.len() >= cap || depth > max_depth || Instant::now() > deadline {
            return;
        }

        let child = match walker.GetFirstChildElementBuildCache(node, cache_req) {
            Ok(c) => c,
            Err(_) => return,
        };

        let mut curr = Some(child);
        let mut local_labels = Vec::new();

        while let Some(child_elem) = curr {
            if elements.len() >= cap || Instant::now() > deadline {
                break;
            }

            let control_type = child_elem.CachedControlType().map(|c| c.0).unwrap_or(0);
            let name_bstr = child_elem.CachedName().unwrap_or_default();
            let name = name_bstr.to_string();
            let name = name.trim();

            let child_frame = if control_type == DOCUMENT_CONTROL_TYPE {
                "page"
            } else {
                frame
            };

            // Collect static labels to supply as neighbouring context to nearby controls
            if control_type == 50020 && !name.is_empty() {
                if local_labels.len() < 5 {
                    local_labels.push(name.to_string());
                }
            }

            if is_interactive_control(control_type) {
                if let Some(snapshot) = build_snapshot(
                    &child_elem,
                    control_type,
                    name,
                    child_frame,
                    container,
                    window_title,
                    parent_path,
                    &local_labels,
                    elements.len() + 1,
                ) {
                    elements.push(snapshot);
                }
            }

            if is_container_control(control_type) || !is_interactive_control(control_type) {
                let mut child_container = container.to_string();
                if is_named_region_control(control_type) && !name.is_empty() {
                    child_container = name.chars().take(60).collect();
                }

                let child_parent_path = if name.is_empty() {
                    parent_path.to_string()
                } else if parent_path.is_empty() {
                    name.to_string()
                } else {
                    format!("{parent_path} > {name}")
                };

                walk_node(
                    walker,
                    cache_req,
                    &child_elem,
                    elements,
                    sibling_labels,
                    depth + 1,
                    max_depth,
                    cap,
                    child_frame,
                    &child_container,
                    window_title,
                    &child_parent_path,
                    deadline,
                );
            }

            curr = walker.GetNextSiblingElementBuildCache(&child_elem, cache_req).ok();
        }
    }

    unsafe fn build_snapshot(
        elem: &IUIAutomationElement,
        control_type: i32,
        name: &str,
        frame: &str,
        container: &str,
        window_title: &str,
        parent_path: &str,
        sibling_labels: &[String],
        id_num: usize,
    ) -> Option<ElementSnapshot> {
        let rect_win = elem.CachedBoundingRectangle().ok()?;
        let left = rect_win.left;
        let top = rect_win.top;
        let right = rect_win.right;
        let bottom = rect_win.bottom;
        let width = right - left;
        let height = bottom - top;

        if width <= MIN_SIZE_PX || height <= MIN_SIZE_PX {
            return None;
        }

        let aria_props_str = elem
            .CachedAriaProperties()
            .map(|b| b.to_string())
            .unwrap_or_default();
        let aria_map = parse_aria_properties(&aria_props_str);

        let mut placeholder = aria_map.get("placeholder").cloned().unwrap_or_default();
        if placeholder.is_empty() {
            placeholder = elem
                .CachedHelpText()
                .map(|b| b.to_string())
                .unwrap_or_default();
        }

        let aria_role = elem
            .CachedAriaRole()
            .map(|b| b.to_string().trim().to_lowercase())
            .unwrap_or_default();
        let role = if !aria_role.is_empty() {
            aria_role
        } else {
            control_type_to_role(control_type).to_string()
        };

        let automation_id = elem
            .CachedAutomationId()
            .map(|b| b.to_string().trim().to_string())
            .unwrap_or_default();

        if name.is_empty() && placeholder.is_empty() && automation_id.is_empty() {
            return None;
        }

        let clean_name = if name.len() > MAX_NAME_LEN {
            name.chars().take(MAX_NAME_LEN).collect()
        } else {
            name.to_string()
        };

        let focused = elem.CachedHasKeyboardFocus().map(|b| b.as_bool()).unwrap_or(false);
        let focusable = elem
            .CachedIsKeyboardFocusable()
            .map(|b| b.as_bool())
            .unwrap_or(false);
        let enabled = elem.CachedIsEnabled().map(|b| b.as_bool()).unwrap_or(true);
        let offscreen = elem.CachedIsOffscreen().map(|b| b.as_bool()).unwrap_or(false);

        Some(ElementSnapshot {
            id: id_num.to_string(),
            name: clean_name,
            role,
            rect: Rect { left, top, right, bottom },
            parent_path: parent_path.to_string(),
            window_title: window_title.to_string(),
            neighbouring_labels: sibling_labels.to_vec(),
            value: String::new(),
            placeholder,
            container: container.to_string(),
            frame: frame.to_string(),
            focused,
            focusable,
            enabled,
            offscreen,
            automation_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aria_properties_parsed_correctly() {
        let raw = "placeholder=Search Google or type a URL;expanded=false;haspopup=true";
        let map = parse_aria_properties(raw);
        assert_eq!(map.get("placeholder").unwrap(), "Search Google or type a URL");
        assert_eq!(map.get("expanded").unwrap(), "false");
        assert_eq!(map.get("haspopup").unwrap(), "true");
    }

    #[test]
    fn control_type_mappings_match_aria_vocabulary() {
        assert_eq!(control_type_to_role(50000), "button");
        assert_eq!(control_type_to_role(50004), "edit");
        assert_eq!(control_type_to_role(50005), "link");
        assert_eq!(control_type_to_role(50030), "document");
        assert_eq!(control_type_to_role(99999), "control");
    }

    #[test]
    fn interactive_controls_identified() {
        assert!(is_interactive_control(50000)); // button
        assert!(is_interactive_control(50004)); // edit
        assert!(is_interactive_control(50005)); // link
        assert!(!is_interactive_control(50020)); // text
        assert!(!is_interactive_control(50033)); // pane
    }

    #[test]
    fn browser_window_detection_checks_classes_and_titles() {
        assert!(is_browser_window("New Tab - Google Chrome", "Chrome_WidgetWin_1"));
        assert!(is_browser_window("Mozilla Firefox", "MozillaWindowClass"));
        assert!(is_browser_window("Search - Microsoft Edge", ""));
        assert!(!is_browser_window("Calculator", "ApplicationFrameWindow"));
        assert!(!is_browser_window("Untitled - Notepad", "Notepad"));
    }

    #[test]
    fn element_snapshot_summary_formats_plan_section_3_fields() {
        let snap = ElementSnapshot {
            id: "1".to_string(),
            name: "Search".to_string(),
            role: "edit".to_string(),
            rect: Rect { left: 10, top: 20, right: 110, bottom: 60 },
            parent_path: "Navigation > Header".to_string(),
            window_title: "My App".to_string(),
            neighbouring_labels: vec!["Find items:".to_string()],
            ..Default::default()
        };

        let summary = snap.summary();
        assert!(summary.contains("name: \"Search\""));
        assert!(summary.contains("role: edit"));
        assert!(summary.contains("parent: \"Navigation > Header\""));
        assert!(summary.contains("window: \"My App\""));
        assert!(summary.contains("near: \"Find items:\""));
    }
}
