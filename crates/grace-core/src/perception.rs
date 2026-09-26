//! Ported from `src/grace/perception/elements.py` and
//! `src/grace/perception/element_graph.py` (the pure element-graph data
//! model and resolution logic), plus the observability/markdown-rendering
//! parts of `src/grace/agent/perception.py`.
//!
//! What's NOT here: the real UIA/DOM/CDP tree walk
//! (`perception/uia_provider.py`, `perception/dom_provider.py`) and OCR
//! (`agent/perception.py`'s Windows Media OCR / Tesseract fallback) - those
//! require the real desktop and belong in `grace-win`'s real-implementation
//! phase; see PORT_STATUS.md. `ElementGraphBuilder::build` here takes
//! already-walked elements from a caller-supplied source instead of walking
//! anything itself, so the merge/dedup/renumbering logic - the part with
//! its own bugs and behaviour worth preserving exactly - is testable without
//! a live window.
//!
//! Also not here: `som_overlay`'s screenshot mark-drawing. `observe_for_planner`
//! reproduces its documented fallback path (`render_or_none` returning
//! nothing when a screenshot can't be marked) rather than a real renderer,
//! which is an honest, narrower claim: blind-window observation always
//! renders as text in this port, never a marked screenshot, until a real
//! `som_overlay` port exists.

pub const FRAME_CHROME: &str = "chrome";
pub const FRAME_PAGE: &str = "page";
pub const FRAME_APP: &str = "app";

pub const SOURCE_UIA: &str = "uia";
pub const SOURCE_DOM: &str = "dom";
pub const SOURCE_OCR: &str = "ocr";

pub const ROLE_TEXT: &str = "text";

const INTERACTIVE_ROLES: &[&str] = &[
    "button", "checkbox", "combobox", "edit", "textbox", "hyperlink", "link",
    "listitem", "menuitem", "option", "radiobutton", "searchbox", "slider",
    "spinbutton", "splitbutton", "tab", "tabitem", "toolbar", "treeitem",
    "switch", "menuitemcheckbox", "menuitemradio",
];

/// Fewest actionable controls a window can report and still be treated as
/// readable. Overridable via `Config::observability_min_actionable`.
pub const DEFAULT_MIN_ACTIONABLE: usize = 8;

#[derive(Debug, Clone, PartialEq)]
pub struct ElementNode {
    pub id: i64,
    pub role: String,
    pub name: String,
    pub rect: (i64, i64, i64, i64), // left, top, right, bottom (screen px)
    pub center: (i64, i64),
    pub value: String,
    pub placeholder: String,
    pub frame: String,
    pub container: String,
    pub focused: bool,
    pub focusable: bool,
    pub enabled: bool,
    pub offscreen: bool,
    pub automation_id: String,
    pub source: String,
}

impl Default for ElementNode {
    fn default() -> Self {
        Self {
            id: 0,
            role: String::new(),
            name: String::new(),
            rect: (0, 0, 0, 0),
            center: (0, 0),
            value: String::new(),
            placeholder: String::new(),
            frame: FRAME_APP.to_string(),
            container: String::new(),
            focused: false,
            focusable: false,
            enabled: true,
            offscreen: false,
            automation_id: String::new(),
            source: SOURCE_UIA.to_string(),
        }
    }
}

impl ElementNode {
    pub fn width(&self) -> i64 {
        self.rect.2 - self.rect.0
    }

    pub fn height(&self) -> i64 {
        self.rect.3 - self.rect.1
    }

    pub fn is_interactive(&self) -> bool {
        INTERACTIVE_ROLES.contains(&self.role.to_lowercase().as_str())
    }

    /// Interactive *and* something a step could actually reach right now.
    pub fn is_actionable(&self) -> bool {
        self.is_interactive() && self.enabled && !self.offscreen
    }

    pub fn contains(&self, x: i64, y: i64) -> bool {
        let (left, top, right, bottom) = self.rect;
        left <= x && x <= right && top <= y && y <= bottom
    }

    pub fn distance_to(&self, x: i64, y: i64) -> f64 {
        let (cx, cy) = self.center;
        (((cx - x).pow(2) + (cy - y).pow(2)) as f64).sqrt()
    }

    /// Everything a name-based lookup should be allowed to match against.
    pub fn search_text(&self) -> String {
        [&self.name, &self.placeholder, &self.value, &self.automation_id]
            .into_iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    /// Serialise. `compact` drops defaults to keep prompts small.
    pub fn to_json(&self, compact: bool) -> serde_json::Value {
        use serde_json::json;
        if !compact {
            return json!({
                "id": self.id, "role": self.role, "name": self.name,
                "rect": [self.rect.0, self.rect.1, self.rect.2, self.rect.3],
                "center": [self.center.0, self.center.1], "frame": self.frame,
                "value": self.value, "placeholder": self.placeholder,
                "container": self.container, "focused": self.focused,
                "focusable": self.focusable, "enabled": self.enabled,
                "offscreen": self.offscreen, "automation_id": self.automation_id,
                "source": self.source,
            });
        }
        let mut data = json!({
            "id": self.id, "role": self.role, "name": self.name,
            "rect": [self.rect.0, self.rect.1, self.rect.2, self.rect.3],
            "center": [self.center.0, self.center.1], "frame": self.frame,
        });
        let obj = data.as_object_mut().unwrap();
        if !self.value.is_empty() {
            obj.insert("value".into(), json!(self.value));
        }
        if !self.placeholder.is_empty() {
            obj.insert("placeholder".into(), json!(self.placeholder));
        }
        if !self.container.is_empty() {
            obj.insert("container".into(), json!(self.container));
        }
        if self.focused {
            obj.insert("focused".into(), json!(true));
        }
        if !self.enabled {
            obj.insert("enabled".into(), json!(false));
        }
        data
    }
}

pub fn count_actionable(elements: &[ElementNode]) -> usize {
    elements.iter().filter(|e| e.is_actionable()).count()
}

pub fn find_by_id(elements: &[ElementNode], element_id: i64) -> Option<&ElementNode> {
    elements.iter().find(|e| e.id == element_id)
}

/// Resolve a target by name, preferring exact matches and visible controls.
pub fn find_by_name<'a>(
    elements: &'a [ElementNode],
    query: &str,
    frame: Option<&str>,
    role: Option<&str>,
) -> Option<&'a ElementNode> {
    if query.is_empty() {
        return None;
    }
    let q = query.to_lowercase();
    let q = q.trim();

    let candidates: Vec<&ElementNode> = elements
        .iter()
        .filter(|e| !e.offscreen && e.enabled)
        .filter(|e| frame.map(|f| e.frame == f).unwrap_or(true))
        .filter(|e| role.map(|r| e.role.to_lowercase() == r.to_lowercase()).unwrap_or(true))
        .collect();
    if candidates.is_empty() {
        return None;
    }

    if let Some(found) = candidates.iter().find(|e| e.name.to_lowercase().trim() == q) {
        return Some(found);
    }
    if let Some(found) = candidates.iter().find(|e| e.placeholder.to_lowercase().trim() == q) {
        return Some(found);
    }
    candidates.into_iter().find(|e| e.search_text().contains(q))
}

/// Snap a predicted coordinate onto a nearby small control.
pub fn find_at_point(elements: &[ElementNode], x: i64, y: i64, tolerance_px: i64, max_size: i64) -> Option<&ElementNode> {
    let mut best: Option<&ElementNode> = None;
    let mut best_dist = f64::INFINITY;

    for element in elements {
        if element.offscreen {
            continue;
        }
        let too_big = element.width() > max_size || element.height() > max_size;
        if element.contains(x, y) && !too_big {
            return Some(element);
        }
        if too_big {
            continue;
        }
        let dist = element.distance_to(x, y);
        if dist <= tolerance_px as f64 && dist < best_dist {
            best_dist = dist;
            best = Some(element);
        }
    }
    best
}

/// Render the element graph as a labelled JSON block for the prompt.
pub fn elements_to_prompt(elements: &[ElementNode], limit: Option<usize>) -> String {
    if elements.is_empty() {
        return "### Interactive Elements: (none detected)".to_string();
    }
    let subset: &[ElementNode] = match limit {
        Some(l) => &elements[..elements.len().min(l)],
        None => elements,
    };
    let frames: std::collections::BTreeSet<&str> = subset.iter().map(|e| e.frame.as_str()).collect();

    let mut header = vec!["### Interactive Elements (click/type by `id`)".to_string()];
    if frames.contains(FRAME_CHROME) && frames.contains(FRAME_PAGE) {
        header.push(
            "Note: `frame` is \"chrome\" for browser UI (address bar, tabs) and \"page\" for content inside the web page. A site's own search box is always frame=\"page\"."
                .to_string(),
        );
    }
    let json_array: Vec<serde_json::Value> = subset.iter().map(|e| e.to_json(true)).collect();
    header.push(serde_json::Value::Array(json_array).to_string());
    if let Some(l) = limit {
        if elements.len() > l {
            header.push(format!("({} further elements not shown)", elements.len() - l));
        }
    }
    header.join("\n")
}

/// An immutable snapshot of the interactive controls in one window.
#[derive(Debug, Clone)]
pub struct ElementGraph {
    pub elements: Vec<ElementNode>,
    pub window_title: String,
    pub is_browser: bool,
    pub sources: Vec<String>,
}

impl ElementGraph {
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn actionable_count(&self) -> usize {
        count_actionable(&self.elements)
    }

    pub fn by_id(&self, element_id: i64) -> Option<&ElementNode> {
        find_by_id(&self.elements, element_id)
    }

    pub fn by_name(&self, query: &str, frame: Option<&str>, role: Option<&str>) -> Option<&ElementNode> {
        find_by_name(&self.elements, query, frame, role)
    }

    pub fn focused(&self) -> Option<&ElementNode> {
        self.elements.iter().find(|e| e.focused)
    }

    /// Resolve a planner-supplied target to a concrete element. Tried in
    /// order of reliability: an explicit id, then a name scoped to the
    /// requested frame, then the same name anywhere.
    pub fn resolve(
        &self,
        element_id: Option<i64>,
        target_name: Option<&str>,
        frame: Option<&str>,
        role: Option<&str>,
    ) -> Option<&ElementNode> {
        if let Some(id) = element_id {
            if let Some(found) = self.by_id(id) {
                return Some(found);
            }
        }

        if let Some(name) = target_name.filter(|n| !n.is_empty()) {
            if let Some(f) = frame {
                return self.by_name(name, Some(f), role);
            }
            if self.is_browser {
                if let Some(found) = self.by_name(name, Some(FRAME_PAGE), role) {
                    return Some(found);
                }
            }
            return self.by_name(name, None, role);
        }

        None
    }

    pub fn to_prompt(&self, limit: Option<usize>) -> String {
        elements_to_prompt(&self.elements, limit)
    }
}

/// Merges already-walked elements from a DOM source and a UIA source into
/// one dense-id `ElementGraph`, exactly mirroring
/// `ElementGraphBuilder.build`'s ordering (DOM first, then UIA with
/// overlapping page-frame duplicates dropped) and renumbering.
pub fn build_graph(
    window_title: impl Into<String>,
    is_browser: bool,
    dom_elements: Vec<ElementNode>,
    uia_elements: Vec<ElementNode>,
) -> ElementGraph {
    let mut elements = Vec::new();
    let mut sources = Vec::new();

    if !dom_elements.is_empty() {
        elements.extend(dom_elements.iter().cloned());
        sources.push(SOURCE_DOM.to_string());
    }

    let uia_elements = if !dom_elements.is_empty() {
        drop_overlapping(uia_elements, &dom_elements, 6)
    } else {
        uia_elements
    };
    if !uia_elements.is_empty() {
        elements.extend(uia_elements);
        sources.push(SOURCE_UIA.to_string());
    }

    for (index, element) in elements.iter_mut().enumerate() {
        element.id = (index + 1) as i64;
    }

    ElementGraph { elements, window_title: window_title.into(), is_browser, sources }
}

/// Remove UIA nodes that duplicate a DOM node already collected.
fn drop_overlapping(uia_elements: Vec<ElementNode>, dom_elements: &[ElementNode], tolerance_px: i64) -> Vec<ElementNode> {
    uia_elements
        .into_iter()
        .filter(|element| {
            if element.frame != FRAME_PAGE {
                return true;
            }
            !dom_elements.iter().any(|dom| {
                (element.center.0 - dom.center.0).abs() <= tolerance_px
                    && (element.center.1 - dom.center.1).abs() <= tolerance_px
            })
        })
        .collect()
}

/// Complete perception snapshot at a single point in time. A stripped-down
/// `ScreenSnapshot` from `agent/perception.py` - carries only what the agent
/// loop and planner actually consume (the element graph, a screenshot for
/// grounding, and window identity); the legacy flat UIA list and OCR lines
/// are dropped since nothing in this port produces them (no real screen
/// capture/OCR - see this module's doc comment).
#[derive(Debug, Clone)]
pub struct ScreenSnapshot {
    pub graph: ElementGraph,
    pub window_title: String,
    pub width: u32,
    pub height: u32,
    pub png_bytes: Option<Vec<u8>>,
    pub image_width: u32,
    pub image_height: u32,
}

impl ScreenSnapshot {
    pub fn actionable_count(&self) -> usize {
        self.graph.actionable_count()
    }

    pub fn is_blind(&self, min_actionable: usize) -> bool {
        observability(self.actionable_count(), min_actionable) == Observability::Blind
    }

    /// Window identity and geometry header, shared by both observation modes.
    pub fn header_markdown(&self) -> Vec<String> {
        vec![
            format!("### Focused Window: '{}'", self.window_title),
            format!("Screen Dimensions: {}x{}", self.width, self.height),
        ]
    }

    /// Whether the snapshot describes anything the planner could act on -
    /// false means the window reported no contents at all (the signature of
    /// an app that does not implement UI Automation).
    pub fn has_elements(&self) -> bool {
        !self.graph.is_empty()
    }
}

/// What `AgentLoop::observe` calls: captures the current desktop state. A
/// real implementation walks UIA/DOM and screenshots the desktop (behind
/// `grace-win`, not implemented in this phase); tests use a fixed/scripted
/// snapshot sequence.
pub trait PerceptionSource: Send {
    fn capture_snapshot(&mut self) -> ScreenSnapshot;
}

/// A scripted `PerceptionSource` for tests: returns queued snapshots in
/// order, repeating the last one once the queue is empty (mirrors a window
/// that stopped changing, which is exactly the case the agent loop's
/// repeated-action ladder is built to detect).
pub struct ScriptedPerception {
    snapshots: std::collections::VecDeque<ScreenSnapshot>,
    last: Option<ScreenSnapshot>,
}

impl ScriptedPerception {
    pub fn new(snapshots: Vec<ScreenSnapshot>) -> Self {
        Self { snapshots: snapshots.into(), last: None }
    }
}

impl PerceptionSource for ScriptedPerception {
    fn capture_snapshot(&mut self) -> ScreenSnapshot {
        if let Some(next) = self.snapshots.pop_front() {
            self.last = Some(next.clone());
            next
        } else {
            self.last.clone().unwrap_or(ScreenSnapshot {
                graph: ElementGraph { elements: vec![], window_title: String::new(), is_browser: false, sources: vec![] },
                window_title: String::new(),
                width: 1920,
                height: 1080,
                png_bytes: None,
                image_width: 0,
                image_height: 0,
            })
        }
    }
}

/// The two ways a window can present itself to the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observability {
    Rich,
    Blind,
}

pub fn observability(actionable_count: usize, min_actionable: usize) -> Observability {
    if actionable_count >= min_actionable {
        Observability::Rich
    } else {
        Observability::Blind
    }
}

/// What the planner is actually shown for one step.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub markdown: String,
    pub image_b64: Option<String>,
    pub mode: Observability,
    pub marks: usize,
}

impl Observation {
    pub fn is_blind(&self) -> bool {
        self.mode == Observability::Blind
    }
}

/// Choose an observation that matches what this window will support.
///
/// Falls back to the rich (element-list) rendering when the window is
/// blind, mirroring the Python fallback path for "no markable screenshot" -
/// see this module's doc comment for why no real screenshot marking exists
/// in this port yet.
pub fn observe_for_planner(snapshot: &ScreenSnapshot, min_actionable: usize) -> Observation {
    let mode = observability(snapshot.actionable_count(), min_actionable);
    // Both branches currently render the same way (element list + header):
    // real "blind" behaviour would draw numbered badges on the screenshot
    // (`som_overlay`), which this port does not implement yet - see this
    // module's doc comment. `mode` is still computed and carried on the
    // `Observation` so downstream code (the escalation ladder, verification)
    // reacts exactly as it would once real marking exists.
    let mut lines = snapshot.header_markdown();
    lines.push(String::new());
    lines.push(snapshot.graph.to_prompt(Some(40)));
    Observation { markdown: lines.join("\n"), image_b64: None, mode, marks: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(id: i64, role: &str, name: &str, frame: &str) -> ElementNode {
        ElementNode {
            id,
            role: role.to_string(),
            name: name.to_string(),
            frame: frame.to_string(),
            rect: (0, 0, 10, 10),
            center: (5, 5),
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn is_actionable_requires_interactive_enabled_and_onscreen() {
        let mut e = elem(1, "button", "Go", FRAME_APP);
        assert!(e.is_actionable());
        e.enabled = false;
        assert!(!e.is_actionable());
        e.enabled = true;
        e.offscreen = true;
        assert!(!e.is_actionable());
    }

    #[test]
    fn ocr_text_role_is_not_interactive() {
        let e = elem(1, ROLE_TEXT, "some recognised text", FRAME_APP);
        assert!(!e.is_interactive());
        assert!(!e.is_actionable());
    }

    #[test]
    fn resolve_by_id_wins_first() {
        let elements = vec![elem(1, "button", "A", FRAME_APP), elem(2, "button", "B", FRAME_APP)];
        let graph = ElementGraph { elements, window_title: "".into(), is_browser: false, sources: vec![] };
        assert_eq!(graph.resolve(Some(2), None, None, None).unwrap().name, "B");
    }

    #[test]
    fn resolve_prefers_page_frame_in_a_browser_when_no_frame_given() {
        let elements = vec![
            elem(1, "edit", "search", FRAME_CHROME),
            elem(2, "edit", "search", FRAME_PAGE),
        ];
        let graph = ElementGraph { elements, window_title: "".into(), is_browser: true, sources: vec![] };
        let found = graph.resolve(None, Some("search"), None, None).unwrap();
        assert_eq!(found.frame, FRAME_PAGE);
    }

    #[test]
    fn resolve_respects_an_explicit_frame_even_off_the_default_preference() {
        let elements = vec![
            elem(1, "edit", "search", FRAME_CHROME),
            elem(2, "edit", "search", FRAME_PAGE),
        ];
        let graph = ElementGraph { elements, window_title: "".into(), is_browser: true, sources: vec![] };
        let found = graph.resolve(None, Some("search"), Some(FRAME_CHROME), None).unwrap();
        assert_eq!(found.frame, FRAME_CHROME);
    }

    #[test]
    fn find_by_name_prefers_exact_over_substring() {
        let elements = vec![
            elem(1, "button", "Search Everywhere", FRAME_APP),
            elem(2, "button", "Search", FRAME_APP),
        ];
        let found = find_by_name(&elements, "search", None, None).unwrap();
        assert_eq!(found.id, 2);
    }

    #[test]
    fn build_graph_drops_overlapping_uia_page_elements() {
        let mut dom = elem(1, "button", "Send", FRAME_PAGE);
        dom.center = (100, 100);
        let mut uia_dup = elem(1, "button", "Send", FRAME_PAGE);
        uia_dup.center = (102, 101); // within the 6px tolerance
        let mut uia_chrome = elem(1, "button", "Back", FRAME_CHROME);
        uia_chrome.center = (5, 5);

        let graph = build_graph("Edge", true, vec![dom], vec![uia_dup, uia_chrome]);
        assert_eq!(graph.len(), 2); // dom Send + uia chrome Back, dup dropped
        // ids are dense 1..=n after merge
        assert_eq!(graph.elements[0].id, 1);
        assert_eq!(graph.elements[1].id, 2);
        assert_eq!(graph.sources, vec!["dom".to_string(), "uia".to_string()]);
    }

    #[test]
    fn observability_threshold_is_inclusive() {
        assert_eq!(observability(8, 8), Observability::Rich);
        assert_eq!(observability(7, 8), Observability::Blind);
    }

    #[test]
    fn observe_for_planner_reports_blind_mode_for_a_sparse_window() {
        let elements = vec![elem(1, "button", "OK", FRAME_APP)];
        let graph = ElementGraph { elements, window_title: "Terminal".into(), is_browser: false, sources: vec![] };
        let snapshot = ScreenSnapshot {
            graph,
            window_title: "Terminal".into(),
            width: 1920,
            height: 1080,
            png_bytes: None,
            image_width: 0,
            image_height: 0,
        };
        let obs = observe_for_planner(&snapshot, DEFAULT_MIN_ACTIONABLE);
        assert_eq!(obs.mode, Observability::Blind);
        assert!(obs.is_blind());
        assert!(obs.markdown.contains("Terminal"));
    }
}
