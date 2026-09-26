//! Ported from `src/grace/intent/tools.py`: the tool schema shown to the
//! planner/intent model. Data + string formatting only - no execution logic
//! (that's `dispatcher.rs`).

#[derive(Debug, Clone, PartialEq)]
pub struct ToolParam {
    pub name: &'static str,
    pub description: &'static str,
    pub required: bool,
    pub default: Option<&'static str>,
    pub param_type: &'static str,
}

impl ToolParam {
    const fn req(name: &'static str, description: &'static str, param_type: &'static str) -> Self {
        Self { name, description, required: true, default: None, param_type }
    }
    const fn opt(
        name: &'static str,
        description: &'static str,
        default: &'static str,
        param_type: &'static str,
    ) -> Self {
        Self { name, description, required: false, default: Some(default), param_type }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: &'static str,
    pub description: &'static str,
    pub params: &'static [ToolParam],
    pub requires_cua: bool,
    pub is_system: bool,
}

macro_rules! cua_tool {
    ($name:expr, $desc:expr, [$($param:expr),* $(,)?]) => {
        ToolDefinition { name: $name, description: $desc, params: &[$($param),*], requires_cua: true, is_system: false }
    };
}

macro_rules! system_tool {
    ($name:expr, $desc:expr, [$($param:expr),* $(,)?]) => {
        ToolDefinition { name: $name, description: $desc, params: &[$($param),*], requires_cua: false, is_system: true }
    };
}

pub static CUA_TOOLS: &[ToolDefinition] = &[
    cua_tool!(
        "cua_click",
        "Click a control, identified by its element id or by name. Never by position.",
        [
            ToolParam::req("window", "Window object target (e.g. {'title': 'Edge', 'id': 1234})", "dict"),
            ToolParam::opt("element_id", "The `id` of an element from the element list, or the number on a badge in the marked screenshot. The two are the same numbering.", "", "int"),
            ToolParam::opt("target_name", "What to click, in plain words, when it has no id or badge (e.g. 'the search box'). Resolved by a visual model.", "", "str"),
            ToolParam::opt("click_count", "Number of clicks (1 for single-click, 2 for double-click)", "1", "int"),
        ]
    ),
    cua_tool!(
        "cua_type_text",
        "Type text characters into the currently focused text field, search box, or address bar in a window. Typing appends to whatever the field already contains.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::req("text", "The exact text string to type", "str"),
            ToolParam::opt("replace", "Clear the field before typing. Use this for a second search in a box that already holds the first one.", "false", "bool"),
        ]
    ),
    cua_tool!(
        "cua_press_key",
        "Press a special key or hotkey combination in a target window.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::req("key", "Key name or hotkey combo (e.g. 'Return', 'Tab', 'Escape', 'Control_L+a', 'Control_L+c', 'Control_L+v')", "str"),
        ]
    ),
    cua_tool!(
        "cua_screenshot",
        "Capture a desktop screenshot of a window to perceive visual state and layout.",
        [ToolParam::req("window", "Target window object", "dict")]
    ),
    cua_tool!(
        "cua_text",
        "Capture accessibility text metadata and Win32 control element hierarchy from a window.",
        [ToolParam::req("window", "Target window object", "dict")]
    ),
    cua_tool!(
        "cua_scroll",
        "Scroll the contents of the foreground window.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::opt("scrollX", "Horizontal scroll distance (positive=right, negative=left)", "0", "int"),
            ToolParam::opt("scrollY", "Vertical scroll distance (positive=down, negative=up)", "500", "int"),
        ]
    ),
    cua_tool!(
        "cua_drag",
        "Drag one control onto another, e.g. to reorder a list or move a file.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::req("from_element_id", "The `id` of the element to drag", "int"),
            ToolParam::req("to_element_id", "The `id` of the element to drop it on", "int"),
        ]
    ),
    cua_tool!(
        "cua_activate",
        "Restore and bring an open window to the foreground focus.",
        [ToolParam::req("window", "Target window object with title, app, or hwnd", "dict")]
    ),
    cua_tool!("cua_list_apps", "List all installed system applications and active window processes.", []),
    cua_tool!("cua_list_windows", "Query and list all currently open desktop windows, titles, and bounding coordinates.", []),
    cua_tool!(
        "cua_get_window",
        "Query window metadata by its handle or window ID.",
        [
            ToolParam::req("id", "Window handle ID", "str"),
            ToolParam::req("app", "Application name", "str"),
        ]
    ),
    cua_tool!(
        "cua_launch",
        "Launch an application by its name, shortcut (.lnk), or executable path.",
        [ToolParam::req("app", "Application name or executable path to launch (e.g. 'WhatsApp', 'Epic Games Launcher', 'msedge')", "str")]
    ),
    cua_tool!(
        "cua_set_value",
        "Replace the entire contents of an editable field with a new value. The field must be identified; this cannot be aimed at 'whatever is focused'.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::req("element_id", "The `id` of the field from the interactive elements list", "int"),
            ToolParam::req("value", "New text value to set", "str"),
        ]
    ),
    cua_tool!(
        "cua_secondary_action",
        "Trigger a context menu or secondary action on a target element.",
        [
            ToolParam::req("window", "Target window object", "dict"),
            ToolParam::req("element_id", "The `id` of the element from the interactive elements list", "int"),
            ToolParam::req("action", "Action name (e.g. 'Raise', 'Scroll Up', 'Context Menu')", "str"),
        ]
    ),
];

pub static SYSTEM_TOOLS: &[ToolDefinition] = &[
    system_tool!(
        "open_app",
        "Launch any installed Windows application (WhatsApp, Epic Games Launcher, Spotify, Discord, VS Code, Calculator, Edge, etc.) or open a web URL.",
        [
            ToolParam::req("name", "Application or website name (e.g., 'WhatsApp', 'Epic Games Launcher', 'Calculator', 'YouTube')", "str"),
            ToolParam::opt("url", "Optional website URL to open in browser (e.g. 'https://youtube.com')", "", "str"),
        ]
    ),
    system_tool!(
        "close_app",
        "Force close a running desktop application process by name.",
        [ToolParam::req("name", "Application process name (e.g. 'WhatsApp', 'msedge', 'EpicGamesLauncher')", "str")]
    ),
    system_tool!(
        "search_files",
        "Search for files and documents across the local filesystem matching a query keyword.",
        [ToolParam::req("query", "Search phrase or filename pattern", "str")]
    ),
    system_tool!(
        "open_file",
        "Open a file or document in its default associated Windows application.",
        [ToolParam::req("name", "File path or file name to open", "str")]
    ),
    system_tool!(
        "read_pdf",
        "Extract and read a local PDF document aloud using sub-second RAG chunking.",
        [ToolParam::req("path", "File path to the target PDF document", "str")]
    ),
    system_tool!(
        "summarize_pdf",
        "Summarize a local PDF document using TF-IDF sub-second RAG retrieval.",
        [ToolParam::req("path", "File path to the target PDF document", "str")]
    ),
    system_tool!(
        "adjust_volume",
        "Adjust system audio volume levels (increase, decrease, set absolute level, or percentage).",
        [
            ToolParam::req("amount", "Volume amount integer (0 to 100)", "int"),
            ToolParam::req("mode", "Adjustment mode: 'increase', 'decrease', 'set', or 'percent'", "str"),
        ]
    ),
    system_tool!("lock_computer", "Lock the Windows workstation session immediately (equivalent to Win+L).", []),
    system_tool!("open_calculator", "Open the native Windows Calculator application.", []),
    system_tool!(
        "delete_file",
        "Safely move a file to the Windows Recycle Bin (preserves restore undo).",
        [ToolParam::req("name", "File path or name to move to Recycle Bin", "str")]
    ),
    system_tool!(
        "converse",
        "Speak directly to the user to answer questions or report goal completion.",
        [ToolParam::req("response", "Direct spoken answer to communicate to the user", "str")]
    ),
    system_tool!("undo", "Undo the last action, text edit, or command using Windows Undo (Ctrl+Z).", []),
    system_tool!(
        "describe_screen",
        "Describe what is currently visible on the screen, including the active window and main interactive controls.",
        []
    ),
    system_tool!(
        "set_speech_rate",
        "Set the TTS speech speed rate multiplier (e.g. 0.8 for slower, 1.0 for normal, 1.2 for faster).",
        [ToolParam::opt("rate", "Speech rate speed multiplier (e.g. 0.5 to 2.0, where 1.0 is normal)", "1.0", "float")]
    ),
];

pub fn all_tools() -> Vec<&'static ToolDefinition> {
    CUA_TOOLS.iter().chain(SYSTEM_TOOLS.iter()).collect()
}

fn signature(tool: &ToolDefinition) -> String {
    let parts: Vec<String> = tool
        .params
        .iter()
        .map(|p| {
            let mut piece = format!("{}: {}", p.name, p.param_type);
            if !p.required {
                if let Some(default) = p.default {
                    if !default.is_empty() {
                        piece.push_str(&format!("={default}"));
                    }
                }
            }
            piece
        })
        .collect();
    format!("{}({})", tool.name, parts.join(", "))
}

/// One line per tool, for the intent prompt.
pub fn format_tools_compact() -> String {
    let mut lines = vec!["1. System Tools (Native Windows):".to_string()];
    lines.extend(
        SYSTEM_TOOLS
            .iter()
            .map(|t| format!("- {}: {}", signature(t), t.description)),
    );
    lines.push(String::new());
    lines.push("2. CUA Tools (UI Automation):".to_string());
    lines.extend(
        CUA_TOOLS
            .iter()
            .map(|t| format!("- {}: {}", signature(t), t.description)),
    );
    lines.join("\n")
}

/// Format all available tools into a rich, detailed schema string for system prompts.
pub fn format_tools_for_prompt() -> String {
    let mut lines = Vec::new();
    for tool in all_tools() {
        lines.push(format!("### `{}`", tool.name));
        lines.push(format!("Description: {}", tool.description));
        if !tool.params.is_empty() {
            lines.push("Parameters:".to_string());
            for p in tool.params {
                let req_str = if p.required {
                    "required".to_string()
                } else {
                    format!("optional, default: {}", p.default.unwrap_or(""))
                };
                lines.push(format!(
                    "  - `{}` ({}, {}): {}",
                    p.name, p.param_type, req_str, p.description
                ));
            }
        } else {
            lines.push("Parameters: None".to_string());
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_has_every_cua_and_system_tool() {
        assert_eq!(all_tools().len(), CUA_TOOLS.len() + SYSTEM_TOOLS.len());
        assert!(all_tools().iter().any(|t| t.name == "cua_secondary_action"));
        assert!(all_tools().iter().any(|t| t.name == "cua_get_window"));
        assert!(all_tools().iter().any(|t| t.name == "converse"));
    }

    #[test]
    fn signature_includes_default_only_for_optional_params_with_one() {
        let click = CUA_TOOLS.iter().find(|t| t.name == "cua_click").unwrap();
        assert_eq!(
            signature(click),
            "cua_click(window: dict, element_id: int, target_name: str, click_count: int=1)"
        );
    }

    #[test]
    fn format_tools_compact_has_both_sections() {
        let compact = format_tools_compact();
        assert!(compact.contains("1. System Tools (Native Windows):"));
        assert!(compact.contains("2. CUA Tools (UI Automation):"));
        assert!(compact.contains("open_app("));
        assert!(compact.contains("cua_click("));
    }

    #[test]
    fn format_tools_for_prompt_documents_every_param() {
        let prompt = format_tools_for_prompt();
        assert!(prompt.contains("### `adjust_volume`"));
        assert!(prompt.contains("`amount` (int, required)"));
        assert!(prompt.contains("### `lock_computer`"));
        assert!(prompt.contains("Parameters: None"));
    }
}
