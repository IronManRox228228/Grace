"""Tool definitions for the intent system.

Defines all available tools with rich descriptions and parameter schemas
optimized for high-accuracy tool calling in Qwen 3.5 4B.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolParam:
    """A single parameter definition for a tool."""
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None
    param_type: str = "str"  # str, int, float, bool, dict, list


@dataclass
class ToolDefinition:
    """A tool definition with its metadata."""
    name: str
    description: str
    params: list[ToolParam] = field(default_factory=list)
    requires_cua: bool = False
    is_system: bool = False


# CUA tool definitions
CUA_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="cua_click",
        description="Click a control, identified by its element id or by name. Never by position.",
        params=[
            ToolParam("window", "Window object target (e.g. {'title': 'Edge', 'id': 1234})", True, None, "dict"),
            ToolParam("element_id", "The `id` of an element from the element list, or the number on a badge in the marked screenshot. The two are the same numbering.", False, None, "int"),
            ToolParam("target_name", "What to click, in plain words, when it has no id or badge (e.g. 'the search box'). Resolved by a visual model.", False, None, "str"),
            ToolParam("click_count", "Number of clicks (1 for single-click, 2 for double-click)", False, "1", "int"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_type_text",
        description="Type text characters into the currently focused text field, search box, or address bar in a window. Typing appends to whatever the field already contains.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("text", "The exact text string to type", True, None, "str"),
            ToolParam("replace", "Clear the field before typing. Use this for a second search in a box that already holds the first one.", False, "false", "bool"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_press_key",
        description="Press a special key or hotkey combination in a target window.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("key", "Key name or hotkey combo (e.g. 'Return', 'Tab', 'Escape', 'Control_L+a', 'Control_L+c', 'Control_L+v')", True, None, "str"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_screenshot",
        description="Capture a desktop screenshot of a window to perceive visual state and layout.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_text",
        description="Capture accessibility text metadata and Win32 control element hierarchy from a window.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_scroll",
        description="Scroll the contents of the foreground window.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("scrollX", "Horizontal scroll distance (positive=right, negative=left)", False, "0", "int"),
            ToolParam("scrollY", "Vertical scroll distance (positive=down, negative=up)", False, "500", "int"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_drag",
        description="Drag one control onto another, e.g. to reorder a list or move a file.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("from_element_id", "The `id` of the element to drag", True, None, "int"),
            ToolParam("to_element_id", "The `id` of the element to drop it on", True, None, "int"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_activate",
        description="Restore and bring an open window to the foreground focus.",
        params=[
            ToolParam("window", "Target window object with title, app, or hwnd", True, None, "dict"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_list_apps",
        description="List all installed system applications and active window processes.",
        params=[],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_list_windows",
        description="Query and list all currently open desktop windows, titles, and bounding coordinates.",
        params=[],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_get_window",
        description="Query window metadata by its handle or window ID.",
        params=[
            ToolParam("id", "Window handle ID", True, None, "str"),
            ToolParam("app", "Application name", True, None, "str"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_launch",
        description="Launch an application by its name, shortcut (.lnk), or executable path.",
        params=[
            ToolParam("app", "Application name or executable path to launch (e.g. 'WhatsApp', 'Epic Games Launcher', 'msedge')", True, None, "str"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_set_value",
        description="Replace the entire contents of an editable field with a new value. The field must be identified; this cannot be aimed at 'whatever is focused'.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("element_id", "The `id` of the field from the interactive elements list", True, None, "int"),
            ToolParam("value", "New text value to set", True, None, "str"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_secondary_action",
        description="Trigger a context menu or secondary action on a target element.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("element_id", "The `id` of the element from the interactive elements list", True, None, "int"),
            ToolParam("action", "Action name (e.g. 'Raise', 'Scroll Up', 'Context Menu')", True, None, "str"),
        ],
        requires_cua=True,
    ),
]

# System (hardcoded) tool definitions
SYSTEM_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="open_app",
        description="Launch any installed Windows application (WhatsApp, Epic Games Launcher, Spotify, Discord, VS Code, Calculator, Edge, etc.) or open a web URL.",
        params=[
            ToolParam("name", "Application or website name (e.g., 'WhatsApp', 'Epic Games Launcher', 'Calculator', 'YouTube')", True, None, "str"),
            ToolParam("url", "Optional website URL to open in browser (e.g. 'https://youtube.com')", False, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="close_app",
        description="Force close a running desktop application process by name.",
        params=[
            ToolParam("name", "Application process name (e.g. 'WhatsApp', 'msedge', 'EpicGamesLauncher')", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="search_files",
        description="Search for files and documents across the local filesystem matching a query keyword.",
        params=[
            ToolParam("query", "Search phrase or filename pattern", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="open_file",
        description="Open a file or document in its default associated Windows application.",
        params=[
            ToolParam("name", "File path or file name to open", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="read_pdf",
        description="Extract and read a local PDF document aloud using sub-second RAG chunking.",
        params=[
            ToolParam("path", "File path to the target PDF document", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="summarize_pdf",
        description="Summarize a local PDF document using TF-IDF sub-second RAG retrieval.",
        params=[
            ToolParam("path", "File path to the target PDF document", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="adjust_volume",
        description="Adjust system audio volume levels (increase, decrease, set absolute level, or percentage).",
        params=[
            ToolParam("amount", "Volume amount integer (0 to 100)", True, None, "int"),
            ToolParam("mode", "Adjustment mode: 'increase', 'decrease', 'set', or 'percent'", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="lock_computer",
        description="Lock the Windows workstation session immediately (equivalent to Win+L).",
        params=[],
        is_system=True,
    ),
    ToolDefinition(
        name="open_calculator",
        description="Open the native Windows Calculator application.",
        params=[],
        is_system=True,
    ),
    ToolDefinition(
        name="delete_file",
        description="Safely move a file to the Windows Recycle Bin (preserves restore undo).",
        params=[
            ToolParam("name", "File path or name to move to Recycle Bin", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="converse",
        description="Speak directly to the user to answer questions or report goal completion.",
        params=[
            ToolParam("response", "Direct spoken answer to communicate to the user", True, None, "str"),
        ],
        is_system=True,
    ),
    ToolDefinition(
        name="undo",
        description="Undo the last action, text edit, or command using Windows Undo (Ctrl+Z).",
        params=[],
        is_system=True,
    ),
    ToolDefinition(
        name="describe_screen",
        description="Describe what is currently visible on the screen, including the active window and main interactive controls.",
        params=[],
        is_system=True,
    ),
    ToolDefinition(
        name="set_speech_rate",
        description="Set the TTS speech speed rate multiplier (e.g. 0.8 for slower, 1.0 for normal, 1.2 for faster).",
        params=[
            ToolParam("rate", "Speech rate speed multiplier (e.g. 0.5 to 2.0, where 1.0 is normal)", True, "1.0", "float"),
        ],
        is_system=True,
    ),
]

ALL_TOOLS: list[ToolDefinition] = CUA_TOOLS + SYSTEM_TOOLS


def _signature(tool: ToolDefinition) -> str:
    """`name(param: type, optional: type=default)` for one tool."""
    parts = []
    for p in tool.params:
        piece = f"{p.name}: {p.param_type}"
        if not p.required and p.default is not None:
            piece += f"={p.default}"
        parts.append(piece)
    return f"{tool.name}({', '.join(parts)})"


def format_tools_compact() -> str:
    """One line per tool, for the intent prompt.

    The intent prompt is on the critical path for every single utterance, so it
    stays terse for prefill latency - but it is generated from this list rather
    than hand-written. The hand-written copy had already lost
    `cua_secondary_action` and `cua_get_window`, so the model could never call
    them even though the parser accepted them.
    """
    lines = ["1. System Tools (Native Windows):"]
    lines += [f"- {_signature(t)}: {t.description}" for t in SYSTEM_TOOLS]
    lines.append("")
    lines.append("2. CUA Tools (UI Automation):")
    lines += [f"- {_signature(t)}: {t.description}" for t in CUA_TOOLS]
    return "\n".join(lines)


def format_tools_for_prompt() -> str:
    """Format all available tools into a rich, detailed schema string for system prompts."""
    lines = []
    for tool in ALL_TOOLS:
        lines.append(f"### `{tool.name}`")
        lines.append(f"Description: {tool.description}")
        if tool.params:
            lines.append("Parameters:")
            for p in tool.params:
                req_str = "required" if p.required else f"optional, default: {p.default}"
                lines.append(f"  - `{p.name}` ({p.param_type}, {req_str}): {p.description}")
        else:
            lines.append("Parameters: None")
        lines.append("")
    return "\n".join(lines)
