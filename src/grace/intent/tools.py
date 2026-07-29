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
        description="Perform mouse clicks at coordinates (relative to window top-left or absolute screen coordinates) or on an accessibility element in a target window.",
        params=[
            ToolParam("window", "Window object target (e.g. {'title': 'Edge', 'id': 1234})", True, None, "dict"),
            ToolParam("x", "X pixel coordinate (e.g. 200 or 633)", False, None, "int"),
            ToolParam("y", "Y pixel coordinate (e.g. 200 or 280)", False, None, "int"),
            ToolParam("element_index", "Element index integer from accessibility tree context", False, None, "int"),
            ToolParam("click_count", "Number of clicks (1 for single-click, 2 for double-click)", False, "1", "int"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_type_text",
        description="Type text characters into the currently focused text field, search box, or address bar in a window.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("text", "The exact text string to type", True, None, "str"),
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
        description="Scroll window contents vertically or horizontally from a coordinate origin.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("x", "X origin coordinate to scroll from", True, None, "int"),
            ToolParam("y", "Y origin coordinate to scroll from", True, None, "int"),
            ToolParam("scrollX", "Horizontal scroll distance (positive=right, negative=left)", False, "0", "int"),
            ToolParam("scrollY", "Vertical scroll distance (positive=down, negative=up)", False, "500", "int"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_drag",
        description="Perform a mouse drag operation from start pixel coordinates to end pixel coordinates.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("from_x", "Starting X coordinate", True, None, "int"),
            ToolParam("from_y", "Starting Y coordinate", True, None, "int"),
            ToolParam("to_x", "Ending X coordinate", True, None, "int"),
            ToolParam("to_y", "Ending Y coordinate", True, None, "int"),
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
        description="Directly set the string value of an editable field element by element index.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("element_index", "Element index integer", True, None, "int"),
            ToolParam("value", "New text value to set", True, None, "str"),
        ],
        requires_cua=True,
    ),
    ToolDefinition(
        name="cua_secondary_action",
        description="Trigger a context menu or secondary action on a target element index.",
        params=[
            ToolParam("window", "Target window object", True, None, "dict"),
            ToolParam("element_index", "Element index integer", True, None, "int"),
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
]

ALL_TOOLS: list[ToolDefinition] = CUA_TOOLS + SYSTEM_TOOLS


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
