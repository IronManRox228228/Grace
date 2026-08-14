"""Every tool Grace advertises must be one it can actually run.

A tool name is agreed on by four places that never look at each other:

- ``intent/tools.py``   declares it to the model, in the prompt
- ``ComputerUse``       implements it, found by reflection (``_<action>``)
- ``Dispatcher``        routes it, via a dict literal inside ``_execute``
- ``CapabilityRouter``  classifies it as fast-path or agentic

Each is individually correct and internally consistent, so no unit test can see
a disagreement between them. The failure mode is specific: ``perform`` resolves
handlers with ``getattr(self, f"_{action}")`` (``computer_use.py:66``), so a
tool advertised to the LLM with no matching method returns ``{"error": "Unknown
action"}`` at runtime - the model picks it, the user hears an error, and every
test still passes.
"""

import ast
import inspect
import textwrap

import pytest

from grace.automation.computer_use import ComputerUse
from grace.intent.parser import VALID_TOOLS
from grace.intent.router import CapabilityRouter
from grace.intent.tools import ALL_TOOLS
from grace.tools.dispatcher import Dispatcher

pytestmark = pytest.mark.integration


TOOL_NAMES = {tool.name for tool in ALL_TOOLS}
CUA_TOOLS = {name for name in TOOL_NAMES if name.startswith("cua_")}
SYSTEM_TOOLS = TOOL_NAMES - CUA_TOOLS

# Declared tools the router deliberately does not classify.
#
# `converse` never reaches the router as a tool - `Intent.is_conversation`
# short-circuits it. The other three are read-only observation steps the planner
# uses *during* an agentic goal; they are never the opening move of a request,
# and falling through to the agentic default is the right answer for them.
#
# Pinned so that a genuinely new tool cannot join this set by accident: an
# unrouted tool silently defaults to AGENTIC_GOAL, which is safe for a screen
# read and wrong for anything that acts.
UNROUTED_BY_DESIGN = {"converse", "cua_get_window", "cua_screenshot", "cua_text"}


def dispatcher_handler_map_keys() -> set[str]:
    """The tool names wired in ``Dispatcher._execute``'s handler_map.

    Read out of the source rather than by calling the method: the map is a local
    dict literal built per call, and reaching it any other way would mean
    executing a dispatch. Checking `hasattr(Dispatcher, "_open_app")` instead
    would miss the case this exists to catch - a handler that is implemented but
    never wired into the map.
    """
    # dedent: getsource returns the method still indented inside its class.
    tree = ast.parse(textwrap.dedent(inspect.getsource(Dispatcher._execute)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "handler_map" for t in node.targets
        ):
            return {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError("handler_map literal not found in Dispatcher._execute")


class TestComputerUseHandlers:
    def test_every_cua_tool_has_a_handler(self):
        # `cua_click` -> action `click` (dispatcher.py: `action = tool[4:]`)
        # -> `ComputerUse._click` (computer_use.py: getattr(self, f"_{action}")).
        missing = sorted(
            name for name in CUA_TOOLS
            if not callable(getattr(ComputerUse, f"_{name[4:]}", None))
        )
        assert missing == [], (
            f"declared to the model but not implemented: {missing}. "
            f"perform() resolves handlers by reflection, so this fails only in "
            f"front of the user."
        )

    def test_no_orphaned_cua_handlers(self):
        # The reverse: a handler nothing can reach. Not a user-visible fault,
        # but it is dead code that reads as capability.
        implemented = {
            f"cua_{name[1:]}"
            for name, member in inspect.getmembers(ComputerUse, callable)
            if name.startswith("_") and not name.startswith("__")
        }
        # Only the ones that look like actions - helpers share the underscore.
        orphans = sorted((implemented & {f"cua_{n[4:]}" for n in CUA_TOOLS}) - CUA_TOOLS)
        assert orphans == []


class TestDispatcherRouting:
    def test_every_system_tool_is_wired(self):
        wired = dispatcher_handler_map_keys()
        missing = sorted(SYSTEM_TOOLS - wired)
        assert missing == [], f"declared but unreachable through Dispatcher: {missing}"

    def test_no_phantom_handlers(self):
        wired = dispatcher_handler_map_keys()
        phantom = sorted(wired - TOOL_NAMES)
        assert phantom == [], (
            f"Dispatcher routes tools the model is never told about: {phantom}"
        )


class TestRouterCoverage:
    def test_the_router_only_classifies_real_tools(self):
        routed = CapabilityRouter.FAST_PATH_TOOLS | CapabilityRouter.AGENTIC_TOOLS
        phantom = sorted(routed - TOOL_NAMES)
        assert phantom == [], (
            f"the router classifies tools that do not exist: {phantom}. These "
            f"branches can never be taken."
        )

    def test_a_tool_cannot_quietly_become_unrouted(self):
        routed = CapabilityRouter.FAST_PATH_TOOLS | CapabilityRouter.AGENTIC_TOOLS
        unrouted = TOOL_NAMES - routed
        assert unrouted == UNROUTED_BY_DESIGN, (
            f"the set of unrouted tools changed: {sorted(unrouted)}. An unrouted "
            f"tool defaults to AGENTIC_GOAL. Decide whether that is right for it "
            f"and update UNROUTED_BY_DESIGN."
        )

    def test_fast_path_and_agentic_do_not_overlap(self):
        both = CapabilityRouter.FAST_PATH_TOOLS & CapabilityRouter.AGENTIC_TOOLS
        assert both == set(), f"classified as both fast-path and agentic: {sorted(both)}"


class TestParserAgreement:
    def test_the_parser_accepts_exactly_the_declared_tools(self):
        # The parser validates tool names against VALID_TOOLS before anything
        # downstream sees them, so a drift here rejects a legal plan.
        assert VALID_TOOLS == TOOL_NAMES
