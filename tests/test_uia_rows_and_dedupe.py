"""Cockpit 6 item 2: named list rows are collected; shared subtrees are walked once."""

from unittest.mock import patch

from grace.perception import uia_provider
from grace.perception.uia_provider import UiaProvider

PROPS = {
    "name": 1, "control_type": 2, "rect": 3, "automation_id": 4, "class_name": 5,
    "focusable": 6, "focused": 7, "enabled": 8, "offscreen": 9, "help_text": 10,
    "aria_role": 11, "aria_props": 12, "value": 13, "is_password": 14,
    "runtime_id": 15, "invoke": 16, "selection_item": 17, "legacy_action": 18,
}


class FakeEl:
    def __init__(self, ctype, name="", rect=(0, 0, 100, 30), rid=None, invoke=False,
                 selection=False, action="", children=None):
        self.children = children or []
        self.v = {
            PROPS["name"]: name, PROPS["control_type"]: ctype,
            PROPS["rect"]: (rect[0], rect[1], rect[2], rect[3]),
            PROPS["runtime_id"]: rid, PROPS["invoke"]: invoke,
            PROPS["selection_item"]: selection, PROPS["legacy_action"]: action,
            PROPS["enabled"]: True,
        }
        self.parents = []

    def GetCachedPropertyValue(self, pid):
        return self.v.get(pid)


class FakeWalker:
    def GetFirstChildElementBuildCache(self, node, _req):
        return node.children[0] if node.children else None

    def GetNextSiblingElementBuildCache(self, node, _req):
        parent = node._walk_parent
        i = parent.children.index(node)
        return parent.children[i + 1] if i + 1 < len(parent.children) else None


class FakeWalkerShared(FakeWalker):
    """A node reachable from two parents keeps its sibling link per visit."""

    def GetFirstChildElementBuildCache(self, node, _req):
        if not node.children:
            return None
        node.children[0]._walk_parent = node
        for c in node.children:
            c._walk_parent = node
        return node.children[0]


class FakeUiaRoot:
    def __init__(self, root):
        self.root = root

    def ElementFromHandleBuildCache(self, hwnd, req):
        return self.root


class FakeUia:
    def __init__(self, root):
        self.props = PROPS
        self.walker = FakeWalkerShared()
        self.cache_request = object()
        self.uia = FakeUiaRoot(root)


def _run(root):
    with patch.object(uia_provider, "_get_uia", return_value=FakeUia(root)):
        return UiaProvider().snapshot(1, title="WhatsApp", class_name="WinUIDesktopWin32WindowClass")


def _whatsapp_like(shared=False):
    row = lambda i, n: FakeEl(50026, n, rect=(0, 200 + i * 60, 300, 258 + i * 60), rid=(42, 100 + i),
                              invoke=True, children=[FakeEl(50020, n, rect=(10, 205 + i * 60, 200, 225 + i * 60), rid=(42, 200 + i))])
    chat_list = FakeEl(50026, "Chat list", rect=(0, 150, 300, 900), rid=(42, 50), children=[
        row(0, "Mamma"), row(1, "Papa"),
    ])
    box = FakeEl(50004, "Search or start a new chat", rect=(10, 90, 250, 120), rid=(42, 60))
    pane = FakeEl(50033, "", rid=(42, 40), children=[box, chat_list])
    doc = FakeEl(50030, "WhatsApp", rect=(0, 0, 1000, 1000), rid=(42, 30), children=[pane])
    # The same UIA node exposed under the parent twice: distinct proxy objects,
    # identical runtime id and subtree.
    twin = FakeEl(50030, "WhatsApp", rect=(0, 0, 1000, 1000), rid=(42, 30), children=[pane])
    top = [doc, twin] if shared else [doc]
    return FakeEl(50032, "WhatsApp", rid=(42, 1), children=top)


def test_named_list_rows_in_group_are_elements():
    els = _run(_whatsapp_like())
    names = [e.name for e in els]
    assert "Mamma" in names and "Papa" in names
    assert "Search or start a new chat" in names
    # unnamed / non-clickable groups are not targets
    assert "Chat list" not in names


def test_group_without_invoke_or_name_is_not_reported():
    plain = FakeEl(50026, "Just a label group", rid=(1, 1))
    unnamed = FakeEl(50026, "", rid=(1, 2), invoke=True)
    root = FakeEl(50032, "w", rid=(1, 0), children=[plain, unnamed])
    assert _run(root) == []


def test_tree_visited_twice_yields_one_copy():
    els = _run(_whatsapp_like(shared=True))
    keys = [(e.role, e.name, e.rect) for e in els]
    assert len(keys) == len(set(keys))
    assert sum(1 for e in els if e.name == "Search or start a new chat") == 1
    assert sum(1 for e in els if e.name == "Mamma") == 1


def test_duplicate_without_runtime_id_still_deduped_by_role_name_rect():
    a = FakeEl(50000, "OK", rect=(0, 0, 50, 20))
    b = FakeEl(50000, "OK", rect=(0, 0, 50, 20))
    root = FakeEl(50032, "w", children=[a, b])
    els = _run(root)
    assert len(els) == 1 and els[0].id == 1
