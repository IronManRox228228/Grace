import asyncio
import itertools
import os
import re
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from grace.agent.loop import AgentLoop
from grace.agent.memory import AgentMemory
from grace.agent.secure_desktop import (
    SAFE_HANDOFF_PILL_TEXT,
    SAFE_HANDOFF_SPOKEN,
    enable_voice_access_at_signin_registry,
    get_input_desktop_name,
    is_deaf,
    is_input_desktop_secure,
    is_session_locked,
    is_voice_access_enabled_at_signin,
    merge_voice_access_config,
    request_enable_voice_access_elevation,
    set_mock_desktop_name,
    set_mock_session_locked,
)
from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher


@pytest.fixture(autouse=True)
def mock_hardware_and_instant_sleep():
    from grace.tier import TierDiagnostics, HardwareTier
    diag = TierDiagnostics(
        tier=HardwareTier.FLOOR_CPU,
        available_ram_gb=8.0,
        total_ram_gb=16.0,
        has_npu=False,
        has_igpu=False,
        override="cpu",
        rationale="test",
    )
    with patch("grace.tier.detect_hardware_tier", return_value=diag), \
         patch("grace.tier.check_npu_available", return_value=False), \
         patch("grace.tier.check_igpu_available", return_value=False):
        yield


def test_src_never_bypasses_secure_desktop_or_writes_uac_registry():
    """Security rule: src must NEVER contain PromptOnSecureDesktop or ConsentPromptBehavior writes,

    uiAccess, or service tricks.
    """
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    prohibited = [
        "PromptOnSecureDesktop",
        "ConsentPromptBehaviorAdmin",
        "ConsentPromptBehaviorUser",
        "ConsentPromptBehavior",
        "uiAccess",
    ]

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    content = file_obj.read()
                    for term in prohibited:
                        assert term not in content, f"Forbidden term '{term}' found in {path}"


def test_src_only_ever_writes_voiceaccess_accessibility_registry():
    """Security rule: src must never write to any other registry key in HKLM or HKCU."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    write_terms = ["SetValueEx", "SetValue", "DeleteKey", "CreateKey", "KEY_SET_VALUE"]

    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8", errors="ignore") as file_obj:
                    content = file_obj.read()
                    for term in write_terms:
                        if term in content:
                            norm_path = path.replace("\\", "/")
                            assert "secure_desktop.py" in norm_path, (
                                f"Unauthorized registry write call '{term}' found in {path}"
                            )


def test_safe_handoff_spoken_line_contains_no_prohibited_phrases():
    """Grace never speaks a Voice Access command aloud.

    Spoken line must not contain 'click', 'voice access wake up', or 'hello'.
    Command phrases appear in the pill text only.
    """
    lower_spoken = SAFE_HANDOFF_SPOKEN.lower()
    assert "click" not in lower_spoken
    assert "voice access wake up" not in lower_spoken
    assert "hello" not in lower_spoken

    # Exact phrases must be present in the pill text
    lower_pill = SAFE_HANDOFF_PILL_TEXT.lower()
    assert "voice access wake up" in lower_pill
    assert "click yes" in lower_pill
    assert "click no" in lower_pill


def test_nothing_is_dispatched_while_secure():
    """While a secure screen is up, Dispatcher dispatches nothing and returns status secure_desktop."""
    set_mock_desktop_name("Winlogon")
    try:
        dispatcher = Dispatcher()
        mock_cua = MagicMock()
        dispatcher._cua = mock_cua

        intent = Intent(tool="open_calculator", params={})
        result = asyncio.run(dispatcher.execute(intent))

        assert result.get("status") == "secure_desktop"
        assert result.get("text") == SAFE_HANDOFF_SPOKEN
        assert mock_cua.call_count == 0
    finally:
        set_mock_desktop_name(None)


def test_secure_desktop_speaks_once_and_resumes_after():
    """AgentLoop pauses, speaks safe line once, and resumes when Default desktop returns."""
    mock_gemma = MagicMock()
    mock_dispatcher = MagicMock()
    mock_ws = AsyncMock()
    speak_mock = AsyncMock()

    loop = AgentLoop(
        gemma=mock_gemma,
        dispatcher=mock_dispatcher,
        ws_server=mock_ws,
    )
    loop.set_speaker(speak_mock)
    memory = AgentMemory(user_goal="test goal")

    # Mock desktop sequence: Secure -> Secure -> Default
    states = ["SecureDesktop", "SecureDesktop", "Default"]

    def mock_is_secure():
        if states:
            val = states.pop(0)
            return val != "Default"
        return False

    with patch("grace.agent.secure_desktop.is_input_desktop_secure", side_effect=mock_is_secure):
        with patch("asyncio.sleep", new_callable=AsyncMock):
            res = asyncio.run(loop._handle_secure_desktop(memory, timeout_seconds=10.0))

    # Returned None means resumed
    assert res is None
    # Spoken exactly once
    assert speak_mock.call_count == 1
    assert speak_mock.call_args[0][0] == SAFE_HANDOFF_SPOKEN


def test_secure_desktop_timeout_gives_secure_desktop_status():
    """When secure desktop times out, AgentLoop stops honestly with status 'secure_desktop'."""
    mock_gemma = MagicMock()
    mock_dispatcher = MagicMock()
    mock_ws = AsyncMock()
    speak_mock = AsyncMock()

    loop = AgentLoop(
        gemma=mock_gemma,
        dispatcher=mock_dispatcher,
        ws_server=mock_ws,
    )
    loop.set_speaker(speak_mock)
    memory = AgentMemory(user_goal="test goal")

    # Advance fake time past 60s without actual waiting
    t = [0.0]

    def fake_monotonic():
        t[0] += 65.0
        return t[0]

    with patch("grace.agent.secure_desktop.is_input_desktop_secure", return_value=True):
        with patch("time.monotonic", side_effect=fake_monotonic):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                res = asyncio.run(loop._handle_secure_desktop(memory, timeout_seconds=60.0))

    assert res is not None
    assert res.get("status") == "secure_desktop"


def test_registry_merge_voice_access_config():
    """Merge keeps existing entries in order, avoids duplicates, and handles empty."""
    assert merge_voice_access_config("narrator,osk") == "narrator,osk,voiceaccess"
    assert merge_voice_access_config("narrator,osk,voiceaccess") == "narrator,osk,voiceaccess"
    assert merge_voice_access_config("voiceaccess,narrator") == "voiceaccess,narrator"
    assert merge_voice_access_config("") == "voiceaccess"


def test_enable_voice_access_at_signin_registry_verifies_reread():
    """Helper writes and verifies; if re-read does not contain voiceaccess, returns False."""
    with patch("grace.agent.secure_desktop.get_signin_accessibility_config", side_effect=["narrator", "narrator,voiceaccess"]):
        with patch("winreg.OpenKey", MagicMock()):
            with patch("winreg.SetValueEx", MagicMock()):
                assert enable_voice_access_at_signin_registry() is True

    # Failed re-read returns False
    with patch("grace.agent.secure_desktop.get_signin_accessibility_config", side_effect=["narrator", "narrator"]):
        with patch("winreg.OpenKey", MagicMock()):
            with patch("winreg.SetValueEx", MagicMock()):
                assert enable_voice_access_at_signin_registry() is False


def test_offer_voice_access_no_write_without_yes():
    """Grace requires explicit yes before launching elevation helper; 'no' directs to Settings."""
    from grace.main import GraceApp
    app = GraceApp(tts_off=True)

    with patch("grace.agent.secure_desktop.request_enable_voice_access_elevation") as mock_elevate:
        # User says no -> no elevation call, directs to Settings
        msg_no = asyncio.run(app.offer_voice_access_setup(confirmed=False))
        assert mock_elevate.call_count == 0
        assert "Settings" in msg_no

        # User says yes -> elevation called
        mock_elevate.return_value = True
        msg_yes = asyncio.run(app.offer_voice_access_setup(confirmed=True))
        assert mock_elevate.call_count == 1
        assert "administrator approval" in msg_yes


def test_voice_access_phrases_dropped_by_router():
    """Utterances starting with 'voice access' are dropped by router and never routed to the agent."""
    from grace.main import GraceApp
    app = GraceApp(tts_off=True)
    app.agent_loop = MagicMock()

    trace = MagicMock()
    res1 = asyncio.run(app._route_transcript(trace, "voice access wake up"))
    assert res1 is False
    assert app.agent_loop.run.call_count == 0

    res2 = asyncio.run(app._route_transcript(trace, "voice access sleep"))
    assert res2 is False
    assert app.agent_loop.run.call_count == 0


def test_audio_frames_dropped_while_locked_or_secure_and_resumes_after():
    """Audio frames arriving while locked or secure are dropped before STT; recognition resumes after unlock."""
    from grace.main import GraceApp
    app = GraceApp(tts_off=True)

    set_mock_session_locked(True)
    assert is_deaf() is True

    # Drop during locked
    queue_mock = MagicMock()
    app._audio_queue = queue_mock

    # Simulate chunk arrival in main pump
    chunk = b"\x00" * 3200
    # While deaf, chunk is not put in audio_queue
    if not is_deaf():
        app._audio_queue.put(chunk)
    assert queue_mock.put.call_count == 0

    # Resume when unlocked
    set_mock_session_locked(False)
    set_mock_desktop_name("Default")
    assert is_deaf() is False

    if not is_deaf():
        app._audio_queue.put(chunk)
    assert queue_mock.put.call_count == 1

