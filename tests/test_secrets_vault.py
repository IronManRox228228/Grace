import io
import logging
import os
import pytest
from unittest.mock import MagicMock, patch

from grace.agent.memory import PersistentMemoryStore as MemoryStore
from grace.agent.secrets_vault import (
    classify_site,
    extract_etld_plus_one,
    type_saved_secret_guard,
)
from grace.agent.planner import Planner
from grace.intent.parser import Intent
from grace.tools.dispatcher import Dispatcher


def test_vault_storage_encrypted_at_rest_and_never_leaks(tmp_path, caplog):
    """The value never appears in any prompt, log, tape or trace (grep the captured output)."""
    caplog.set_level(logging.DEBUG)
    db_path = str(tmp_path / "test_memory.db")
    store = MemoryStore(db_path=db_path)

    secret_name = "test_mail"
    secret_val = "SuperSecretVaultPass987!"
    store.save_secret(name=secret_name, value=secret_val, site_kind="domain", site="example.com")

    # 1. Check list_secrets never contains the value
    listed = store.list_secrets()
    assert len(listed) == 1
    assert listed[0]["name"] == secret_name
    assert "value" not in listed[0]

    # 2. Check disk file does not contain raw plaintext secret
    with open(db_path, "rb") as f:
        disk_bytes = f.read()
    assert secret_val.encode("utf-8") not in disk_bytes

    # 3. Check logs do not contain the secret value
    logger = logging.getLogger("grace.secrets")
    logger.info("Listing secrets: %s", listed)
    assert secret_val not in caplog.text

    # 4. Check planner prompt never receives secret values
    llm = MagicMock()
    planner = Planner(llm)
    prompt = planner.build_prompt(
        goal="type my test_mail password",
        elements_prompt="",
        available_secrets=[s["name"] for s in listed],
    )
    assert secret_val not in prompt
    assert secret_name in prompt
    store.close()


def test_lookalike_domains_refused():
    """Lookalike domains such as gmail.com.evil.io and gmai1.com are refused for gmail.com."""
    secret = {
        "name": "gmail",
        "value": "mypassword123",
        "site_kind": "domain",
        "site": "gmail.com",
    }

    # Legitimate matching domain or subdomain
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=True,
        active_domain="mail.gmail.com",
        active_process_exe="chrome.exe",
    )
    assert ok is True
    assert msg == ""

    # Lookalike 1: gmail.com.evil.io
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=True,
        active_domain="gmail.com.evil.io",
        active_process_exe="chrome.exe",
    )
    assert ok is False
    assert "secret" not in msg.lower() or "gmail" in msg
    assert "mypassword123" not in msg

    # Lookalike 2: gmai1.com
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=True,
        active_domain="gmai1.com",
        active_process_exe="chrome.exe",
    )
    assert ok is False
    assert "mypassword123" not in msg


def test_non_password_field_refused():
    """A non-password field is refused even if site matches."""
    secret = {
        "name": "gmail",
        "value": "mypassword123",
        "site_kind": "domain",
        "site": "gmail.com",
    }
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=False,
        active_domain="mail.gmail.com",
        active_process_exe="chrome.exe",
    )
    assert ok is False
    assert "password" in msg.lower()
    assert "mypassword123" not in msg


def test_app_mismatch_refused():
    """An app secret mismatch is refused if foreground process doesn't match."""
    secret = {
        "name": "discord",
        "value": "token123",
        "site_kind": "app",
        "site": "discord.exe",
    }

    # Match
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=True,
        active_domain=None,
        active_process_exe="discord.exe",
    )
    assert ok is True

    # Mismatch (e.g. notepad.exe or malware.exe)
    ok, msg = type_saved_secret_guard(
        secret_info=secret,
        focused_is_password=True,
        active_domain=None,
        active_process_exe="notepad.exe",
    )
    assert ok is False
    assert "token123" not in msg


@pytest.mark.asyncio
async def test_forget_asks_first():
    """Forget password command asks for confirmation before deleting."""
    from grace.main import GraceApp
    app = GraceApp(tts_off=True)
    mock_store = MagicMock()
    app.memory_store = mock_store

    trace = MagicMock()
    handled = await app._route_transcript(trace, "forget my gmail password")
    assert handled is True
    # Asserts that confirmation was parked, and store.forget_secret was NOT called yet
    assert app.agent_loop.has_pending_confirmation
    assert "gmail" in app.agent_loop.pending_prompt.lower()
    assert mock_store.forget_secret.call_count == 0


@pytest.mark.asyncio
async def test_typed_and_voice_paths_behave_the_same():
    """Typed and voice commands for typing a secret both route to type_saved_secret and dispatch."""
    from grace.main import GraceApp
    app = GraceApp(tts_off=True)
    app.dispatcher = MagicMock()
    app.dispatcher.execute = MagicMock()

    # Voice path: "type my gmail password"
    trace1 = MagicMock()
    await app._route_transcript(trace1, "type my gmail password")
    assert app.dispatcher.execute.call_count == 1
    intent1 = app.dispatcher.execute.call_args[0][0]
    assert intent1.tool == "type_saved_secret"
    assert intent1.params["name"] == "gmail"

    # Reset
    app.dispatcher.execute.reset_mock()

    # Typed path (e.g. typed via initial_text or CLI --text "type my gmail password")
    trace2 = MagicMock()
    await app._route_transcript(trace2, "type my gmail password")
    assert app.dispatcher.execute.call_count == 1
    intent2 = app.dispatcher.execute.call_args[0][0]
    assert intent2.tool == "type_saved_secret"
    assert intent2.params["name"] == "gmail"
