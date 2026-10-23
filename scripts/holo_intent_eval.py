"""Measure the production intent route for the demo commands against live Holo.

For every line in ``bench/demo_intents.jsonl`` this runs exactly the production
intent step and nothing else: ``GemmaClient.generate_intent`` with the
production prompt, then ``IntentParser.parse``, then
``CapabilityRouter.classify``. No tool is dispatched and no desktop, mic,
speaker or screen call is made anywhere.

It prints the per-line raw model output and a confusion table over the three
routes (``agent``, the fast path, ``converse``), and writes a JSON report to
``%TEMP%\\grace-holo\\intent_eval.json`` so a fix's re-run can be diffed
against the baseline by hand.

The server is started for the run and stopped in ``finally`` - one
llama-server, one foreground command. ``--server-url`` points at an already
running llama-server instead of starting one.

``--backend gemini`` (also picked up automatically from ``USE_CLOUD_LLM``)
runs every decision through the same production ``GemmaClient`` cloud path
with no llama-server involved at all. The intent step sends text only - no
real screen data ever flows - so the R19 cloud-screenshot consent gate stays
untouched. This is how a machine without local model weights verifies live
routing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from typing import Optional

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(repo_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "src"))

from grace.intent.parser import IntentParser  # noqa: E402
from grace.intent.prompt import get_system_prompt  # noqa: E402
from grace.intent.router import CapabilityRouter, TaskComplexity  # noqa: E402
from grace.llm.gemma_client import GemmaClient  # noqa: E402

ROUTE_ORDER = ["agent", "fast_path", "converse"]


def _route_name(complexity: TaskComplexity) -> str:
    return {
        TaskComplexity.AGENTIC_GOAL: "agent",
        TaskComplexity.FAST_PATH: "fast_path",
        TaskComplexity.CONVERSATION: "converse",
    }[complexity]


def _resolve_backend(requested: str) -> str:
    """Follow --backend, or USE_CLOUD_LLM when 'auto' (same parse as Config)."""
    if requested != "auto":
        return requested
    return ("gemini"
            if os.getenv("USE_CLOUD_LLM", "").strip().lower() in ("true", "1", "yes")
            else "local")


def _build_client(url: str, backend: str, api_key: Optional[str],
                  model_name: Optional[str]) -> GemmaClient:
    """One client per (re)build; the cloud path ignores base_url entirely."""
    if api_key:
        return GemmaClient(
            base_url=url,
            api_key=api_key,
            model_name=model_name or os.getenv("GEMINI_MODEL_NAME",
                                               "gemini-3.1-flash-lite"),
            allow_cloud_screenshots=False,  # R19: this step sends text only
        )
    return GemmaClient(base_url=url, api_key=None)


def start_server(args) -> subprocess.Popen:
    cmd = [
        args.server,
        "-m", args.model,
        "--mmproj", args.mmproj,
        "-ngl", str(args.ngl),
        "-c", "8192",
        "--port", str(args.port),
        "--host", args.host,
    ]
    server_dir = os.path.dirname(os.path.abspath(args.server))
    print(f"Starting llama-server on port {args.port}...")
    return subprocess.Popen(cmd, cwd=server_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as res:
                if res.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def load_lines(path: str) -> list[dict]:
    lines: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            rec = json.loads(raw)
            rec["_line_no"] = no
            lines.append(rec)
    return lines


async def run_eval(url: str, lines: list[dict], *, backend: str = "local",
                   api_key: Optional[str] = None,
                   model_name: Optional[str] = None) -> list[dict]:
    """The production intent step only: generate -> parse -> classify.

    `generate_intent` returns None when the SSE stream or pooled connection
    dies mid-call; llama-server closes keepalive connections aggressively, so
    a dead pooled session is a transport flake, not a measurement. A None gets
    one retry per fresh session (close + rebuild the client) before the line
    is recorded as undecided.
    """
    prompt = get_system_prompt()
    parser = IntentParser()
    rows: list[dict] = []
    for rec in lines:
        text = rec["text"]
        raw_output: Optional[str] = None
        parsed_tool: Optional[str] = None
        decided: Optional[str] = None
        error: Optional[str] = None
        retries = 0
        started = time.time()

        client = _build_client(url, backend, api_key, model_name)
        try:
            for attempt in range(3):
                try:
                    raw_output = await client.generate_intent(
                        user_message=text,
                        system_prompt=prompt,
                    )
                    if raw_output is not None:
                        error = None
                        break
                    error = (
                        "generate_intent returned nothing "
                        "(stream or transport failure)"
                    )
                except Exception as e:
                    error = f"generate_intent failed: {e}"
                    raw_output = None
                retries = attempt + 1
                try:
                    await client.close()
                except Exception:
                    pass
                client = _build_client(url, backend, api_key, model_name)
                await asyncio.sleep(1.0)
        finally:
            try:
                await client.close()
            except Exception:
                pass

        if raw_output is not None:
            intent = None
            try:
                intent = parser.parse(raw_output)
                parsed_tool = intent.tool
            except Exception as e:
                error = f"intent parse failed: {e}"
            decided = _route_name(
                CapabilityRouter.classify(text, intent)
            )
        rows.append({
            "line_no": rec["_line_no"],
            "text": text,
            "expected": rec["expected"],
            "decided": decided,
            "tool": parsed_tool,
            "raw_output": raw_output,
            "error": error,
            "retries": retries,
            "elapsed_s": round(time.time() - started, 2),
        })
    return rows


def print_rows_and_table(rows: list[dict]) -> dict:
    table = {expected: {name: 0 for name in ROUTE_ORDER} for expected in ROUTE_ORDER}
    parsed_failures = 0
    print("\n--- per-line results (production intent step, live model) ---")
    for row in rows:
        expected, decided = row["expected"], row["decided"]
        if decided is None:
            parsed_failures += 1
            print(f"[{row['line_no']:>2}] {expected:>9} -> NO DECISION  "
                  f"error: {row['error']}")
            continue
        if expected in table and decided in table[expected]:
            table[expected][decided] += 1
        flag = "= " if expected == decided else "!="
        print(f"[{row['line_no']:>2}] {expected:>9} {flag} {decided:>9} "
              f"tool={str(row['tool']):<14} "
              f"raw: {_trim(row['raw_output'])}")

    print("\n--- confusion table (rows = expected, columns = decided) ---")
    header = "expected \\ decided | " + " | ".join(f"{c:>9}" for c in ROUTE_ORDER)
    print(header)
    print("-" * len(header))
    cells = {}
    for expected in ROUTE_ORDER:
        cells[expected] = {name: table[expected][name] for name in ROUTE_ORDER}
        row = " | ".join(f"{table[expected][name]:>9}" for name in ROUTE_ORDER)
        print(f"{expected:>18} | {row}")

    counts = {
        "agent": sum(cells["agent"].values()),
        "fast_path": sum(cells["fast_path"].values()),
        "converse": sum(cells["converse"].values()),
        "undecided": parsed_failures,
    }
    summary = {"counts": counts, "table": cells}
    return summary


def _trim(raw: Optional[str], limit: int = 400) -> str:
    text = (raw or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",
                        default=os.path.join(repo_root, "bench", "demo_intents.jsonl"))
    parser.add_argument("--model", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.Q4_K_M.gguf"))
    parser.add_argument("--mmproj", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.mmproj-q8_0.gguf"))
    parser.add_argument("--server", default=os.path.join(repo_root, "llama cpp", "llama-server.exe"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0,
                        help="port for the stack (0 = pick a free one)")
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--server-url", default=None,
                        help="an already-running llama-server to use instead of starting one")
    parser.add_argument("--backend", choices=("auto", "local", "gemini"),
                        default="auto",
                        help="auto follows USE_CLOUD_LLM; gemini needs GEMINI_API_KEY "
                             "and starts no local server")
    args = parser.parse_args(argv)

    if not args.port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((args.host, 0))
            args.port = s.getsockname()[1]
    url = args.server_url or f"http://{args.host}:{args.port}"

    backend = _resolve_backend(args.backend)
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    if backend == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("ERROR: backend 'gemini' selected but GEMINI_API_KEY is not set.")
            print("       (Set it in .env; .env is applied when any grace module imports.)")
            return 2
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
        print(f"Backend: cloud Gemini ({model_name}); no llama-server is started "
              f"and no real screen data is sent (R19 untouched).")

    lines = load_lines(args.dataset)
    if not lines:
        print(f"ERROR: no labelled lines in {args.dataset}")
        return 2
    print(f"{len(lines)} labelled commands from {args.dataset}")

    proc: Optional[subprocess.Popen] = None
    pre_existing = _llama_pids()
    exit_code = 0
    out_path = os.path.join(tempfile.gettempdir(), "grace-holo", "intent_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        if backend == "local" and not args.server_url:
            for label, path in (("model", args.model), ("mmproj", args.mmproj),
                                ("llama-server", args.server)):
                if not os.path.exists(path):
                    print(f"ERROR: {label} not found at: {path}")
                    return 2
            proc = start_server(args)
            if not wait_for_server(url, 120):
                print("ERROR: llama-server did not become healthy in 120s")
                return 1
            print(f"llama-server healthy at {url} (pid {proc.pid})")

        rows = asyncio.run(
            run_eval(url, lines, backend=backend, api_key=api_key,
                     model_name=model_name))
        summary = print_rows_and_table(rows)
        summary["backend"] = backend
        if backend == "local":
            summary["base_url"] = url
        summary["dataset"] = args.dataset
        summary["prompt"] = get_system_prompt()
        summary["rows"] = rows
        summary["regressions"] = [
            {"text": r["text"], "expected": r["expected"],
             "decided": r["decided"], "tool": r["tool"],
             "raw_output": r["raw_output"]}
            for r in rows
            if r["expected"] in ("fast_path", "converse")
            and r["decided"] not in (None, r["expected"])
        ]
        agent_rows = [r for r in rows if r["expected"] == "agent"]
        summary["agent_lines_ok"] = all(r["decided"] == "agent" for r in agent_rows)
        summary["agent_lines_missed"] = [
            {"text": r["text"], "decided": r["decided"], "tool": r["tool"],
             "raw_output": r["raw_output"]}
            for r in agent_rows if r["decided"] != "agent"
        ]
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\nJSON report written to {out_path}")
        print(f"agent lines routed to the agent: "
              f"{len(agent_rows) - len(summary['agent_lines_missed'])}/{len(agent_rows)}")

    finally:
        if proc is not None:
            print("\nStopping llama-server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            alive = proc.poll() is None
            print(f"llama-server stopped (pid {proc.pid}, exited: {not alive}).")
        remaining = _llama_pids()
        new_left = [pid for pid in remaining if pid not in pre_existing]
        if new_left:
            print(f"WARNING: llama-server processes still running: {new_left}")
        else:
            print("No llama-server processes remain.")

    return exit_code


def _llama_pids() -> list[int]:
    pids: list[int] = []
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            try:
                if (proc.info.get("name") or "").lower() == "llama-server.exe":
                    pids.append(proc.pid)
            except Exception:
                continue
    except Exception:
        pass
    return pids


if __name__ == "__main__":
    raise SystemExit(main())
