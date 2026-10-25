"""Live-model demo gate for Grace.

Starts llama-server with Holo 3.1-4B once, runs:
1. Smoke tests: visual grounding and streaming planner tool-call.
2. Intent evaluation: verifies all 'agent'-labelled intents route to agent.
3. Tape drive: runs agentic tapes (agentic_two_step, agentic_read_pdf,
   agentic_youtube_music, agentic_pdf_to_notes).
   A tape verdict is PASS only on `status: ok` with the goal reached.
   `no_progress` or an honest failure is `NOT PROVEN` (tapes run against a
   stateful sim desktop, a PASS means the sim reached its goal state, and only
   a real-desktop run proves that real apps behave like the sim).

Stops the server in finally, prints one PASS/FAIL/NOT PROVEN table, and exits
non-zero on FAIL (exit code 1) or NOT PROVEN (exit code 2).
All desktop, mic, and speaker edges stay mocked.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from io import BytesIO
from typing import Optional

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if os.path.join(repo_root, "src") not in sys.path:
    sys.path.insert(0, os.path.join(repo_root, "src"))

from grace.agent.planner import get_planner_system_prompt, parse_planned_step
from scripts.holo_intent_eval import load_lines, run_eval
from scripts.holo_tape_drive import drive_tape


def get_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def make_synthetic_image() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([500, 300, 700, 380], fill=(70, 130, 180), outline=(0, 0, 0), width=2)
    draw.text((580, 330), "Submit", fill=(255, 255, 255))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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
    return subprocess.Popen(
        cmd, cwd=server_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


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


def run_smoke_grounding(url: str) -> bool:
    print("\n--- [1/3] Smoke: Visual Grounding ---")
    png_bytes = make_synthetic_image()
    b64_img = base64.b64encode(png_bytes).decode("utf-8")
    grounding_system_prompt = (
        "You are a GUI grounding model. You are shown a screenshot and asked to locate a single element.\n"
        "Respond with the coordinates of the center of the element.\n"
        "Format: Thought: <brief description>\nAction: click(point=(x, y))\n"
        "If the element is not visible, respond with Action: wait()."
    )
    messages = [
        {"role": "system", "content": grounding_system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": "Locate this element: Submit button"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
        ]},
    ]
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": False,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = (time.perf_counter() - t0) * 1000
        content = data["choices"][0]["message"]["content"]
        print(f"Grounding latency: {latency:.1f}ms")
        print(f"Grounding response: {content.strip()}")
        return "Action: click(" in content or "click(point=" in content
    except Exception as e:
        print(f"Grounding request failed: {e}")
        return False


def run_smoke_planner(url: str) -> bool:
    print("\n--- [1/3] Smoke: Planner Tool-call ---")
    planner_system = get_planner_system_prompt()
    planner_user = (
        "### Goal\nOpen the calculator\n\n"
        "### Screen\nactive_window: Desktop\ncontrols:\n- [1] button 'Start'\n- [2] button 'Taskbar'\n\n"
        "### History\nNo steps taken yet.\n\n"
        "### Last step result\nNone"
    )
    messages = [
        {"role": "system", "content": planner_system},
        {"role": "user", "content": planner_user},
    ]
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps({
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
            "stream": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_json = line[6:].strip()
                if data_json == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_json)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        chunks.append(delta)
        content = "".join(chunks)
        print(f"Planner response:\n{content.strip()}")
        parsed = parse_planned_step(content)
        if parsed is not None:
            print(f"parse_planned_step: ACCEPTED (action={parsed.action!r})")
            return True
        print("parse_planned_step: NOT accepted")
        return False
    except Exception as e:
        print(f"Planner smoke failed: {e}")
        return False


def run_intents_gate(url: str, dataset_path: str) -> tuple[bool, int, int]:
    print("\n--- [2/3] Intent Evaluation ---")
    lines = load_lines(dataset_path)
    rows = asyncio.run(run_eval(url, lines, backend="local"))
    agent_rows = [r for r in rows if r["expected"] == "agent"]
    routed_agent = sum(1 for r in agent_rows if r["decided"] == "agent")
    total_agent = len(agent_rows)
    all_agent_ok = routed_agent == total_agent

    print(f"\nAgent intent routing: {routed_agent}/{total_agent} correctly routed to agent")
    if not all_agent_ok:
        for r in agent_rows:
            if r["decided"] != "agent":
                print(f"  MISSED [{r['line_no']}] text={r['text']!r} -> decided={r['decided']!r} (tool={r['tool']!r})")
    return all_agent_ok, routed_agent, total_agent


SIM_DESKTOP_NOTE = (
    "Tapes run against a stateful sim desktop, a PASS means the sim reached "
    "its goal state, and only a real-desktop run proves that real apps behave like the sim."
)
FROZEN_SCREEN_NOTE = SIM_DESKTOP_NOTE


def resolve_tape_path(name: str) -> str:
    """Find a tape directory in the permanent corpus or generated task tapes."""
    candidates = [
        os.path.join(repo_root, "corpus", name),
        os.path.join(tempfile.gettempdir(), "sim-tapes-demo", "tapes", name),
        os.path.join(tempfile.gettempdir(), "grace-demo", "tapes", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(repo_root, "corpus", name)


def evaluate_tape_verdict(res: Optional[dict]) -> str:
    """Evaluate a tape drive result dictionary into PASS, NOT PROVEN, or FAIL.

    - A tape verdict is PASS only on `status: ok` with the goal reached (within step cap).
    - `no_progress` or an honest failure is `NOT PROVEN`.
    - Stuck loop (>12 steps), unhandled exception, crash, or error status is `FAIL`.
    """
    if not isinstance(res, dict):
        return "FAIL"

    status = res.get("goal_status")
    steps = res.get("steps") or []
    final_resp = res.get("final_response")
    sim_goal_reached = res.get("sim_goal_reached", True)

    # Stuck loop / runaway exceeding step cap
    if len(steps) > 12:
        return "FAIL"

    # Only PASS if status is ok, goal was completed, and sim reached goal state
    if status == "ok" and res.get("ok", True) and sim_goal_reached:
        return "PASS"

    # Planner claimed done early, but sim is not in goal state -> NOT PROVEN
    if status == "ok" and not sim_goal_reached:
        return "NOT PROVEN"

    # no_progress or honest failure
    if status == "no_progress":
        return "NOT PROVEN"
    if status in ("failed", "budget_exceeded"):
        return "NOT PROVEN"
    if bool(final_resp) and status is not None and status != "error":
        return "NOT PROVEN"

    return "FAIL"


def compute_overall_status(summary: dict[str, str]) -> tuple[str, int]:
    """Compute overall status string and exit code from check summary.

    - Any FAIL -> ('FAIL', 1)
    - Any NOT PROVEN -> ('NOT PROVEN', 2)
    - All PASS -> ('PASS', 0)
    """
    statuses = list(summary.values())
    if any(s == "FAIL" for s in statuses):
        return "FAIL", 1
    if any(s == "NOT PROVEN" for s in statuses):
        return "NOT PROVEN", 2
    return "PASS", 0


def extract_raw_planner_outputs(res: dict) -> list[str]:
    """Extract raw planner output strings from tape drive result."""
    outputs: list[str] = []
    for r in (res.get("llm_records") or []):
        if isinstance(r, dict):
            call = r.get("call", "")
            if call == "generate_text" or ("intent" not in call and r.get("raw")):
                raw = r.get("raw")
                if raw and isinstance(raw, str):
                    outputs.append(raw)
        elif isinstance(r, str):
            outputs.append(r)
    return outputs


def run_tapes_gate(url: str, tapes: list[str], args) -> dict[str, str]:
    print("\n--- [3/3] Tape Drives ---")
    print(f"Notice: {FROZEN_SCREEN_NOTE}")
    results: dict[str, str] = {}
    temp_enrollment = os.path.join(tempfile.gettempdir(), "grace-demo-gate", "enrollment")
    os.makedirs(temp_enrollment, exist_ok=True)

    for tape_path in tapes:
        tape_name = os.path.basename(os.path.normpath(tape_path))
        print(f"\nDriving tape: {tape_name}...")
        try:
            res = asyncio.run(drive_tape(
                tape_path, args, temp_enrollment,
                intent_from_tape=False, backend="local"
            ))
            status = res.get("goal_status")
            final_resp = res.get("final_response")
            steps = res.get("steps") or []
            dispatches = res.get("dispatches") or []
            print(f"  final status: {status}")
            print(f"  planner steps: {len(steps)}")
            print(f"  dispatches: {len(dispatches)}")
            print(f"  final_response: {final_resp!r}")

            raw_outputs = extract_raw_planner_outputs(res)
            last_2 = raw_outputs[-2:]
            print("  last 2 raw planner outputs:")
            if not last_2:
                print("    (none)")
            else:
                for idx, out in enumerate(last_2, 1):
                    lines = out.strip().splitlines()
                    if len(lines) == 1:
                        print(f"    [{idx}] {lines[0]}")
                    else:
                        print(f"    [{idx}] {lines[0]}")
                        for line in lines[1:]:
                            print(f"        {line}")

            verdict = evaluate_tape_verdict(res)
            results[tape_name] = verdict
            print(f"  verdict: {verdict}")
        except Exception as e:
            print(f"  error driving {tape_name}: {e}")
            results[tape_name] = "FAIL"

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.Q4_K_M.gguf"))
    parser.add_argument("--mmproj", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.mmproj-q8_0.gguf"))
    parser.add_argument("--server", default=os.path.join(repo_root, "llama cpp", "llama-server.exe"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--server-url", default=None)
    args = parser.parse_args(argv)

    for label, path in (("model", args.model), ("mmproj", args.mmproj), ("llama-server", args.server)):
        if not args.server_url and not os.path.exists(path):
            print(f"ERROR: {label} not found at: {path}")
            return 1

    if not args.port:
        args.port = get_free_port(args.host)
    url = args.server_url or f"http://{args.host}:{args.port}"

    proc: Optional[subprocess.Popen] = None
    pre_existing = _llama_pids()
    overall_status = "FAIL"
    exit_code = 1
    summary: dict[str, str] = {}

    try:
        if not args.server_url:
            proc = start_server(args)
            if not wait_for_server(url, 120):
                print("ERROR: llama-server did not become healthy in 120s")
                return 1
            print(f"llama-server healthy at {url} (pid {proc.pid})")

        # 1. Smoke tests
        smoke_ground = run_smoke_grounding(url)
        summary["Smoke Grounding"] = "PASS" if smoke_ground else "FAIL"

        smoke_plan = run_smoke_planner(url)
        summary["Smoke Planner Parse"] = "PASS" if smoke_plan else "FAIL"

        # 2. Intent routing eval
        dataset_path = os.path.join(repo_root, "bench", "demo_intents.jsonl")
        intent_ok, agent_n, agent_total = run_intents_gate(url, dataset_path)
        summary[f"Agent Intent Routing ({agent_n}/{agent_total})"] = "PASS" if intent_ok else "FAIL"

        # 3. Tape drives
        tapes = [
            resolve_tape_path("agentic_two_step"),
            resolve_tape_path("agentic_read_pdf"),
            resolve_tape_path("agentic_youtube_music"),
            resolve_tape_path("agentic_pdf_to_notes"),
        ]
        tape_results = run_tapes_gate(url, tapes, args)
        for tname, verdict in tape_results.items():
            summary[f"Tape Drive ({tname})"] = verdict

        overall_status, exit_code = compute_overall_status(summary)

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

    print("\n" + "="*50)
    print("           LIVE-MODEL DEMO GATE RESULTS          ")
    print("="*50)
    for check, res in summary.items():
        print(f"  {check:<40} : {res}")
    print("="*50)
    print(f"OVERALL GATE STATUS: {overall_status}")
    print("="*50)
    print(f"Notice: {FROZEN_SCREEN_NOTE}")
    print("="*50)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
