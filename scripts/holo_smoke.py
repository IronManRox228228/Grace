"""Holo 3.1-4B smoke drive, headless.

Boots llama-server on a free port, runs a synthetic visual-grounding request
and a planner request against it, and reports whether the server started,
time to first token, tokens per second, and whether ``parse_planned_step``
accepts the planner reply.

Copied from the buildathon smoke script (``%TEMP%/grace-holo/smoke_test.py``).
The only changes are that the hardcoded paths and thread count are arguments,
and that the planner request streams so time-to-first-token is measured on
the real model path instead of being absent.

Usage::

    python scripts/holo_smoke.py \
        --model  "C:\\Users\\...\\Downloads\\Holo-3.1-4B.Q4_K_M.gguf" \
        --mmproj "C:\\Users\\...\\Downloads\\Holo-3.1-4B.mmproj-q8_0.gguf" \
        --server "llama cpp\\llama-server.exe"
"""

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from typing import Optional

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_repo_src() -> Optional[str]:
    """Add the repo's src to sys.path so the planner prompt can be imported."""
    src = os.path.join(repo_root, "src")
    if not os.path.isdir(src):
        return None
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.Q4_K_M.gguf"))
    parser.add_argument("--mmproj", default=os.path.join(
        os.path.expanduser("~"), "Downloads", "Holo-3.1-4B.mmproj-q8_0.gguf"))
    parser.add_argument("--server", default=os.path.join(repo_root, "llama cpp", "llama-server.exe"))
    parser.add_argument("--ngl", type=int, default=999)
    args = parser.parse_args(argv)

    for label, path in (("model", args.model), ("mmproj", args.mmproj), ("llama-server", args.server)):
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at: {path}")
            return 2
        print(f"{label}: {path}")

    default_repo_src()
    try:
        from grace.agent.planner import get_planner_system_prompt, parse_planned_step
    except ImportError as exc:
        print(f"ERROR: could not import the planner from the repo: {exc}")
        return 2

    def get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def make_synthetic_image():
        from PIL import Image, ImageDraw

        # 1280x720 white canvas
        img = Image.new("RGB", (1280, 720), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Button 1: "Save" at (100, 100, 260, 160)
        draw.rectangle([100, 100, 260, 160], fill=(200, 200, 200), outline=(0, 0, 0), width=2)
        draw.text((150, 120), "Save", fill=(0, 0, 0))

        # Button 2: "Submit" at (500, 300, 700, 380)
        draw.rectangle([500, 300, 700, 380], fill=(70, 130, 180), outline=(0, 0, 0), width=2)
        draw.text((580, 330), "Submit", fill=(255, 255, 255))

        # Button 3: "Cancel" at (900, 500, 1060, 560)
        draw.rectangle([900, 500, 1060, 560], fill=(220, 20, 60), outline=(0, 0, 0), width=2)
        draw.text((950, 520), "Cancel", fill=(255, 255, 255))

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    png_bytes = make_synthetic_image()
    b64_img = base64.b64encode(png_bytes).decode("utf-8")

    port = get_free_port()
    server = os.path.abspath(args.server)
    server_dir = os.path.dirname(server)
    cmd = [
        server,
        "-m", os.path.abspath(args.model),
        "--mmproj", os.path.abspath(args.mmproj),
        "-ngl", str(args.ngl),
        "-c", "8192",
        "--port", str(port),
        "--host", "127.0.0.1",
    ]
    print(f"Starting llama-server on port {port}...")
    print(f"  cmd: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=server_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    exit_code = 0
    server_started = False
    try:
        url = f"http://127.0.0.1:{port}"
        print("Waiting for health check...")
        ready = False
        start_wait = time.time()
        while time.time() - start_wait < 120:
            if proc.poll() is not None:
                out = proc.stdout.read().decode("utf-8", errors="replace")[-1500:]
                err = proc.stderr.read().decode("utf-8", errors="replace")[-1500:]
                print(f"ERROR: llama-server exited early (code {proc.returncode})")
                if out:
                    print(f"  stdout tail:\n{out}")
                if err:
                    print(f"  stderr tail:\n{err}")
                return 1
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(1)
        if not ready:
            print("ERROR: llama-server did not become healthy in 120s")
            return 1

        server_started = True
        print(f"llama-server healthy after {time.time() - start_wait:.1f}s "
              f"(proc pid {proc.pid})")

        # Test 1: visual grounding on the synthetic image (non-streaming, so
        # llama.cpp's `timings` object gives exact prompt/eval token counts).
        grounding_system_prompt = (
            "You are a GUI grounding model. You are shown a screenshot and asked to locate a single element.\n"
            "Respond with the coordinates of the center of the element.\n"
            "Format: Thought: <brief description>\nAction: click(point=(x, y))\n"
            "If the element is not visible, respond with Action: wait()."
        )
        target = "Submit button"
        messages = [
            {"role": "system", "content": grounding_system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Locate this element: {target}"},
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

        print("\n--- Sending visual grounding request ---")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as resp:
            ground_resp = json.loads(resp.read().decode("utf-8"))
        ground_latency = (time.perf_counter() - t0) * 1000
        ground_content = ground_resp["choices"][0]["message"]["content"]
        print(f"Grounding response in {ground_latency:.1f}ms:\n{ground_content}")
        timings = ground_resp.get("timings") or {}
        if timings.get("predicted_n"):
            tok_s = timings["predicted_n"] / (timings["predicted_ms"] / 1000.0)
            print(
                "Grounding timings: prompt_n={prompt_n} prompt_ms={prompt_ms:.1f} "
                "predicted_n={predicted_n} predicted_ms={predicted_ms:.1f} -> {tok_s:.1f} tok/s".format(
                    tok_s=tok_s, **{k: timings.get(k, 0) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}
                )
            )

        # Test 2: planner tool-call, streamed, so time to first token is real.
        planner_system = get_planner_system_prompt()
        planner_user = (
            "### Goal\nOpen the calculator\n\n"
            "### Screen\nactive_window: Desktop\ncontrols:\n- [1] button 'Start'\n- [2] button 'Taskbar'\n\n"
            "### History\nNo steps taken yet.\n\n"
            "### Last step result\nNone"
        )
        planner_messages = [
            {"role": "system", "content": planner_system},
            {"role": "user", "content": planner_user},
        ]
        stream_req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps({
                "messages": planner_messages,
                "temperature": 0.0,
                "max_tokens": 512,
                "stream": True,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        print("\n--- Sending planner tool-call request (streaming) ---")
        t0 = time.perf_counter()
        ttft = None
        chunks: list[str] = []
        with urllib.request.urlopen(stream_req, timeout=120) as resp:
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
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        chunks.append(text)
        planner_latency = (time.perf_counter() - t0) * 1000
        planner_content = "".join(chunks)
        if ttft is None:
            print("ERROR: planner stream produced no content chunks")
            exit_code = 1
        else:
            elapsed_total = planner_latency / 1000.0
            gen_seconds = max(elapsed_total - ttft / 1000.0, 1e-6)
            print(f"Time to first token: {ttft:.1f}ms")
            print(f"Planner stream: {len(chunks)} chunks in {planner_latency:.1f}ms "
                  f"(~{len(chunks) / gen_seconds:.1f} chunks/s; a chunk is roughly one token)")
            print(f"Planner response:\n{planner_content}")

        parsed_step = parse_planner(planner_content)
        if parsed_step is None:
            print("parse_planned_step: NOT accepted")
            exit_code = 1
        else:
            print(f"parse_planned_step: ACCEPTED. action={parsed_step.action!r}, "
                  f"params={parsed_step.params}")

        if not server_started:
            exit_code = 1

    finally:
        print("\nStopping llama-server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        stopped_cleanly = proc.poll() is not None
        print(f"llama-server stopped (exited: {stopped_cleanly}, pid {proc.pid}).")

    print(f"\nRESULT: llama-server_started={server_started}")
    return exit_code


def parse_planner(planner_content: str):
    from grace.agent.planner import parse_planned_step

    return parse_planned_step(planner_content)


if __name__ == "__main__":
    raise SystemExit(main())
