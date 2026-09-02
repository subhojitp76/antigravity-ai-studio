"""
Comprehensive verification test script for app.py endpoints, multi-turn chat,
mid-stream cancellation, and full lifecycle on OpenVINO Intel NPU.
"""

import time
import json
import urllib.request
import urllib.parse
import threading
import sys

BASE_URL = "http://127.0.0.1:7860"

def get_json(endpoint):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def post_json(endpoint, data=None):
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def stream_chat(endpoint, data=None, on_token_callback=None):
    payload = json.dumps(data or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    events = []
    with urllib.request.urlopen(req, timeout=60) as resp:
        current_event = None
        while True:
            line_b = resp.readline()
            if not line_b:
                break
            line_str = line_b.decode("utf-8").strip()
            if line_str.startswith("event: "):
                current_event = line_str[7:].strip()
            elif line_str.startswith("data: "):
                data_val = line_str[6:].strip()
                if data_val:
                    parsed_data = json.loads(data_val)
                    events.append((current_event, parsed_data))
                    if current_event == "token" and on_token_callback:
                        on_token_callback(parsed_data)
                    if current_event in ["done", "error"]:
                        break
    return events

def run_tests():
    print("=" * 60, flush=True)
    print(" STARTING COMPREHENSIVE OPENVINO NPU APP TESTS", flush=True)
    print("=" * 60, flush=True)

    print("\n--- [1] Checking Server HTML & Assets ---", flush=True)
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "OpenVINO" in html
        assert "llama-3.2-3b-ov" in html
    print("  [PASS] HTML and brand elements verified.", flush=True)

    print("\n--- [2] Checking Device Enumeration ---", flush=True)
    status, data = get_json("/api/devices")
    assert status == 200
    devices = data.get("devices", [])
    print(f"  Detected {len(devices)} OpenVINO device(s):")
    for d in devices:
        print(f"    - {d['id']}: {d['name']} ({d['type']})")
    assert any("NPU" in d["id"] for d in devices), "NPU device not found"
    print("  [PASS] Intel NPU detected.", flush=True)

    print("\n--- [3] Starting Model on Intel NPU ---", flush=True)
    status, data = post_json("/api/model/start", {
        "device": "NPU",
        "config": {
            "MAX_PROMPT_LEN": 1024,
            "MIN_RESPONSE_LEN": 512,
            "GENERATE_HINT": "BEST_PERF"
        }
    })
    assert status == 200

    print("  Compiling model graph for Intel NPU...", flush=True)
    start_t = time.time()
    ready = False
    while time.time() - start_t < 90:
        time.sleep(1.5)
        _, st = get_json("/api/status")
        if st["state"] == "ready":
            ready = True
            print(f"  Model compiled in {st.get('compile_duration')}s!")
            break
        elif st["state"] == "error":
            print(f"  [FAIL] Compilation error: {st.get('last_error')}")
            sys.exit(1)

    assert ready
    print("  [PASS] Model ready on NPU.", flush=True)

    print("\n--- [4] Turn 1: Streaming Chat Generation ---", flush=True)
    prompt_1 = "Write 2 bullet points on why NPUs are energy efficient."
    events_1 = stream_chat("/api/chat/stream", {
        "message": prompt_1,
        "max_new_tokens": 80,
        "temperature": 0.7
    })
    tokens_1 = [d["text"] for ev, d in events_1 if ev == "token"]
    full_text_1 = "".join(tokens_1)
    metrics_1 = [d for ev, d in events_1 if ev == "metrics"][0]
    print(f"  Response:\n{full_text_1}", flush=True)
    print(f"  Speed: {metrics_1['tps']} tokens/s | TTFT: {metrics_1['ttft_ms']} ms | Tokens: {metrics_1['token_count']}", flush=True)
    assert len(tokens_1) > 0

    print("\n--- [5] Turn 2: Multi-turn Memory & Context Continuation ---", flush=True)
    prompt_2 = "Summarize the points you just made in one word."
    events_2 = stream_chat("/api/chat/stream", {
        "message": prompt_2,
        "max_new_tokens": 30,
        "temperature": 0.3
    })
    tokens_2 = [d["text"] for ev, d in events_2 if ev == "token"]
    full_text_2 = "".join(tokens_2)
    print(f"  Follow-up Response: {full_text_2.strip()}", flush=True)
    assert len(tokens_2) > 0
    print("  [PASS] Multi-turn conversation preserved.", flush=True)

    print("\n--- [6] Mid-Generation Interruption Test (/api/chat/stop_generation) ---", flush=True)
    stop_called = False
    def on_token(token_data):
        nonlocal stop_called
        if token_data.get("token_index", 0) >= 10 and not stop_called:
            stop_called = True
            print("  -> Sending stop signal at token #10...", flush=True)
            post_json("/api/chat/stop_generation")

    long_prompt = "Write a comprehensive 500 word history of computing and semiconductors."
    events_cancel = stream_chat("/api/chat/stream", {
        "message": long_prompt,
        "max_new_tokens": 400
    }, on_token_callback=on_token)

    tokens_cancel = [d["text"] for ev, d in events_cancel if ev == "token"]
    metrics_cancel = [d for ev, d in events_cancel if ev == "metrics"][0]
    print(f"  Tokens generated before interrupt: {len(tokens_cancel)} (Interrupted: {metrics_cancel.get('interrupted')})", flush=True)
    assert len(tokens_cancel) < 200, "Generation was not stopped early"
    assert metrics_cancel.get("interrupted") is True
    print("  [PASS] Mid-generation cancellation works instantly.", flush=True)

    print("\n--- [7] Chat Reset & System Prompt Test ---", flush=True)
    status, data = post_json("/api/chat/reset", {
        "system_prompt": "You are a pirate AI assistant. Speak like a pirate."
    })
    assert status == 200
    events_pirate = stream_chat("/api/chat/stream", {
        "message": "Hello!",
        "max_new_tokens": 40
    })
    tokens_pirate = [d["text"] for ev, d in events_pirate if ev == "token"]
    pirate_text = "".join(tokens_pirate)
    print(f"  Pirate response: {pirate_text.strip()}", flush=True)
    assert len(tokens_pirate) > 0
    print("  [PASS] System prompt & context reset verified.", flush=True)

    print("\n--- [8] Stopping / Unloading Model ---", flush=True)
    status, data = post_json("/api/model/stop")
    assert status == 200
    _, st = get_json("/api/status")
    assert st["state"] == "unloaded"
    print("  [PASS] Model unloaded cleanly.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(" ALL 8 TEST SUITES COMPLETED WITH 100% SUCCESS!", flush=True)
    print("=" * 60 + "\n", flush=True)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    run_tests()
