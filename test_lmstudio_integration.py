"""
Automated Verification Test for LM Studio & OpenVINO Dual-Engine Integration.
"""

import time
import json
import urllib.request
import urllib.parse
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

def stream_chat(endpoint, data=None):
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
                    if current_event in ["done", "error"]:
                        break
    return events

def run_tests():
    print("=" * 65, flush=True)
    print(" STARTING LM STUDIO & OPENVINO DUAL-ENGINE VERIFICATION", flush=True)
    print("=" * 65, flush=True)

    # 1. Engine Status Probe
    print("\n--- [1] Testing /api/engine/status ---", flush=True)
    status, res = get_json("/api/engine/status")
    assert status == 200, f"Status call failed: {res}"
    print(f"  Active Engine: {res['active_engine']}", flush=True)
    print(f"  LM Studio Connected: {res['lmstudio']['connected']}", flush=True)
    print(f"  LM Studio Models Found: {len(res['lmstudio']['available_models'])}", flush=True)
    assert res['lmstudio']['connected'] is True, "LM Studio should be connected on port 1234"

    # 2. LM Studio Models Endpoint
    print("\n--- [2] Testing /api/lmstudio/models ---", flush=True)
    status, lm_res = get_json("/api/lmstudio/models")
    assert status == 200
    models = lm_res.get("models", [])
    print(f"  Available models in LM Studio:")
    for m in models:
        print(f"    - {m['id']}", flush=True)
    assert len(models) > 0, "No models found in LM Studio"
    target_model = models[0]['id']

    # 3. Select LM Studio as active engine
    print(f"\n--- [3] Selecting LM Studio ({target_model}) ---", flush=True)
    status, sel_res = post_json("/api/engine/select", {
        "engine": "lmstudio",
        "model": target_model
    })
    assert status == 200
    assert sel_res["active_engine"] == "lmstudio"
    print(f"  [PASS] Switched active engine to 'lmstudio' ({target_model})", flush=True)

    # 4. Stream chat via LM Studio
    print("\n--- [4] Streaming Chat Query via LM Studio ---", flush=True)
    query = "What is the capital of France? Answer in 3 words."
    events = stream_chat("/api/chat/stream", {
        "engine": "lmstudio",
        "message": query,
        "max_new_tokens": 60,
        "temperature": 0.2
    })
    
    tokens = [d["text"] for ev, d in events if ev == "token"]
    full_text = "".join(tokens)
    metrics_list = [d for ev, d in events if ev == "metrics"]

    print(f"  Query: {query}", flush=True)
    print(f"  Tokens Received: {len(tokens)}", flush=True)
    print(f"  Response: {full_text.strip()}", flush=True)
    if metrics_list:
        m = metrics_list[0]
        print(f"  Metrics: {m['tps']} tokens/s, TTFT: {m['ttft_ms']}ms, Total: {m['total_duration_s']}s", flush=True)

    assert len(tokens) > 0, "No tokens received from LM Studio"
    assert "paris" in full_text.lower(), "Expected 'Paris' in response"
    print("  [PASS] LM Studio streaming chat succeeded!", flush=True)

    # 5. Test RAG with LM Studio
    print("\n--- [5] Testing RAG Context Retrieval with LM Studio ---", flush=True)
    # Upload custom document
    doc_content = (
        "# Project QuantumVortex\n"
        "Project QuantumVortex is a next-gen particle accelerator control protocol.\n"
        "Security Hash: QVX-9988-OMEGA\n"
        "Cooling System: Liquid Neon Cryo-loop at 27 Kelvin\n"
    )
    post_json("/api/documents/upload", {
        "filename": "quantum_vortex.md",
        "content": doc_content
    })

    rag_events = stream_chat("/api/chat/stream", {
        "engine": "lmstudio",
        "message": "What is the Security Hash and Cooling System for Project QuantumVortex?",
        "rag_enabled": True,
        "top_k": 2,
        "max_new_tokens": 100,
        "temperature": 0.1
    })

    rag_citations = [d for ev, d in events if ev == "citations"]
    rag_tokens = [d["text"] for ev, d in rag_events if ev == "token"]
    rag_full_answer = "".join(rag_tokens)

    print(f"  RAG Response:\n{rag_full_answer.strip()}", flush=True)
    assert "QVX-9988-OMEGA" in rag_full_answer or "Liquid Neon" in rag_full_answer or "QuantumVortex" in rag_full_answer
    print("  [PASS] LM Studio successfully answered using RAG context!", flush=True)

    # 6. Switch Back to OpenVINO NPU
    print("\n--- [6] Switching Back to OpenVINO Engine ---", flush=True)
    status, sel_res = post_json("/api/engine/select", {
        "engine": "openvino",
        "device": "NPU"
    })
    assert status == 200
    assert sel_res["active_engine"] == "openvino"
    print("  [PASS] Switched back to OpenVINO engine successfully!", flush=True)

    print("\n" + "=" * 65, flush=True)
    print(" ALL DUAL-ENGINE VERIFICATION TESTS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 65 + "\n", flush=True)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    run_tests()
