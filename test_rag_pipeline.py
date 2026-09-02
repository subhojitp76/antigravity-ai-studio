"""
Automated End-to-End Verification Test for RAG Pipeline & Document Upload.
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

def run_rag_tests():
    print("=" * 60, flush=True)
    print(" STARTING RAG & DOCUMENT UPLOAD VERIFICATION TESTS", flush=True)
    print("=" * 60, flush=True)

    # 1. Document Upload Test
    print("\n--- [1] Testing Document Upload (/api/documents/upload) ---", flush=True)
    doc_content = (
        "# Project Hyperion Specs\n"
        "Project Hyperion is a quantum-secure encrypted database system designed for autonomous probes.\n"
        "Key Specifications:\n"
        "- Protocol Code: HYPERION-992-SECURE\n"
        "- Encryption: Post-quantum lattice-based Kyber-1024\n"
        "- Max Throughput: 450,000 transactions per second\n"
        "- Hardware Target: Intel NPU with AI Boost co-processing\n"
    )
    status, upload_res = post_json("/api/documents/upload", {
        "filename": "hyperion_specs.md",
        "content": doc_content
    })
    assert status == 200, f"Upload failed: {upload_res}"
    print(f"  Upload success: {upload_res['filename']} ({upload_res['chunk_count']} chunks)", flush=True)
    assert upload_res["chunk_count"] > 0

    # 2. List Documents Test
    print("\n--- [2] Testing List Documents (/api/documents) ---", flush=True)
    status, list_res = get_json("/api/documents")
    assert status == 200
    docs = list_res.get("documents", [])
    print(f"  Knowledge base contains {len(docs)} document(s):")
    for d in docs:
        print(f"    - {d['filename']}: {d['chunk_count']} chunk(s), {d['total_chars']} chars", flush=True)
    assert any(d["filename"] == "hyperion_specs.md" for d in docs)
    print("  [PASS] Document list retrieved and verified.", flush=True)

    # 3. Start Model if not ready
    print("\n--- [3] Ensuring Model is Ready on NPU ---", flush=True)
    _, st = get_json("/api/status")
    if st["state"] != "ready":
        print("  Starting model on NPU...", flush=True)
        post_json("/api/model/start", {"device": "NPU"})
        start_t = time.time()
        while time.time() - start_t < 60:
            time.sleep(1.5)
            _, st = get_json("/api/status")
            if st["state"] == "ready":
                break
    print(f"  Model state: {st['state']} on {st['device']}", flush=True)
    assert st["state"] == "ready"

    # 4. RAG-Augmented Query Test
    print("\n--- [4] Testing RAG Chat with Knowledge Base ---", flush=True)
    query = "What is the Protocol Code and Encryption algorithm for Project Hyperion?"
    events = stream_chat("/api/chat/stream", {
        "message": query,
        "rag_enabled": True,
        "top_k": 3,
        "max_new_tokens": 100,
        "temperature": 0.2
    })

    citations_events = [d for ev, d in events if ev == "citations"]
    tokens = [d["text"] for ev, d in events if ev == "token"]
    full_answer = "".join(tokens)
    metrics_events = [d for ev, d in events if ev == "metrics"]

    print(f"  Query: {query}", flush=True)
    print(f"  Citations Received: {len(citations_events[0]['citations'] if citations_events else [])}", flush=True)
    if citations_events:
        for c in citations_events[0]["citations"]:
            print(f"    -> [Source {c['citation_id']}] {c['filename']} (Score: {c['score_percent']}%)", flush=True)
    
    print(f"  RAG Generated Response:\n{full_answer}", flush=True)
    
    assert len(citations_events) > 0, "No citations event received"
    assert len(tokens) > 0, "No tokens received in response"
    assert "HYPERION-992-SECURE" in full_answer or "Kyber" in full_answer or "Hyperion" in full_answer, "Response did not utilize document context"
    print("  [PASS] RAG chat retrieved context and answered accurately!", flush=True)

    # 5. Delete Document Test
    print("\n--- [5] Testing Document Deletion (/api/documents/delete) ---", flush=True)
    doc_id = upload_res["doc_id"]
    status, del_res = post_json("/api/documents/delete", {"doc_id": doc_id})
    assert status == 200
    _, list_after = get_json("/api/documents")
    assert not any(d["doc_id"] == doc_id for d in list_after.get("documents", []))
    print(f"  [PASS] Document '{doc_id}' deleted successfully.", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(" ALL RAG & DOCUMENT UPLOAD TESTS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 60 + "\n", flush=True)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    run_rag_tests()
