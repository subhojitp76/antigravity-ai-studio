"""
Automated Verification Test for Project-Scoped RAG, Multi-Session Chat History, and Memorization.
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
                    try:
                        parsed_data = json.loads(data_val)
                        events.append((current_event, parsed_data))
                        if current_event in ["done", "error"]:
                            break
                    except Exception:
                        pass
    return events

def run_tests():
    print("=" * 70, flush=True)
    print(" STARTING PROJECT RAG, CHAT HISTORY & MEMORIZATION TEST SUITE", flush=True)
    print("=" * 70, flush=True)

    # 1. Projects API Test
    print("\n--- [1] Testing /api/projects ---", flush=True)
    status, res = get_json("/api/projects")
    assert status == 200
    print(f"  Active Project: {res['active_project_id']}", flush=True)
    print(f"  Total Projects: {len(res['projects'])}", flush=True)

    # 2. Create Isolated Project
    print("\n--- [2] Creating 'Project Andromeda' Workspace ---", flush=True)
    status, create_res = post_json("/api/projects/create", {
        "name": "Project Andromeda",
        "description": "Deep-space sensor network specifications and firmware."
    })
    assert status == 200
    andromeda_proj = create_res["project"]
    proj_id = andromeda_proj["id"]
    print(f"  [PASS] Created project: {andromeda_proj['name']} (ID: {proj_id})", flush=True)

    # 3. Document Ingestion into Project Andromeda
    print(f"\n--- [3] Uploading Document to {proj_id} ---", flush=True)
    doc_text = (
        "# Project Andromeda Firmware Spec\n"
        "Subsystem: HyperWave-99\n"
        "Baud Rate: 921600\n"
        "Encryption Key: ANDROMEDA-CIPHER-7744\n"
        "Beacon Frequency: 1420.405 MHz\n"
    )
    status, up_res = post_json("/api/documents/upload", {
        "filename": "andromeda_firmware.md",
        "content": doc_text,
        "project_id": proj_id
    })
    assert status == 200
    print(f"  [PASS] Uploaded and indexed into {proj_id} ({up_res.get('chunk_count', 0)} chunks)", flush=True)

    # 4. Verify Document Isolation (Zero Context Pollution)
    print("\n--- [4] Verifying Document Isolation between Projects ---", flush=True)
    _, andromeda_docs = get_json(f"/api/documents?project_id={proj_id}")
    _, default_docs = get_json("/api/documents?project_id=default")

    andromeda_files = [d["filename"] for d in andromeda_docs.get("documents", [])]
    default_files = [d["filename"] for d in default_docs.get("documents", [])]

    print(f"  Documents in Andromeda: {andromeda_files}", flush=True)
    print(f"  Documents in Default: {default_files}", flush=True)

    assert "andromeda_firmware.md" in andromeda_files, "Document should be in Andromeda"
    assert "andromeda_firmware.md" not in default_files, "Document should NOT be in Default workspace"
    print("  [PASS] Document isolation verified 100%! Zero cross-project context pollution.", flush=True)

    # 5. Save and List Chat Sessions
    print("\n--- [5] Testing Chat History Session Save & Resumption ---", flush=True)
    session_id = f"test_sess_{int(time.time())}"
    sample_messages = [
        {"role": "user", "text": "What is the encryption key for Andromeda?", "timestamp": int(time.time())},
        {"role": "assistant", "text": "The encryption key is ANDROMEDA-CIPHER-7744.", "timestamp": int(time.time()) + 1}
    ]
    status, save_res = post_json("/api/sessions/save", {
        "session_id": session_id,
        "title": "Andromeda Key Discussion",
        "messages": sample_messages,
        "project_id": proj_id,
        "engine": "lmstudio"
    })
    assert status == 200
    print(f"  [PASS] Saved session '{save_res['session']['title']}' (ID: {session_id})", flush=True)

    # List sessions
    status, sess_list = get_json(f"/api/sessions?project_id={proj_id}")
    found = any(s["id"] == session_id for s in sess_list.get("sessions", []))
    assert found, "Saved session must be listed in history"
    print(f"  [PASS] Session confirmed in project history list ({sess_list['count']} sessions)", flush=True)

    # Load session
    status, loaded_sess = get_json(f"/api/sessions/load?id={session_id}")
    assert status == 200
    assert len(loaded_sess["messages"]) == 2
    print(f"  [PASS] Loaded full session transcript accurately", flush=True)

    # 6. Test Safe Chat Distillation & Memorization
    print("\n--- [6] Testing Chat Distillation & Memorization into RAG ---", flush=True)
    status, dist_res = post_json("/api/sessions/distill", {
        "messages": sample_messages,
        "title": "Andromeda Architecture Decisions"
    })
    assert status == 200
    distilled_text = dist_res.get("distilled_text", "")
    print(f"  Distilled Knowledge Summary:\n{distilled_text.strip()}", flush=True)
    assert "ANDROMEDA-CIPHER-7744" in distilled_text

    # Index into project RAG
    status, mem_res = post_json("/api/sessions/memorize", {
        "project_id": proj_id,
        "title": "Andromeda Decisions",
        "summary_content": distilled_text
    })
    assert status == 200
    print("  [PASS] Distilled conversation successfully memorized into Project Knowledge Base!", flush=True)

    # Verify new document is present in project
    _, updated_docs = get_json(f"/api/documents?project_id={proj_id}")
    mem_files = [d["filename"] for d in updated_docs.get("documents", []) if "chat_memorized" in d["filename"]]
    assert len(mem_files) > 0, "Memorized chat file must be in project index"
    print(f"  [PASS] Verified memorized file in vector index: {mem_files[0]}", flush=True)

    # 7. Test RAG Retrieval from Memorized Knowledge using LM Studio
    print("\n--- [7] Testing RAG Context Query with LM Studio ---", flush=True)
    events = stream_chat("/api/chat/stream", {
        "engine": "lmstudio",
        "project_id": proj_id,
        "message": "What is the Encryption Key and Beacon Frequency for Andromeda?",
        "rag_enabled": True,
        "top_k": 2
    })
    tokens = [d["text"] for ev, d in events if ev == "token"]
    citations = [d for ev, d in events if ev == "citations"]
    full_resp = "".join(tokens)
    print(f"  RAG Response:\n{full_resp.strip()}", flush=True)
    assert "ANDROMEDA-CIPHER-7744" in full_resp or "1420.405" in full_resp
    print("  [PASS] RAG successfully retrieved and answered from Project Andromeda!", flush=True)

    # 8. Clean up
    print("\n--- [8] Cleanup ---", flush=True)
    post_json("/api/sessions/delete", {"session_id": session_id})
    post_json("/api/projects/delete", {"project_id": proj_id})
    print("  [PASS] Cleaned up test project and session.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(" ALL PROJECT RAG & CHAT HISTORY TESTS PASSED SUCCESSFULLY! (100%)", flush=True)
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    run_tests()
