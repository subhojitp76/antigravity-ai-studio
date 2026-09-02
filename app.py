"""
Model Manager & Chat Application for OpenVINO (NPU / CPU / GPU) & LM Studio with Multi-Workspace Project RAG.
High-performance backend server providing REST API, SSE streaming, Document Knowledge Bases,
Multi-Session Chat History, and Safe Chat Memorization.
"""

import os
import sys
import time
import json
import base64
import socket
import gc
import psutil
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# OpenVINO imports
try:
    import openvino as ov
    import openvino_genai as ov_genai
except ImportError as e:
    print(f"[ERROR] Required OpenVINO packages not found: {e}")
    print("Please ensure openvino and openvino-genai are installed.")
    sys.exit(1)

# RAG & Project imports
from rag import RAGEngine, ProjectManager

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR_DEFAULT = BASE_DIR / "llama-3.2-3b-ov"

# Initialize Project Manager (which handles project-scoped RAG engines and persistent sessions)
project_mgr = ProjectManager(data_dir=str(DATA_DIR))


# Global Model State Machine
class ModelState:
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    ERROR = "error"


class EngineType:
    OPENVINO = "openvino"
    LMSTUDIO = "lmstudio"


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = ModelState.UNLOADED
        self.pipe = None
        self.device = "NPU"
        self.model_path = str(MODEL_DIR_DEFAULT)
        self.compile_start_time = None
        self.compile_duration = None
        self.last_error = None
        self.cancel_requested = False
        self.chat_history = None
        self.system_prompt = ""
        self.config = {
            "MAX_PROMPT_LEN": 1024,
            "MIN_RESPONSE_LEN": 512,
            "GENERATE_HINT": "BEST_PERF"
        }
        self.metrics_history = []
        self.total_tokens_generated = 0

    def get_available_devices(self):
        core = ov.Core()
        devices = []
        available = core.available_devices

        if "NPU" in available:
            devices.append({"id": "NPU", "name": "Intel NPU (AI Boost)", "is_recommended": True})
        if "CPU" in available:
            devices.append({"id": "CPU", "name": "Intel Core Ultra CPU", "is_recommended": False})
        if "GPU.0" in available:
            devices.append({"id": "GPU.0", "name": "Intel Graphics (iGPU)", "is_recommended": False})
        if "GPU.1" in available:
            devices.append({"id": "GPU.1", "name": "NVIDIA RTX 5070 Ti (dGPU)", "is_recommended": False})
        elif "GPU" in available:
            devices.append({"id": "GPU", "name": "GPU Device", "is_recommended": False})

        for d in available:
            if not any(dev["id"] == d for dev in devices):
                devices.append({"id": d, "name": f"Device {d}", "is_recommended": False})

        return devices

    def start_model(self, device="NPU", model_path=None, config=None):
        with self.lock:
            if self.state in [ModelState.LOADING, ModelState.READY, ModelState.GENERATING]:
                return {"success": False, "message": f"Model is currently in '{self.state}' state."}

            self.state = ModelState.LOADING
            self.device = device
            if model_path:
                self.model_path = model_path
            if config:
                self.config.update(config)

            self.compile_start_time = time.time()
            self.compile_duration = None
            self.last_error = None

        thread = threading.Thread(target=self._compile_and_load_worker, daemon=True)
        thread.start()
        return {"success": True, "message": f"Compilation started on {device}."}

    def _compile_and_load_worker(self):
        try:
            print(f"[ModelManager] Compiling LLMPipeline on {self.device}...")
            print(f"[ModelManager] Model Path: {self.model_path}")
            
            npu_config = {}
            if self.device.startswith("NPU"):
                npu_config = {
                    "MAX_PROMPT_LEN": int(self.config.get("MAX_PROMPT_LEN", 1024)),
                    "MIN_RESPONSE_LEN": int(self.config.get("MIN_RESPONSE_LEN", 512)),
                    "GENERATE_HINT": str(self.config.get("GENERATE_HINT", "BEST_PERF"))
                }
                pipe = ov_genai.LLMPipeline(self.model_path, self.device, **npu_config)
            else:
                pipe = ov_genai.LLMPipeline(self.model_path, self.device)

            duration = round(time.time() - self.compile_start_time, 2)
            with self.lock:
                self.pipe = pipe
                self.state = ModelState.READY
                self.compile_duration = duration
                self.chat_history = ov_genai.ChatHistory()
                if self.system_prompt:
                    self.chat_history.append({"role": "system", "content": self.system_prompt})

            print(f"[ModelManager] Pipeline READY on {self.device} in {duration}s!")
        except Exception as e:
            err_msg = str(e)
            print(f"[ModelManager] Compilation ERROR on {self.device}: {err_msg}")
            with self.lock:
                self.state = ModelState.ERROR
                self.last_error = err_msg
                self.pipe = None

    def stop_model(self):
        with self.lock:
            if self.pipe is not None:
                del self.pipe
                self.pipe = None
            if self.chat_history is not None:
                del self.chat_history
                self.chat_history = None
            gc.collect()
            self.state = ModelState.UNLOADED
            self.compile_duration = None
            self.cancel_requested = False
            print("[ModelManager] Model pipeline stopped and unloaded.")
            return {"success": True, "message": "Model unloaded successfully."}

    def reset_chat(self, system_prompt=None):
        with self.lock:
            if system_prompt is not None:
                self.system_prompt = system_prompt
            if self.pipe is not None:
                self.chat_history = ov_genai.ChatHistory()
                if self.system_prompt:
                    self.chat_history.append({"role": "system", "content": self.system_prompt})
            return {"success": True, "message": "Chat context reset."}

    def stop_generation(self):
        self.cancel_requested = True
        return {"success": True, "message": "Stop requested."}

    def get_status(self):
        with self.lock:
            elapsed = None
            if self.state == ModelState.LOADING and self.compile_start_time:
                elapsed = round(time.time() - self.compile_start_time, 1)

            mem = psutil.virtual_memory()
            system_info = {
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024**3), 2),
                "ram_total_gb": round(mem.total / (1024**3), 2),
                "cpu_percent": psutil.cpu_percent(interval=None)
            }

            return {
                "state": self.state,
                "device": self.device,
                "model_name": Path(self.model_path).name,
                "compile_duration": self.compile_duration,
                "elapsed_loading": elapsed,
                "last_error": self.last_error,
                "config": self.config,
                "total_tokens_generated": self.total_tokens_generated,
                "system": system_info
            }


class LMStudioClient:
    """Client for communicating with LM Studio's local OpenAI-compatible REST server."""
    def __init__(self, base_url="http://127.0.0.1:1234"):
        self.base_url = base_url.rstrip("/")
        self.active_model = None
        self.cancel_requested = False
        self.chat_history = []
        self.system_prompt = ""
        self.last_error = None
        self.lock = threading.Lock()

    def probe(self):
        """Checks if LM Studio is reachable and returns list of models."""
        try:
            url = f"{self.base_url}/v1/models"
            req = urllib.request.Request(url, headers={"User-Agent": "OpenVINO-Studio"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = []
                for item in data.get("data", []):
                    model_id = item.get("id")
                    if model_id and not model_id.startswith("text-embedding"):
                        models.append({
                            "id": model_id,
                            "name": model_id,
                            "owned_by": item.get("owned_by", "")
                        })
                
                with self.lock:
                    if models and not self.active_model:
                        self.active_model = models[0]["id"]
                    elif models and self.active_model not in [m["id"] for m in models]:
                        self.active_model = models[0]["id"]
                    self.last_error = None

                return True, models, None
        except Exception as e:
            with self.lock:
                self.last_error = str(e)
            return False, [], str(e)

    def set_model(self, model_id):
        with self.lock:
            self.active_model = model_id

    def reset_chat(self, system_prompt=None):
        with self.lock:
            if system_prompt is not None:
                self.system_prompt = system_prompt
            self.chat_history = []
        return {"success": True, "message": "LM Studio chat context reset."}

    def stop_generation(self):
        self.cancel_requested = True
        return {"success": True, "message": "LM Studio stop requested."}

    def get_status(self):
        is_online, models, err = self.probe()
        return {
            "connected": is_online,
            "base_url": self.base_url,
            "active_model": self.active_model,
            "available_models": models,
            "error": err
        }


# Global Manager Instances
model_mgr = ModelManager()
lm_studio = LMStudioClient()
active_engine_mode = EngineType.OPENVINO  # Default engine mode


class StudioHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence static polling log noise safely
        try:
            msg = format % args
            if "GET /api/status" in msg or "GET /api/engine/status" in msg:
                return
        except Exception:
            pass
        super().log_message(format, *args)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = (STATIC_DIR / rel_path).resolve()
            if STATIC_DIR in file_path.parents or file_path == STATIC_DIR:
                mime_type = self._get_mime_type(file_path)
                self._serve_file(file_path, mime_type)
            else:
                self.send_error(403, "Forbidden")
        elif path == "/api/devices":
            self._send_json({"devices": model_mgr.get_available_devices()})
        elif path == "/api/status":
            self._send_json(model_mgr.get_status())
        elif path == "/api/engine/status":
            global active_engine_mode
            lm_status = lm_studio.get_status()
            ov_status = model_mgr.get_status()
            self._send_json({
                "active_engine": active_engine_mode,
                "active_project_id": project_mgr.active_project_id,
                "openvino": ov_status,
                "lmstudio": lm_status,
                "system": ov_status.get("system", {})
            })
        elif path == "/api/lmstudio/models":
            connected, models, err = lm_studio.probe()
            self._send_json({
                "connected": connected,
                "models": models,
                "active_model": lm_studio.active_model,
                "error": err
            })
        elif path == "/api/system_info":
            mem = psutil.virtual_memory()
            self._send_json({
                "ram_percent": mem.percent,
                "ram_used_gb": round(mem.used / (1024**3), 2),
                "ram_total_gb": round(mem.total / (1024**3), 2),
                "cpu_percent": psutil.cpu_percent(interval=None)
            })

        # --- PROJECTS API ---
        elif path == "/api/projects":
            projects = project_mgr.list_projects()
            self._send_json({"projects": projects, "active_project_id": project_mgr.active_project_id})

        elif path == "/api/projects/active":
            proj = project_mgr.get_project(project_mgr.active_project_id)
            self._send_json(proj or {"id": "default", "name": "General Workspace"})

        # --- SESSIONS API ---
        elif path == "/api/sessions":
            p_id = query_params.get("project_id", [None])[0]
            q = query_params.get("q", [""])[0]
            sessions = project_mgr.list_sessions(project_id=p_id, query=q)
            self._send_json({"sessions": sessions, "count": len(sessions)})

        elif path == "/api/sessions/load":
            sess_id = query_params.get("id", [None])[0]
            if not sess_id:
                self._send_json({"error": "Missing session id"}, status=400)
                return
            session_data = project_mgr.get_session(sess_id)
            if session_data:
                self._send_json(session_data)
            else:
                self._send_json({"error": "Session not found"}, status=404)

        # --- DOCUMENTS API (Project Scoped) ---
        elif path == "/api/documents":
            p_id = query_params.get("project_id", [project_mgr.active_project_id])[0]
            rag = project_mgr.get_rag_engine(p_id)
            docs = rag.list_documents()
            self._send_json({"documents": docs, "total": len(docs), "project_id": p_id})

        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        global active_engine_mode
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            req_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            req_data = {}

        if path == "/api/engine/select":
            engine = req_data.get("engine", EngineType.OPENVINO)
            if engine in [EngineType.OPENVINO, EngineType.LMSTUDIO]:
                active_engine_mode = engine
                if engine == EngineType.LMSTUDIO and "model" in req_data:
                    lm_studio.set_model(req_data["model"])
                elif engine == EngineType.OPENVINO and "device" in req_data:
                    model_mgr.device = req_data["device"]
                self._send_json({"success": True, "active_engine": active_engine_mode})
            else:
                self._send_json({"error": "Invalid engine specified."}, status=400)

        elif path == "/api/model/start":
            device = req_data.get("device", "NPU")
            config = req_data.get("config", {
                "MAX_PROMPT_LEN": 1024,
                "MIN_RESPONSE_LEN": 512,
                "GENERATE_HINT": "BEST_PERF"
            })
            model_path = req_data.get("model_path", None)
            res = model_mgr.start_model(device=device, model_path=model_path, config=config)
            self._send_json(res)

        elif path == "/api/model/stop":
            res = model_mgr.stop_model()
            self._send_json(res)

        elif path == "/api/lmstudio/select_model":
            model_id = req_data.get("model")
            if model_id:
                lm_studio.set_model(model_id)
                self._send_json({"success": True, "active_model": model_id})
            else:
                self._send_json({"error": "Missing model parameter."}, status=400)

        # --- PROJECTS API ---
        elif path == "/api/projects/create":
            name = req_data.get("name", "New Project").strip()
            desc = req_data.get("description", "").strip()
            if not name:
                self._send_json({"error": "Project name cannot be empty."}, status=400)
                return
            new_proj = project_mgr.create_project(name=name, description=desc)
            project_mgr.set_active_project(new_proj["id"])
            self._send_json({"success": True, "project": new_proj})

        elif path == "/api/projects/select":
            proj_id = req_data.get("project_id")
            if proj_id and project_mgr.set_active_project(proj_id):
                self._send_json({"success": True, "active_project_id": proj_id})
            else:
                self._send_json({"error": "Invalid project id"}, status=400)

        elif path == "/api/projects/delete":
            proj_id = req_data.get("project_id")
            if not proj_id or proj_id == "default":
                self._send_json({"error": "Cannot delete default project"}, status=400)
                return
            success = project_mgr.delete_project(proj_id)
            self._send_json({"success": success, "active_project_id": project_mgr.active_project_id})

        # --- SESSIONS API ---
        elif path == "/api/sessions/save":
            sess_id = req_data.get("session_id") or f"sess_{int(time.time())}"
            title = req_data.get("title", "")
            messages = req_data.get("messages", [])
            p_id = req_data.get("project_id") or project_mgr.active_project_id
            engine = req_data.get("engine", active_engine_mode)
            saved = project_mgr.save_session(
                session_id=sess_id,
                title=title,
                messages=messages,
                project_id=p_id,
                engine=engine
            )
            self._send_json({"success": True, "session": saved})

        elif path == "/api/sessions/delete":
            sess_id = req_data.get("session_id")
            if sess_id:
                success = project_mgr.delete_session(sess_id)
                self._send_json({"success": success})
            else:
                self._send_json({"error": "Missing session_id"}, status=400)

        elif path == "/api/sessions/distill":
            messages = req_data.get("messages", [])
            title = req_data.get("title", "Conversation Summary")
            distilled_text = project_mgr.distill_chat_summary(messages, session_title=title)
            self._send_json({"success": True, "distilled_text": distilled_text})

        elif path == "/api/sessions/memorize":
            p_id = req_data.get("project_id") or project_mgr.active_project_id
            title = req_data.get("title", "Conversation")
            summary_content = req_data.get("summary_content", "").strip()
            if not summary_content:
                self._send_json({"error": "Summary content cannot be empty"}, status=400)
                return
            res = project_mgr.memorize_conversation(project_id=p_id, title=title, summary_content=summary_content)
            self._send_json({"success": True, "result": res})

        elif path == "/api/chat/reset":
            sys_prompt = req_data.get("system_prompt", None)
            res_ov = model_mgr.reset_chat(system_prompt=sys_prompt)
            res_lm = lm_studio.reset_chat(system_prompt=sys_prompt)
            self._send_json({"success": True, "message": "Chat context reset for all engines."})

        elif path == "/api/chat/stop_generation":
            model_mgr.stop_generation()
            lm_studio.stop_generation()
            self._send_json({"success": True, "message": "Stop requested."})

        elif path == "/api/chat/stream":
            self._handle_chat_stream(req_data)

        # --- DOCUMENTS API (Project Scoped) ---
        elif path == "/api/documents/upload":
            self._handle_document_upload(req_data)

        elif path == "/api/documents/delete":
            doc_id = req_data.get("doc_id")
            p_id = req_data.get("project_id") or project_mgr.active_project_id
            if not doc_id:
                self._send_json({"error": "Missing doc_id"}, status=400)
                return
            rag = project_mgr.get_rag_engine(p_id)
            success = rag.delete_document(doc_id)
            self._send_json({"success": success, "message": "Document removed from project knowledge base."})

        elif path == "/api/documents/clear":
            p_id = req_data.get("project_id") or project_mgr.active_project_id
            rag = project_mgr.get_rag_engine(p_id)
            rag.clear_all()
            self._send_json({"success": True, "message": "Project knowledge base cleared."})

        else:
            self.send_error(404, "Not Found")

    def _handle_document_upload(self, req_data):
        filename = req_data.get("filename", "document.txt")
        content = req_data.get("content", "")
        base64_content = req_data.get("base64", None)
        p_id = req_data.get("project_id") or project_mgr.active_project_id

        rag = project_mgr.get_rag_engine(p_id)

        if base64_content:
            try:
                raw_bytes = base64.b64decode(base64_content)
                dest_path = rag.documents_dir / Path(filename).name
                with open(dest_path, "wb") as f:
                    f.write(raw_bytes)
                res = rag.ingest_file(str(dest_path))
                self._send_json(res)
                return
            except Exception as e:
                self._send_json({"error": f"Failed to process binary upload: {e}"}, status=400)
                return

        if not content.strip():
            self._send_json({"error": "Empty file content"}, status=400)
            return

        try:
            res = rag.ingest_text_buffer(content=content, filename=filename)
            self._send_json(res)
        except Exception as e:
            self._send_json({"error": f"Document ingestion failed: {e}"}, status=500)

    def _handle_chat_stream(self, req_data):
        global active_engine_mode
        message = req_data.get("message", "").strip()
        rag_enabled = req_data.get("rag_enabled", False)
        top_k = int(req_data.get("top_k", 3))
        p_id = req_data.get("project_id") or project_mgr.active_project_id
        engine_override = req_data.get("engine", active_engine_mode)

        if not message:
            self._send_json({"error": "Empty message prompt."}, status=400)
            return

        # Prepare SSE response headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def send_sse_event(event_name, data_obj):
            try:
                payload = f"event: {event_name}\ndata: {json.dumps(data_obj)}\n\n".encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()
            except Exception:
                pass

        # Perform RAG retrieval from active project if enabled
        citations = []
        effective_prompt = message
        if rag_enabled:
            rag = project_mgr.get_rag_engine(p_id)
            citations = rag.retrieve(message, top_k=top_k)
            effective_prompt = rag.build_rag_prompt(message, citations)
            send_sse_event("citations", {"citations": citations, "count": len(citations), "project_id": p_id})

        if engine_override == EngineType.LMSTUDIO:
            self._stream_lm_studio(effective_prompt, citations, req_data, send_sse_event)
        else:
            self._stream_openvino(effective_prompt, citations, req_data, send_sse_event)

    def _stream_openvino(self, effective_prompt, citations, req_data, send_sse_event):
        if model_mgr.state != ModelState.READY or model_mgr.pipe is None:
            send_sse_event("error", {"message": "OpenVINO Model is not ready. Please click 'Start Model' first."})
            self.close_connection = True
            return

        model_mgr.cancel_requested = False
        with model_mgr.lock:
            model_mgr.state = ModelState.GENERATING

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        collected_text = []

        def streamer_callback(subword):
            nonlocal first_token_time, token_count
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            token_count += 1
            collected_text.append(subword)

            elapsed_gen = now - first_token_time if first_token_time else 0.001
            live_tps = round((token_count - 1) / max(elapsed_gen, 0.001), 1) if token_count > 1 else 0.0

            send_sse_event("token", {
                "text": subword,
                "token_index": token_count,
                "live_tps": live_tps
            })

            if model_mgr.cancel_requested:
                return ov_genai.StreamingStatus.STOP

            return ov_genai.StreamingStatus.RUNNING

        gen_kwargs = {}
        if "max_new_tokens" in req_data:
            gen_kwargs["max_new_tokens"] = int(req_data["max_new_tokens"])
        if "temperature" in req_data and float(req_data["temperature"]) > 0:
            gen_kwargs["temperature"] = float(req_data["temperature"])
            gen_kwargs["do_sample"] = True
        if "top_p" in req_data and float(req_data["top_p"]) < 1.0:
            gen_kwargs["top_p"] = float(req_data["top_p"])
            gen_kwargs["do_sample"] = True
        if "repetition_penalty" in req_data:
            gen_kwargs["repetition_penalty"] = float(req_data["repetition_penalty"])

        try:
            send_sse_event("start", {"status": "generating", "engine": "openvino", "device": model_mgr.device})
            
            # Sliding Window Context Compaction for NPU (Keep last 6 turns to avoid buffer overflow)
            history_messages = req_data.get("history", [])
            with model_mgr.lock:
                model_mgr.chat_history = ov_genai.ChatHistory()
                if model_mgr.system_prompt:
                    model_mgr.chat_history.append({"role": "system", "content": model_mgr.system_prompt})

                # Append sliding window of previous turns
                window = history_messages[-6:] if len(history_messages) > 6 else history_messages
                for m in window:
                    role = m.get("role")
                    content = m.get("text", "")
                    if role in ["user", "assistant"] and content:
                        model_mgr.chat_history.append({"role": role, "content": content})

                # Append current user prompt
                model_mgr.chat_history.append({"role": "user", "content": effective_prompt})
                chat_hist = model_mgr.chat_history

            # Inference
            model_mgr.pipe.generate(chat_hist, streamer=streamer_callback, **gen_kwargs)
            end_time = time.perf_counter()

            full_response_str = "".join(collected_text)
            with model_mgr.lock:
                if model_mgr.chat_history is not None:
                    model_mgr.chat_history.append({"role": "assistant", "content": full_response_str})

            total_duration = end_time - start_time
            ttft_ms = round((first_token_time - start_time) * 1000, 1) if first_token_time else 0.0
            gen_duration = (end_time - first_token_time) if first_token_time else total_duration
            tps = round((token_count - 1) / max(gen_duration, 0.001), 2) if token_count > 1 else 0.0

            model_mgr.total_tokens_generated += token_count
            metrics = {
                "engine": "OpenVINO (NPU)",
                "ttft_ms": ttft_ms,
                "tps": tps,
                "token_count": token_count,
                "total_duration_s": round(total_duration, 2),
                "interrupted": model_mgr.cancel_requested,
                "citations_count": len(citations)
            }
            send_sse_event("metrics", metrics)
            send_sse_event("done", {
                "full_text": full_response_str,
                "metrics": metrics,
                "citations": citations
            })
        except Exception as e:
            send_sse_event("error", {"message": str(e)})
        finally:
            with model_mgr.lock:
                if model_mgr.state == ModelState.GENERATING:
                    model_mgr.state = ModelState.READY
            self.close_connection = True

    def _stream_lm_studio(self, effective_prompt, citations, req_data, send_sse_event):
        lm_studio.cancel_requested = False
        
        # Verify active model
        if not lm_studio.active_model:
            ok, models, err = lm_studio.probe()
            if not ok or not models:
                send_sse_event("error", {"message": f"LM Studio is offline or has no models loaded: {err or 'Check port 1234'}"})
                self.close_connection = True
                return

        # Build messages payload with sliding window
        history_messages = req_data.get("history", [])
        messages = []
        if lm_studio.system_prompt:
            messages.append({"role": "system", "content": lm_studio.system_prompt})
        
        # Append sliding window of previous turns
        window = history_messages[-10:] if len(history_messages) > 10 else history_messages
        for m in window:
            role = m.get("role")
            content = m.get("text", "")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": effective_prompt})

        max_tokens = int(req_data.get("max_new_tokens", 512))
        payload = {
            "model": lm_studio.active_model,
            "messages": messages,
            "stream": True,
            "max_tokens": max(max_tokens, 256)
        }
        if "temperature" in req_data:
            payload["temperature"] = float(req_data["temperature"])
        if "top_p" in req_data:
            payload["top_p"] = float(req_data["top_p"])

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        collected_text = []
        collected_reasoning = []

        try:
            send_sse_event("start", {"status": "generating", "engine": "lmstudio", "model": lm_studio.active_model})
            
            req = urllib.request.Request(
                f"{lm_studio.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Antigravity-Studio"}
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                for line_b in resp:
                    if lm_studio.cancel_requested:
                        break
                    line_str = line_b.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data: "):
                        continue
                    data_chunk = line_str[6:].strip()
                    if data_chunk == "[DONE]":
                        break
                    
                    try:
                        parsed = json.loads(data_chunk)
                        choices = parsed.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        
                        # Handle reasoning tokens if model is DeepSeek-R1, Gemma-4, Qwen, etc.
                        reasoning_piece = delta.get("reasoning_content")
                        if reasoning_piece:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            token_count += 1
                            collected_reasoning.append(reasoning_piece)
                            send_sse_event("reasoning", {"text": reasoning_piece})

                        content_piece = delta.get("content")
                        if content_piece:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            token_count += 1
                            collected_text.append(content_piece)

                            elapsed_gen = now - first_token_time if first_token_time else 0.001
                            live_tps = round((token_count - 1) / max(elapsed_gen, 0.001), 1) if token_count > 1 else 0.0

                            send_sse_event("token", {
                                "text": content_piece,
                                "token_index": token_count,
                                "live_tps": live_tps
                            })
                    except Exception:
                        pass

            end_time = time.perf_counter()
            full_response_str = "".join(collected_text)
            full_reasoning_str = "".join(collected_reasoning)
            
            # Fallback if model only output reasoning tokens
            display_response = full_response_str if full_response_str else full_reasoning_str

            total_duration = end_time - start_time
            ttft_ms = round((first_token_time - start_time) * 1000, 1) if first_token_time else 0.0
            gen_duration = (end_time - first_token_time) if first_token_time else total_duration
            tps = round((token_count - 1) / max(gen_duration, 0.001), 2) if token_count > 1 else 0.0

            metrics = {
                "engine": f"LM Studio ({lm_studio.active_model})",
                "ttft_ms": ttft_ms,
                "tps": tps,
                "token_count": token_count,
                "total_duration_s": round(total_duration, 2),
                "interrupted": lm_studio.cancel_requested,
                "citations_count": len(citations)
            }
            send_sse_event("metrics", metrics)
            send_sse_event("done", {
                "full_text": display_response,
                "reasoning": full_reasoning_str,
                "metrics": metrics,
                "citations": citations
            })
        except Exception as e:
            send_sse_event("error", {"message": f"LM Studio communication error: {e}"})
        finally:
            self.close_connection = True

    def _serve_file(self, file_path, mime_type):
        if not file_path.is_file():
            self.send_error(404, f"File not found: {file_path.name}")
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            if "text/html" not in mime_type:
                self.send_header("Cache-Control", "public, max-age=3600")
            else:
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def _get_mime_type(self, file_path):
        suffix = file_path.suffix.lower()
        mimes = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon"
        }
        return mimes.get(suffix, "application/octet-stream")

    def _send_json(self, data, status=200):
        try:
            payload = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            print(f"[ERROR] Failed to send JSON response: {e}")


def run_server(port=7860, open_browser=True):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    server_address = ("127.0.0.1", port)
    
    try:
        httpd = ThreadingHTTPServer(server_address, StudioHTTPHandler)
    except OSError as e:
        print(f"[ERROR] Could not start server on port {port}: {e}")
        # Try finding open port
        for fallback_port in range(7861, 7870):
            try:
                server_address = ("127.0.0.1", fallback_port)
                httpd = ThreadingHTTPServer(server_address, StudioHTTPHandler)
                port = fallback_port
                break
            except OSError:
                continue

    url = f"http://localhost:{port}"
    print("=" * 60)
    print(f"  OPENVINO & LM STUDIO MULTI-WORKSPACE RAG STUDIO")
    print(f"  Running at: {url}")
    print(f"  OpenVINO IR Path: {MODEL_DIR_DEFAULT}")
    print(f"  LM Studio Endpoint: {lm_studio.base_url}")
    print(f"  Projects Data Path: {DATA_DIR / 'projects'}")
    print("=" * 60)

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        model_mgr.stop_model()
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenVINO & LM Studio Multi-Workspace Studio")
    parser.add_argument("--port", type=int, default=7860, help="Port to run web server on")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open web browser")
    args = parser.parse_args()

    run_server(port=args.port, open_browser=not args.no_browser)
