"""
Project Manager for Multi-Workspace RAG and Chat History.
Handles isolated document repositories, vector indices, and persistent conversation sessions per project.
"""

import os
import json
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

from .rag_engine import RAGEngine

class ProjectManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.projects_dir = self.data_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of RAGEngine instances per project_id
        self._rag_engines: Dict[str, RAGEngine] = {}
        self.active_project_id = "default"
        
        # Ensure default project exists
        self._ensure_default_project()

    def _ensure_default_project(self):
        default_dir = self.projects_dir / "default"
        if not default_dir.exists():
            self.create_project(
                project_id="default",
                name="General Workspace",
                description="Default workspace for general chat and unassigned documents."
            )

    def list_projects(self) -> List[Dict[str, Any]]:
        """Lists all available projects with document and session counts."""
        projects = []
        for p_dir in self.projects_dir.iterdir():
            if p_dir.is_dir():
                meta_file = p_dir / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        meta = {"id": p_dir.name, "name": p_dir.name}
                else:
                    meta = {"id": p_dir.name, "name": p_dir.name}

                # Count documents and sessions
                docs_dir = p_dir / "documents"
                sessions_dir = p_dir / "sessions"
                doc_count = len(list(docs_dir.glob("*"))) if docs_dir.exists() else 0
                session_count = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0

                meta["doc_count"] = doc_count
                meta["session_count"] = session_count
                meta["is_active"] = (meta.get("id") == self.active_project_id)
                projects.append(meta)

        # Sort with default first, then newest
        projects.sort(key=lambda p: (0 if p["id"] == "default" else 1, -p.get("created_at", 0)))
        return projects

    def create_project(self, name: str, description: str = "", project_id: Optional[str] = None) -> Dict[str, Any]:
        """Creates a new isolated project workspace."""
        if not project_id:
            safe_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())[:24].strip('_')
            project_id = f"proj_{int(time.time())}_{safe_slug}" if safe_slug else f"proj_{int(time.time())}"

        proj_dir = self.projects_dir / project_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "documents").mkdir(exist_ok=True)
        (proj_dir / "sessions").mkdir(exist_ok=True)

        meta = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": int(time.time()),
            "updated_at": int(time.time())
        }

        with open(proj_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return meta

    def delete_project(self, project_id: str) -> bool:
        """Deletes a project workspace and its vector store (cannot delete default)."""
        if project_id == "default":
            return False

        proj_dir = self.projects_dir / project_id
        if proj_dir.exists():
            # Remove from memory cache
            if project_id in self._rag_engines:
                del self._rag_engines[project_id]
            shutil.rmtree(proj_dir, ignore_errors=True)
            if self.active_project_id == project_id:
                self.active_project_id = "default"
            return True
        return False

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Gets project metadata."""
        proj_dir = self.projects_dir / project_id
        meta_file = proj_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_rag_engine(self, project_id: Optional[str] = None) -> RAGEngine:
        """Returns or creates a cached RAGEngine for the given project."""
        p_id = project_id or self.active_project_id or "default"
        proj_dir = self.projects_dir / p_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        docs_dir = proj_dir / "documents"
        docs_dir.mkdir(exist_ok=True)

        if p_id not in self._rag_engines:
            self._rag_engines[p_id] = RAGEngine(
                data_dir=str(proj_dir),
                documents_dir=str(docs_dir)
            )

        return self._rag_engines[p_id]

    def set_active_project(self, project_id: str) -> bool:
        """Sets the active project workspace."""
        proj_dir = self.projects_dir / project_id
        if proj_dir.exists():
            self.active_project_id = project_id
            return True
        return False

    # --- SESSION MANAGEMENT ---

    def list_sessions(self, project_id: Optional[str] = None, query: str = "") -> List[Dict[str, Any]]:
        """Lists saved chat sessions, optionally filtered by project."""
        sessions = []
        target_projects = [self.projects_dir / project_id] if project_id else [p for p in self.projects_dir.iterdir() if p.is_dir()]

        for p_dir in target_projects:
            s_dir = p_dir / "sessions"
            if not s_dir.exists():
                continue
            
            proj_meta = self.get_project(p_dir.name) or {"name": p_dir.name}
            for s_file in s_dir.glob("*.json"):
                try:
                    with open(s_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        msg_count = len(data.get("messages", []))
                        last_msg = ""
                        for m in reversed(data.get("messages", [])):
                            if m.get("role") == "user":
                                last_msg = m.get("text", "")[:120]
                                break

                        # Search filter
                        title = data.get("title", "Untitled Session")
                        if query:
                            q_low = query.lower()
                            if q_low not in title.lower() and q_low not in last_msg.lower():
                                continue

                        sessions.append({
                            "id": data.get("id", s_file.stem),
                            "title": title,
                            "project_id": p_dir.name,
                            "project_name": proj_meta.get("name", p_dir.name),
                            "created_at": data.get("created_at", int(time.time())),
                            "updated_at": data.get("updated_at", int(time.time())),
                            "message_count": msg_count,
                            "last_snippet": last_msg,
                            "engine": data.get("engine", "openvino")
                        })
                except Exception:
                    continue

        sessions.sort(key=lambda s: -s.get("updated_at", 0))
        return sessions

    def save_session(self, session_id: str, title: str, messages: List[Dict[str, Any]], project_id: Optional[str] = None, engine: Optional[str] = None) -> Dict[str, Any]:
        """Saves or updates a conversation session transcript."""
        p_id = project_id or self.active_project_id or "default"
        proj_dir = self.projects_dir / p_id
        sessions_dir = proj_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Auto-generate title from first user message if default
        if (not title or title == "New Chat") and messages:
            for m in messages:
                if m.get("role") == "user" and m.get("text"):
                    clean = m["text"].strip().replace("\n", " ")
                    title = clean[:38] + ("..." if len(clean) > 38 else "")
                    break
        if not title:
            title = "Chat Session"

        s_file = sessions_dir / f"{session_id}.json"
        existing_created = int(time.time())
        if s_file.exists():
            try:
                with open(s_file, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    existing_created = old_data.get("created_at", existing_created)
            except Exception:
                pass

        session_data = {
            "id": session_id,
            "title": title,
            "project_id": p_id,
            "engine": engine or "openvino",
            "created_at": existing_created,
            "updated_at": int(time.time()),
            "messages": messages
        }

        with open(s_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2)

        return session_data

    def get_session(self, session_id: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves a specific chat session."""
        target_projects = [self.projects_dir / project_id] if project_id else [p for p in self.projects_dir.iterdir() if p.is_dir()]
        for p_dir in target_projects:
            s_file = p_dir / "sessions" / f"{session_id}.json"
            if s_file.exists():
                try:
                    with open(s_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
        return None

    def delete_session(self, session_id: str, project_id: Optional[str] = None) -> bool:
        """Deletes a chat session."""
        target_projects = [self.projects_dir / project_id] if project_id else [p for p in self.projects_dir.iterdir() if p.is_dir()]
        for p_dir in target_projects:
            s_file = p_dir / "sessions" / f"{session_id}.json"
            if s_file.exists():
                try:
                    os.remove(s_file)
                    return True
                except Exception:
                    return False
        return False

    # --- CHAT MEMORIZATION & DISTILLATION ---

    def distill_chat_summary(self, messages: List[Dict[str, Any]], session_title: str = "") -> str:
        """
        Converts multi-turn chat dialogues into a clean, structured reference entry
        free of conversational noise (greetings, formatting chatter) for safe RAG ingestion.
        """
        if not messages:
            return ""

        title = session_title or "Conversation Notes"
        lines = [f"# Distilled Knowledge: {title}", ""]
        lines.append(f"*Recorded on:* {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("## Key Insights & Q&A Summary:")
        lines.append("")

        qa_count = 0
        current_q = None
        for m in messages:
            role = m.get("role")
            text = m.get("text", "").strip()
            if not text:
                continue

            if role == "user":
                current_q = text
            elif role == "assistant" and current_q:
                qa_count += 1
                lines.append(f"### Q{qa_count}: {current_q}")
                # Filter out raw pleasantries or code wrap noise
                cleaned_ans = text.strip()
                lines.append(f"**Answer / Synthesis:**\n{cleaned_ans}")
                lines.append("")
                current_q = None

        return "\n".join(lines)

    def memorize_conversation(self, project_id: str, title: str, summary_content: str) -> Dict[str, Any]:
        """
        Indexes a distilled conversation summary into the project's RAG knowledge base.
        """
        rag = self.get_rag_engine(project_id)
        filename = f"chat_memorized_{int(time.time())}.md"
        res = rag.ingest_text_buffer(
            text=summary_content,
            filename=filename,
            source_type="chat_memorization"
        )
        return res
