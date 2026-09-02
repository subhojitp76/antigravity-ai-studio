"""
RAG Engine Orchestrator Module.
Coordinates document loading, chunking, vector indexing, context retrieval,
prompt augmentation, and citation metadata generation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rag.document_loader import DocumentLoader, Document
from rag.chunker import TextChunker, Chunk
from rag.vector_store import VectorStore


class RAGEngine:
    """End-to-end RAG orchestrator for OpenVINO LLM Studio."""

    def __init__(self, data_dir: str = None, documents_dir: str = None):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = Path(data_dir) if data_dir else self.base_dir / "data"
        self.documents_dir = Path(documents_dir) if documents_dir else self.base_dir / "documents"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.data_dir / "rag_index.json"
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=450, chunk_overlap=80)
        self.vector_store = VectorStore(persistence_path=str(self.index_path))

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingests a file from disk into the RAG vector index."""
        doc = self.loader.load_file(file_path)
        chunks = self.chunker.chunk_document(doc)
        self.vector_store.add_chunks(chunks)

        return {
            "success": True,
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "chunk_count": len(chunks),
            "char_count": len(doc.content)
        }

    def ingest_text_buffer(self, content: str = "", filename: str = "document.txt", text: str = None, source_type: str = "document") -> Dict[str, Any]:
        """Saves a text buffer to the documents directory and ingests it."""
        payload_text = text if text is not None else content
        safe_filename = Path(filename).name
        dest_path = self.documents_dir / safe_filename
        
        # Save file to documents directory
        with open(dest_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(payload_text)

        return self.ingest_file(str(dest_path))

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata and chunk statistics for all indexed documents."""
        return self.vector_store.get_document_stats()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the vector store and its source file if present."""
        # Find filename
        docs = self.vector_store.get_document_stats()
        target_doc = next((d for d in docs if d["doc_id"] == doc_id), None)
        
        if target_doc:
            filename = target_doc.get("filename")
            if filename:
                disk_file = self.documents_dir / filename
                if disk_file.exists():
                    try:
                        disk_file.unlink()
                    except Exception:
                        pass

        self.vector_store.remove_document(doc_id)
        return True

    def clear_all(self):
        """Clears all indexed documents and vector storage."""
        self.vector_store.clear()
        for f in self.documents_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict[str, Any]]:
        """Retrieves top-K relevant chunks with citation metadata."""
        matches = self.vector_store.search(query, top_k=top_k, min_score=min_score)
        
        citations = []
        for i, (chunk, score) in enumerate(matches, 1):
            citations.append({
                "citation_id": i,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "filename": chunk.filename,
                "score": score,
                "score_percent": int(score * 100),
                "snippet": chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
                "full_content": chunk.content
            })
        return citations

    def build_rag_prompt(self, user_query: str, citations: List[Dict[str, Any]]) -> str:
        """Constructs an augmented prompt embedding retrieved context."""
        if not citations:
            return user_query

        context_blocks = []
        for c in citations:
            block = f"[Source {c['citation_id']}: {c['filename']} (Relevance: {c['score_percent']}%) ]\n{c['full_content']}"
            context_blocks.append(block)

        context_text = "\n\n".join(context_blocks)

        prompt = (
            f"You are an expert AI assistant with access to the following verified reference documents:\n\n"
            f"==================== REFERENCE CONTEXT ====================\n"
            f"{context_text}\n"
            f"===========================================================\n\n"
            f"User Question: {user_query}\n\n"
            f"Instructions:\n"
            f"1. Answer the question thoroughly based on the reference context above.\n"
            f"2. Cite your sources using [Source 1], [Source 2], etc., when referencing specific information.\n"
            f"3. If the context does not contain the answer, explicitly state that the documents do not specify, and then provide your best knowledge."
        )
        return prompt
