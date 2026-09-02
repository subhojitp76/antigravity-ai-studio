"""
Vector Store Module for RAG System.
High-speed in-memory vector index with TF-IDF/Cosine similarity and JSON disk persistence.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunker import Chunk


class VectorStore:
    """In-memory vector store with fast similarity search and persistence."""

    def __init__(self, persistence_path: str = None):
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.chunks: List[Chunk] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            max_features=50000
        )
        self.tfidf_matrix = None
        self._is_fitted = False

        if self.persistence_path and self.persistence_path.exists():
            self.load()

    def add_chunks(self, new_chunks: List[Chunk]):
        """Adds new chunks to the store and rebuilds the vector matrix."""
        if not new_chunks:
            return

        # Deduplicate by chunk_id
        existing_ids = {c.chunk_id for c in self.chunks}
        for chunk in new_chunks:
            if chunk.chunk_id not in existing_ids:
                self.chunks.append(chunk)
                existing_ids.add(chunk.chunk_id)

        self._rebuild_index()
        self.save()

    def remove_document(self, doc_id: str):
        """Removes all chunks for a document ID and rebuilds index."""
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        self._rebuild_index()
        self.save()

    def clear(self):
        """Clears all chunks and resets the index."""
        self.chunks = []
        self.tfidf_matrix = None
        self._is_fitted = False
        if self.persistence_path and self.persistence_path.exists():
            try:
                self.persistence_path.unlink()
            except Exception:
                pass

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Tuple[Chunk, float]]:
        """Searches for the most relevant chunks matching the query string."""
        if not self._is_fitted or not self.chunks or not query.strip():
            return []

        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            # Rank by score descending
            ranked_indices = np.argsort(scores)[::-1]

            results = []
            for idx in ranked_indices[:top_k]:
                score = float(scores[idx])
                if score >= min_score:
                    results.append((self.chunks[idx], round(score, 4)))

            return results
        except Exception as e:
            print(f"[VectorStore Error] Search failed: {e}")
            return []

    def get_document_stats(self) -> List[Dict[str, Any]]:
        """Returns aggregated stats for all indexed documents."""
        docs = {}
        for c in self.chunks:
            if c.doc_id not in docs:
                docs[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "file_type": c.metadata.get("file_type", "unknown"),
                    "chunk_count": 0,
                    "total_chars": 0
                }
            docs[c.doc_id]["chunk_count"] += 1
            docs[c.doc_id]["total_chars"] += len(c.content)

        return list(docs.values())

    def _rebuild_index(self):
        if not self.chunks:
            self.tfidf_matrix = None
            self._is_fitted = False
            return

        corpus = [c.content for c in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._is_fitted = True

    def save(self):
        if not self.persistence_path:
            return
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "chunks": [c.to_dict() for c in self.chunks]
            }
            with open(self.persistence_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VectorStore] Failed to save index to disk: {e}")

    def load(self):
        if not self.persistence_path or not self.persistence_path.exists():
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks_data = data.get("chunks", [])
            self.chunks = [
                Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    filename=item["filename"],
                    content=item["content"],
                    index=item.get("index", 0),
                    metadata=item.get("metadata", {})
                )
                for item in chunks_data
            ]
            self._rebuild_index()
            print(f"[VectorStore] Loaded {len(self.chunks)} chunk(s) from {self.persistence_path.name}")
        except Exception as e:
            print(f"[VectorStore] Failed to load index from disk: {e}")
