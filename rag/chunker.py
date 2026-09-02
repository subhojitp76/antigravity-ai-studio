"""
Text Chunker Module for RAG System.
Implements recursive semantic character and paragraph chunking with overlap.
"""

from typing import List
from rag.document_loader import Document


class Chunk:
    """Represents a discrete indexed chunk of text with source tracking."""
    def __init__(self, chunk_id: str, doc_id: str, filename: str, content: str, index: int, metadata: dict = None):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.filename = filename
        self.content = content.strip()
        self.index = index
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "content": self.content,
            "index": self.index,
            "char_count": len(self.content),
            "metadata": self.metadata
        }


class TextChunker:
    """Splits text documents into overlapping semantic chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

    def chunk_document(self, doc: Document) -> List[Chunk]:
        raw_text = doc.content
        if not raw_text or not raw_text.strip():
            return []

        chunks_text = self._split_text(raw_text, self.chunk_size, self.chunk_overlap)
        result = []
        for i, text in enumerate(chunks_text):
            if not text.strip():
                continue
            chunk_id = f"{doc.doc_id}_chunk_{i}"
            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                filename=doc.filename,
                content=text,
                index=i,
                metadata={
                    "file_type": doc.file_type,
                    "total_doc_chunks": len(chunks_text)
                }
            )
            result.append(chunk)
        return result

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        # Use recursive splitting by separators
        return self._recursive_split(text, self.separators, chunk_size, chunk_overlap)

    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, s in enumerate(separators):
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []

        for s in splits:
            if not s:
                continue
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    other_splits = self._recursive_split(s, new_separators, chunk_size, chunk_overlap)
                    good_splits.extend(other_splits)
                else:
                    # Hard split
                    for j in range(0, len(s), chunk_size - chunk_overlap):
                        good_splits.append(s[j:j + chunk_size])

        # Merge splits with overlap
        current_chunk = []
        current_length = 0

        for split in good_splits:
            split_len = len(split)
            if current_length + split_len + len(separator) > chunk_size and current_chunk:
                merged = separator.join(current_chunk)
                final_chunks.append(merged)
                
                # Build overlap from end of current chunk
                overlap_splits = []
                overlap_len = 0
                for part in reversed(current_chunk):
                    if overlap_len + len(part) <= chunk_overlap:
                        overlap_splits.insert(0, part)
                        overlap_len += len(part)
                    else:
                        break
                current_chunk = overlap_splits
                current_length = sum(len(p) for p in current_chunk) + (len(current_chunk) * len(separator))

            current_chunk.append(split)
            current_length += split_len + len(separator)

        if current_chunk:
            final_chunks.append(separator.join(current_chunk))

        return final_chunks
