"""
Document Loader Module for RAG System.
Supports TXT, Markdown, CSV, JSON, Code (PY, JS, CPP, HTML, etc.), and PDF files.
"""

import os
import re
import json
import csv
from pathlib import Path


class Document:
    """Represents an ingested document with content and metadata."""
    def __init__(self, content: str, doc_id: str, filename: str, file_type: str, metadata: dict = None):
        self.content = content
        self.doc_id = doc_id
        self.filename = filename
        self.file_type = file_type
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "metadata": self.metadata,
            "char_count": len(self.content)
        }


class DocumentLoader:
    """Multi-format document loader."""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".markdown": "markdown",
        ".csv": "csv",
        ".json": "json",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".html": "code",
        ".css": "code",
        ".cpp": "code",
        ".c": "code",
        ".h": "code",
        ".rs": "code",
        ".java": "code",
        ".log": "text",
        ".pdf": "pdf"
    }

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def load_file(cls, file_path: str, doc_id: str = None) -> Document:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        filename = path.name
        doc_id = doc_id or f"doc_{int(path.stat().st_mtime)}_{path.stem}"
        file_type = cls.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        metadata = {
            "size_bytes": path.stat().st_size,
            "modified_time": path.stat().st_mtime,
            "extension": ext
        }

        if ext in [".txt", ".md", ".markdown", ".log", ".py", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".h", ".rs", ".java"]:
            content = cls._load_text_file(path)
        elif ext == ".csv":
            content = cls._load_csv_file(path)
        elif ext == ".json":
            content = cls._load_json_file(path)
        elif ext == ".pdf":
            content = cls._load_pdf_file(path)
        else:
            # Fallback text reader
            content = cls._load_text_file(path)

        return Document(
            content=content,
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            metadata=metadata
        )

    @classmethod
    def _load_text_file(cls, path: Path) -> str:
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    @classmethod
    def _load_csv_file(cls, path: Path) -> str:
        rows_text = []
        for enc in ["utf-8", "latin-1"]:
            try:
                with open(path, "r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == 0:
                            header = ", ".join(row)
                            rows_text.append(f"Columns: {header}")
                        else:
                            rows_text.append(f"Row {i}: {', '.join(row)}")
                return "\n".join(rows_text)
            except Exception:
                continue
        return cls._load_text_file(path)

    @classmethod
    def _load_json_file(cls, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        except Exception:
            return cls._load_text_file(path)

    @classmethod
    def _load_pdf_file(cls, path: Path) -> str:
        """Extract text from PDF using lightweight stream parsing or fallback."""
        text_parts = []
        try:
            # Simple pure Python PDF text stream extractor
            with open(path, "rb") as f:
                content = f.read()
            
            # Find stream objects in PDF
            streams = re.findall(rb"stream[\r\n]+([\s\S]*?)[\r\n]+endstream", content)
            for s in streams:
                try:
                    # Decompress if zlib compressed
                    import zlib
                    decompressed = zlib.decompress(s)
                    # Extract text inside parentheses in BT ... ET blocks
                    matches = re.findall(rb"\((.*?)\)\s*Tj", decompressed)
                    for m in matches:
                        try:
                            text_parts.append(m.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
                except Exception:
                    # Not compressed or raw text
                    matches = re.findall(rb"\((.*?)\)\s*Tj", s)
                    for m in matches:
                        try:
                            text_parts.append(m.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
            
            extracted = " ".join(text_parts).strip()
            if len(extracted) > 50:
                return extracted
        except Exception as e:
            print(f"[DocumentLoader] PDF extraction note: {e}")

        # Fallback if binary streams didn't contain clean text
        return f"[PDF Document: {path.name} (Binary Content - {path.stat().st_size} bytes)]"
