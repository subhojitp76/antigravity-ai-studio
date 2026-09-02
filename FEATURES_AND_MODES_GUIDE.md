# OpenVINO™ LLM & RAG Studio: Features, Project Structure & Modes Guide

This comprehensive guide covers all features available in the **OpenVINO™ LLM & RAG Studio**, the **modular RAG project architecture**, instructions for **document uploading**, configuring **hardware accelerator modes** (Intel NPU, CPU, iGPU, dGPU), and tuning **RAG & generation profiles**.

---

## Table of Contents
1. [Overview & Project Architecture](#overview--project-architecture)
2. [Dual Engine Architecture (OpenVINO NPU + LM Studio)](#dual-engine-architecture-openvino-npu--lm-studio)
3. [Modular RAG System Architecture](#modular-rag-system-architecture)
   - [Document Ingestion & Multi-Format Parsing](#document-ingestion--multi-format-parsing)
   - [Semantic Chunking Engine](#semantic-chunking-engine)
   - [High-Speed Vector Indexing & Search](#high-speed-vector-indexing--search)
   - [Prompt Augmentation & Citation Tracking](#prompt-augmentation--citation-tracking)
4. [How to Upload & Manage Documents](#how-to-upload--manage-documents)
5. [Complete Features Breakdown](#complete-features-breakdown)
   - [Model Lifecycle Management (Start / Stop / Unload)](#1-model-lifecycle-management)
   - [Dual Engine Switching (OpenVINO vs LM Studio)](#2-dual-engine-switching-openvino-vs-lm-studio)
   - [RAG Mode vs Standard Chat Mode](#3-rag-mode-vs-standard-chat-mode)
   - [Real-Time Token & Reasoning Streaming](#4-real-time-token--reasoning-streaming)
   - [Live Hardware Telemetry & Metrics](#5-live-hardware-telemetry--metrics)
   - [Markdown & Code Highlighting](#6-markdown--code-highlighting)
   - [Export & Session Persistence](#7-export--session-persistence)
6. [Hardware Accelerator Modes Setup](#hardware-accelerator-modes-setup)
   - [Mode 1: Intel NPU (Intel AI Boost) - Recommended](#mode-1-intel-npu-intel-ai-boost---recommended)
   - [Mode 2: Intel Core Ultra CPU](#mode-2-intel-core-ultra-cpu)
   - [Mode 3: Intel Graphics (iGPU - GPU.0)](#mode-3-intel-graphics-igpu---gpu0)
   - [Mode 4: NVIDIA GeForce RTX 5070 Ti (dGPU - GPU.1)](#mode-4-nvidia-geforce-rtx-5070-ti-dgpu---gpu1)
   - [Mode 5: LM Studio Local Server (OpenAI REST /v1)](#mode-5-lm-studio-local-server-openai-rest-v1)
7. [Generation & RAG Parameter Tuning](#generation--rag-parameter-tuning)
8. [API Endpoints Reference](#api-endpoints-reference)

---

## Overview & Project Architecture

The application is structured into clean, decoupled layers separating the **Web UI**, the **Server/Lifecycle Manager**, and the **Modular RAG Engine**:

```
Antigravity/
├── app.py                      # Main Python server, REST APIs & SSE stream controller
├── start_app.bat               # 1-click Windows launcher
├── test_app.py                 # Core server & model lifecycle test suite
├── test_rag_pipeline.py        # End-to-end RAG and document upload test suite
├── FEATURES_AND_MODES_GUIDE.md # Comprehensive user & developer guide
├── llama-3.2-3b-ov/            # OpenVINO IR model weights and tokenizer
├── documents/                  # Persistent storage folder for uploaded user documents
├── data/                       # Persistent vector index storage (rag_index.json)
│
├── rag/                        # 🧠 Modular RAG Engine Package
│   ├── __init__.py             # Exports RAGEngine, DocumentLoader, TextChunker, VectorStore
│   ├── document_loader.py      # Multi-format document parser (PDF, TXT, MD, CSV, JSON, Code)
│   ├── chunker.py              # Semantic character & paragraph chunker with overlap
│   ├── vector_store.py         # In-memory TF-IDF/Cosine vector similarity search & persistence
│   └── rag_engine.py           # Orchestrator: Ingestion -> Query -> Retrieve -> Augment -> Citations
│
└── static/                     # 🎨 Frontend Web Dashboard
    ├── index.html              # Modern dashboard layout with upload dropzone & citation pills
    ├── style.css               # Glassmorphism styling, responsive layout, animations
    └── app.js                  # Document upload handlers, RAG mode switch, citation renderer
```

---

## Modular RAG System Architecture

The RAG engine is organized into 4 distinct, reusable components inside `rag/`:

```
User Document (.pdf / .txt / .md / .csv / .py)
                   │
                   ▼
       [ rag/document_loader.py ]  --> Extracted raw text + file metadata
                   │
                   ▼
          [ rag/chunker.py ]        --> Overlapping semantic text chunks (450 chars, 80 overlap)
                   │
                   ▼
        [ rag/vector_store.py ]    --> Sub-millisecond TF-IDF vector matrix + JSON persistence
                   │
                   ▼
     User Question in Chat
                   │
                   ▼
        [ rag/rag_engine.py ]      --> Top-K semantic similarity search
                   │
                   ├──> Injects Context into Llama-3.2-3B prompt
                   └──> Emits [Source Citations] with relevance scores to UI
```

### Document Ingestion & Multi-Format Parsing (`rag/document_loader.py`)
- **Supported Formats**: `.pdf`, `.txt`, `.md`, `.markdown`, `.csv`, `.json`, `.py`, `.js`, `.ts`, `.html`, `.css`, `.cpp`, `.c`, `.h`, `.rs`, `.java`, `.log`.
- Auto-handles character encodings (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`).
- Tabular CSV parser converts structured columns and rows into readable semantic blocks.

### Semantic Chunking Engine (`rag/chunker.py`)
- Splits documents using hierarchical boundary separators (`\n\n`, `\n`, `. `, `? `, `! `, `, `, ` `).
- Configurable chunk size (default `450` characters) and overlap (default `80` characters) ensures continuity between chunks.
- Every chunk tracks its `doc_id`, `filename`, `chunk_index`, and file type.

### High-Speed Vector Indexing & Search (`rag/vector_store.py`)
- Powered by `scikit-learn`'s `TfidfVectorizer` (with `(1,2)` n-grams and sublinear term frequency) and cosine similarity matrix multiplication.
- Instant search latency (< 1ms) with zero heavy external database overhead.
- Automatic disk serialization in `data/rag_index.json`.

### Prompt Augmentation & Citation Tracking (`rag/rag_engine.py`)
- Retrieves top-K chunks above a relevance score threshold.
- Injects verified reference blocks into the LLM context.
- Returns citation objects containing `citation_id`, `filename`, `score_percent`, and exact quote snippets.

---

## How to Upload & Manage Documents

### Method 1: Drag-and-Drop in Sidebar
1. Open the left sidebar in the web app.
2. Drag any supported file (`.txt`, `.md`, `.pdf`, `.csv`, `.json`, `.py`, etc.) directly into the **"Click or Drop Documents"** box.
3. The file is parsed, chunked, and indexed immediately.

### Method 2: Paperclip Upload in Chat Bar
1. Click the **Paperclip (📎)** icon next to the chat prompt input.
2. Select your file(s) from the file dialog.

### Managing Indexed Documents
- View total indexed files, chunk counts, and character lengths in the sidebar.
- Click the **Trash (🗑️)** icon next to any document to delete it and rebuild the index.
- Click **"Clear All"** to flush the entire knowledge base.

---

## Complete Features Breakdown

### 1. Model Lifecycle Management
- **Start / Compile**: Loads and compiles the model graph onto Intel NPU, CPU, or GPU on demand.
- **Unload / Stop**: Frees all pipeline tensors and system RAM.
- **Mid-Generation Interrupt**: Stop output instantly mid-sentence with the red Stop button.

### 2. RAG Mode vs Standard Chat Mode
- **RAG Mode (Toggle Switch = ON)**: The model consults your uploaded knowledge base documents before answering, grounding responses in your private data and displaying source citations.
- **Standard Chat Mode (Toggle Switch = OFF)**: The model answers directly from its internal pre-trained weights without document lookup.

### 3. Real-Time Token Streaming
- HTTP Server-Sent Events (SSE) stream tokens as they are generated.
- Blinking cyan caret and smooth typography.

### 4. Live Hardware Telemetry & Metrics
- ⚡ **Tokens / Sec (TPS)**: Live generation throughput.
- ⏱️ **Time to First Token (TTFT)** in ms.
- 🔢 **Token Count** and **Total Duration**.
- 📚 **Sources Cited Badge**: Displays number of retrieved document snippets.
- 🖥️ **System Telemetry**: Top-bar gauges for **RAM %** and **CPU %**.

### 5. Markdown & Code Highlighting
- Auto-formats code blocks with language tags and a one-click **Copy Code** button.

### 6. Export & Session Persistence
- Saves chat history and settings in browser LocalStorage.
- Click **"Export Chat"** to download a Markdown (`.md`) transcript including all questions, answers, and source citations.

---

## Hardware Accelerator Modes Setup

### Mode 1: Intel NPU (`NPU` - Intel AI Boost) *(Recommended)*
- **Best For**: Ultra-low power consumption, silent operation, consistent ~15-16 tokens/s throughput.
- **Setup**: Set **Device** to `Intel NPU (AI Boost)` and click **"Start Model"**.

### Mode 2: Intel Core Ultra CPU (`CPU`)
- **Best For**: Dynamic shape inputs without pre-allocated buffer boundaries.
- **Setup**: Set **Device** to `Intel Core Ultra CPU` and click **"Start Model"**.

### Mode 3: Intel Graphics iGPU (`GPU.0`)
- **Best For**: Offloading computation from CPU cores to integrated Arc GPU.
- **Setup**: Set **Device** to `Intel Graphics (iGPU)` and click **"Start Model"**.

### Mode 4: NVIDIA GeForce RTX 5070 Ti dGPU (`GPU.1`)
- **Best For**: Maximum parallel raw compute power on AC power.
- **Setup**: Set **Device** to `NVIDIA RTX 5070 Ti (dGPU)` and click **"Start Model"**.

---

## Generation & RAG Parameter Tuning

| Parameter | Default | Recommended Range | Description |
|---|---|---|---|
| **Top-K Context Chunks** | `3` | `1` - `6` | Number of most relevant document chunks injected into prompt |
| **Temperature** | `0.7` | `0.0` (Code/Logic) to `1.2` (Creative) | Modulates randomness in token selection |
| **Top-P** | `0.9` | `0.1` to `1.0` | Nucleus sampling probability cutoff |
| **Max New Tokens** | `512` | `64` to `2048` | Maximum output response length |
| **Repetition Penalty** | `1.1` | `1.0` to `2.0` | Penalizes repetitive phrases |

---

## API Endpoints Reference

| Method | Endpoint | Description | Payload Example |
|---|---|---|---|
| `GET` | `/api/devices` | Lists available hardware devices. | *None* |
| `GET` | `/api/status` | Returns model state, compile duration, and RAM/CPU metrics. | *None* |
| `GET` | `/api/documents` | Lists all indexed documents with chunk stats. | *None* |
| `POST` | `/api/documents/upload` | Ingests and indexes a new document. | `{"filename": "notes.md", "content": "..."}` |
| `POST` | `/api/documents/delete` | Removes a document from knowledge base. | `{"doc_id": "doc_123"}` |
| `POST` | `/api/documents/clear` | Clears all documents and vector index. | `{}` |
| `POST` | `/api/chat/stream` | Streams SSE tokens with optional RAG augmentation. | `{"message": "...", "rag_enabled": true, "top_k": 3}` |
| `POST` | `/api/chat/stop_generation` | Interrupts current generation immediately. | `{}` |
| `POST` | `/api/chat/reset` | Resets conversation context and system prompt. | `{"system_prompt": "..."}` |
