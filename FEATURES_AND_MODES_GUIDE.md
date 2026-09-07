# OpenVINO™ LLM & RAG Studio: Features, Project Structure & Modes Guide

This comprehensive guide covers all features available in the **OpenVINO™ LLM & RAG Studio**, the **Guided Intel NPU Setup Wizard**, the **modular RAG project architecture**, instructions for **document uploading**, configuring **hardware accelerator modes** (Intel NPU, CPU, iGPU, dGPU), and tuning **RAG & generation profiles**.

---

## Table of Contents
1. [Overview & Project Architecture](#overview--project-architecture)
2. [Guided Intel NPU Setup Wizard & Diagnostics](#guided-intel-npu-setup-wizard--diagnostics)
   - [Hardware & Driver Verification](#step-1-hardware--driver-verification)
   - [1-Click Model Acquisition Catalog](#step-2-1-click-model-acquisition-catalog)
   - [Live Dry-Run Pipeline Benchmark](#step-3-live-dry-run-pipeline-benchmark)
   - [CLI Model Export with Optimum Intel](#step-4-cli-model-export-with-optimum-intel)
3. [Dual Engine Architecture (OpenVINO NPU + LM Studio)](#dual-engine-architecture-openvino-npu--lm-studio)
4. [Modular RAG System Architecture](#modular-rag-system-architecture)
   - [Document Ingestion & Multi-Format Parsing](#document-ingestion--multi-format-parsing)
   - [Semantic Chunking Engine](#semantic-chunking-engine)
   - [High-Speed Vector Indexing & Search](#high-speed-vector-indexing--search)
   - [Prompt Augmentation & Citation Tracking](#prompt-augmentation--citation-tracking)
5. [How to Upload & Manage Documents](#how-to-upload--manage-documents)
6. [Complete Features Breakdown](#complete-features-breakdown)
   - [Model Lifecycle Management (Start / Stop / Unload)](#1-model-lifecycle-management)
   - [Dual Engine Switching (OpenVINO vs LM Studio)](#2-dual-engine-switching-openvino-vs-lm-studio)
   - [RAG Mode vs Standard Chat Mode](#3-rag-mode-vs-standard-chat-mode)
   - [Real-Time Token & Reasoning Streaming](#4-real-time-token--reasoning-streaming)
   - [Safe Conversation Memorization](#5-safe-conversation-memorization)
   - [Persistent Multi-Session History](#6-persistent-multi-session-history)
   - [Live Hardware Telemetry & Metrics](#7-live-hardware-telemetry--metrics)
7. [Hardware Accelerator Modes Setup](#hardware-accelerator-modes-setup)
   - [Mode 1: Intel NPU (Intel AI Boost) - Recommended](#mode-1-intel-npu-intel-ai-boost---recommended)
   - [Mode 2: Intel Core Ultra CPU](#mode-2-intel-core-ultra-cpu)
   - [Mode 3: Intel Graphics (iGPU - GPU.0)](#mode-3-intel-graphics-igpu---gpu0)
   - [Mode 4: NVIDIA GeForce RTX 5070 Ti (dGPU - GPU.1)](#mode-4-nvidia-geforce-rtx-5070-ti-dgpu---gpu1)
   - [Mode 5: LM Studio Local Server (OpenAI REST /v1)](#mode-5-lm-studio-local-server-openai-rest-v1)
8. [Generation & RAG Parameter Tuning](#generation--rag-parameter-tuning)
9. [API Endpoints Reference](#api-endpoints-reference)

---

## Overview & Project Architecture

The application is structured into clean, decoupled layers separating the **Web UI**, the **Server/Lifecycle Manager**, the **NPU Setup Engine**, and the **Modular RAG Engine**:

```
Antigravity/
├── app.py                      # Main Python server, REST APIs & SSE stream controller
├── npu_setup.py                # ⚡ Intel NPU Diagnostics, Catalog & Hugging Face Downloader
├── start_app.bat               # 1-click Windows launcher
├── test_app.py                 # Core server & model lifecycle test suite
├── test_npu_setup.py           # NPU Guided Setup & diagnostics test suite
├── test_projects_and_history.py# Project workspaces & chat history test suite
├── test_lmstudio_integration.py# LM Studio dual-engine & reasoning test suite
├── test_rag_pipeline.py        # End-to-end RAG and document upload test suite
├── FEATURES_AND_MODES_GUIDE.md # Comprehensive user & developer guide
├── llama-3.2-3b-ov/            # OpenVINO IR model weights and tokenizer
├── documents/                  # Persistent storage folder for uploaded user documents
├── data/                       # Persistent vector index & session storage
│
├── rag/                        # 🧠 Modular RAG Engine Package
│   ├── __init__.py             # Exports RAGEngine, DocumentLoader, TextChunker, VectorStore, ProjectManager
│   ├── document_loader.py      # Multi-format document parser (PDF, TXT, MD, CSV, JSON, Code)
│   ├── chunker.py              # Semantic character & paragraph chunker with overlap
│   ├── vector_store.py         # In-memory TF-IDF/Cosine vector similarity search & persistence
│   ├── project_manager.py      # Project isolation, session persistence & chat distillation
│   └── rag_engine.py           # Orchestrator: Ingestion -> Query -> Retrieve -> Augment -> Citations
│
└── static/                     # 🎨 Frontend Web Dashboard
    ├── index.html              # Modern dashboard layout with NPU Wizard, Projects & Chat History
    ├── style.css               # Glassmorphism styling, wizard cards, progress bars & animations
    └── app.js                  # Setup wizard controller, upload handlers, RAG switcher & session manager
```

---

## Guided Intel NPU Setup Wizard & Diagnostics

If a user attempts to select or start the Intel NPU before configuring drivers or model weights, the application **automatically intercepts the action** and presents the **Guided Intel NPU Setup Wizard** (also accessible at any time via the **⚡ NPU Setup Guide** button in the top navigation bar).

### Step 1: Hardware & Driver Verification
- **Automated Probe**: Checks processor architecture and executes `ov.Core().available_devices` to verify the presence of the `NPU` device.
- **Driver Readiness**: If the NPU is not detected in Windows Device Manager under *Intel(R) AI Boost*, the wizard provides a direct link to the [Official Intel NPU Driver Download Page](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).
- **Troubleshooting Tips**: Explains how to verify BIOS NPU enablement and how to update device drivers.

### Step 2: 1-Click Model Acquisition Catalog
The wizard includes a built-in catalog of verified INT4 OpenVINO models optimized for Intel Core Ultra NPUs:

| Model Preset | Size | Hugging Face Repository | Description |
|---|---|---|---|
| **Llama 3.2 3B Instruct (INT4)** *(Recommended)* | 1.7 GB | `Godreign/llama-3.2-3b-instruct-openvino-int4-model` | Meta's state-of-the-art 3B instruction model. Best balance of speed and reasoning. |
| **DeepSeek-R1 Distill 1.5B (INT4)** | 1.1 GB | `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov` | Official OpenVINO distillation of DeepSeek-R1 with full thinking process. |
| **TinyLlama 1.1B Chat (INT4)** | 0.7 GB | `OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov` | Ultra-lightweight model with rapid NPU compilation and low RAM footprint. |
| **Phi-4 Mini 3.8B (INT4)** | 2.2 GB | `OpenVINO/Phi-4-mini-instruct-int4-ov` | Microsoft's high-capability compact model for logic and structured tasks. |

- **Live Download Progress**: Clicking **Download** triggers an asynchronous background thread with animated progress bars, downloaded MB, and elapsed time.
- **Custom Model Path**: Users can also specify any local folder containing `openvino_model.xml`, `openvino_model.bin`, and `openvino_tokenizer.xml`.

### Step 3: Live Dry-Run Pipeline Benchmark
- Compiles the selected model graph directly onto the NPU device.
- Measures **Compilation Latency (seconds)** and **Warm-up TTFT (Time to First Token)**.
- Renders a real-time test output response and a **"Start Chatting on NPU"** button.

### Step 4: CLI Model Export with Optimum Intel
For developers who wish to export custom Hugging Face checkpoints to OpenVINO INT4 format:
```bash
pip install optimum-intel[openvino]
optimum-cli export openvino --model meta-llama/Llama-3.2-3B-Instruct --weight-format int4 llama-3.2-3b-ov
```

---

## Dual Engine Architecture (OpenVINO NPU + LM Studio)

Antigravity AI Studio provides a seamless toggle between two independent LLM execution runtimes:

1. **⚡ Intel OpenVINO Pipeline**: Runs natively on the Intel Core Ultra NPU, CPU, or GPU using `openvino_genai`. Completely self-contained with no external server required.
2. **🦙 LM Studio Integration**: Connects over HTTP REST (`http://127.0.0.1:1234/v1`) to LM Studio. Enables running larger models on discrete GPUs (e.g., NVIDIA RTX 5070 Ti) and renders collapsible **Reasoning / Thought Process** blocks in real-time.

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
- Automatic disk serialization in `data/projects/{project_id}/rag_index.json`.

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

### 2. Dual Engine Switching (OpenVINO vs LM Studio)
- Seamlessly toggle between local NPU inference and remote LM Studio instances.
- Automatic model enumeration and status monitoring.

### 3. RAG Mode vs Standard Chat Mode
- **RAG Mode (Toggle Switch = ON)**: The model consults your uploaded knowledge base documents before answering, grounding responses in your private data and displaying source citations.
- **Standard Chat Mode (Toggle Switch = OFF)**: The model answers directly from its internal pre-trained weights without document lookup.

### 4. Real-Time Token & Reasoning Streaming
- HTTP Server-Sent Events (SSE) stream tokens as they are generated.
- Renders collapsible **DeepSeek-R1 Thought Process** blocks when reasoning models are selected.

### 5. Safe Conversation Memorization
- When ending a project chat session, click **"Memorize into RAG"** to distill Q&A pairs into a permanent knowledge artifact.
- Filters out pleasantries and provides an editable preview to prevent hallucination cycles.

### 6. Persistent Multi-Session History
- Saves all project conversation transcripts with timestamps and active engines.
- 1-click continuation restores complete chat contexts.

### 7. Live Hardware Telemetry & Metrics
- ⚡ **Tokens / Sec (TPS)**: Live generation throughput.
- ⏱️ **Time to First Token (TTFT)** in ms.
- 🔢 **Token Count** and **Total Duration**.
- 📚 **Sources Cited Badge**: Displays number of retrieved document snippets.
- 🖥️ **System Telemetry**: Top-bar gauges for **RAM %** and **CPU %**.

---

## Hardware Accelerator Modes Setup

### Mode 1: Intel NPU (`NPU` - Intel AI Boost) *(Recommended)*
- **Best For**: Ultra-low power consumption, silent operation, consistent ~15-16 tokens/s throughput.
- **Setup**: Open the **⚡ NPU Setup Guide**, select a model from the catalog, and click **"Start Model on NPU"**.

### Mode 2: Intel Core Ultra CPU (`CPU`)
- **Best For**: Dynamic shape inputs without pre-allocated buffer boundaries.
- **Setup**: Set **Device** to `Intel Core Ultra CPU` and click **"Start Model"**.

### Mode 3: Intel Graphics iGPU (`GPU.0`)
- **Best For**: Offloading computation from CPU cores to integrated Arc GPU.
- **Setup**: Set **Device** to `Intel Graphics (iGPU)` and click **"Start Model"**.

### Mode 4: NVIDIA GeForce RTX 5070 Ti dGPU (`GPU.1`)
- **Best For**: Maximum parallel raw compute power on AC power.
- **Setup**: Set **Device** to `NVIDIA RTX 5070 Ti (dGPU)` and click **"Start Model"**.

### Mode 5: LM Studio Local Server (OpenAI REST /v1)
- **Best For**: Running models up to 70B parameters with GPU offloading in LM Studio.
- **Setup**: Launch LM Studio, start the local server on port 1234, and select **LM Studio** in the app header.

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

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/npu/diagnostics` | Full NPU hardware, driver, and active model integrity diagnostic report |
| `GET` | `/api/npu/catalog` | List of pre-verified INT4 OpenVINO models for 1-click installation |
| `GET` | `/api/npu/download_status` | Polling endpoint for active background model downloads |
| `POST` | `/api/npu/download` | Trigger asynchronous download of an INT4 model from Hugging Face Hub |
| `POST` | `/api/npu/test` | Run live dry-run NPU pipeline compilation and latency benchmark |
| `POST` | `/api/npu/set_model_path` | Validate and switch active OpenVINO model directory |
| `GET` | `/api/engine/status` | Current engine mode, NPU state, and LM Studio connectivity |
| `POST` | `/api/engine/select` | Switch active engine (`openvino` vs `lmstudio`) |
| `GET` | `/api/projects` | List all project workspaces with document and session stats |
| `POST` | `/api/projects/create` | Create a new isolated project workspace |
| `POST` | `/api/projects/select` | Switch active project context |
| `POST` | `/api/projects/delete` | Delete project workspace and its vector store |
| `GET` | `/api/sessions` | List saved conversations (optionally filtered by `?project_id=`) |
| `POST` | `/api/sessions/save` | Save or update conversation transcript |
| `GET` | `/api/sessions/load?id=` | Load full conversation turns and citations |
| `POST` | `/api/sessions/distill` | Strip pleasantries and generate structured Q&A summary |
| `POST` | `/api/sessions/memorize` | Index distilled conversation into project RAG |
| `GET` | `/api/documents?project_id=` | List indexed files in the specified project |
| `POST` | `/api/documents/upload` | Upload & chunk document (supports Base64 binary & text) |
| `POST` | `/api/documents/delete` | Remove file and update project vector store |
| `POST` | `/api/chat/stream` | Multi-engine SSE streaming endpoint with RAG context injection |
