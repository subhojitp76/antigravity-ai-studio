# ⚡ Antigravity AI Studio

> **High-Performance Local LLM & Project-Scoped RAG Studio powered by Intel® OpenVINO™ (NPU / CPU / GPU) and LM Studio.**

[![OpenVINO](https://img.shields.io/badge/Intel-OpenVINO_2025.0-0071C5?style=flat-square&logo=intel&logoColor=white)](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
[![Hardware](https://img.shields.io/badge/Hardware-Intel_NPU_%7C_CPU_%7C_GPU-00C7FF?style=flat-square)](https://www.intel.com)
[![LM Studio](https://img.shields.io/badge/Integration-LM_Studio_REST-8B5CF6?style=flat-square)](https://lmstudio.ai/)
[![RAG](https://img.shields.io/badge/RAG-Project--Scoped_Vector_Store-10B981?style=flat-square)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-orange?style=flat-square)](LICENSE)

---

## 🌟 Overview

**Antigravity AI Studio** is a unified, privacy-first local AI workstation designed for running Large Language Models with **hardware acceleration on Intel Core Ultra NPUs** and seamless dual-engine integration with **LM Studio**.

It features an isolated **Project-Scoped Retrieval-Augmented Generation (RAG)** architecture, multi-session persistent chat history with sliding-window compaction, and **Safe Conversation Memorization** to prevent hallucination feedback loops.

```
+-----------------------------------------------------------------------------------------+
|                                ANTIGRAVITY AI STUDIO                                   |
+-----------------------------------------------------------------------------------------+
|  [💬 Direct Chat]           [📁 Projects Workspace]           [🕒 Searchable History]   |
+-----------------------------+---------------------------------+-------------------------+
|  Dual Engine Toggle:        |  Isolated Knowledge Bases:     |  Persistent Sessions:   |
|   ⚡ OpenVINO (Intel NPU)    |   • PDF, TXT, MD, CSV, Code     |   • Multi-turn restore  |
|   🦙 LM Studio (Port 1234)  |   • Sub-ms TF-IDF Vector Search |   • NPU Sliding Window  |
|                             |   • Safe Q&A Distillation       |   • Markdown Export     |
+-----------------------------+---------------------------------+-------------------------+
|  Hardware Telemetry:        |  Reasoning Support:             |  UI / UX:               |
|   • Live Tokens/Sec (TPS)   |   • DeepSeek-R1 Thinking Block  |   • Responsive Glass    |
|   • Real-Time TTFT (ms)     |   • Interactive Citation Pills  |   • Global Drag & Drop  |
+-----------------------------------------------------------------------------------------+
```

---

## 🚀 Key Features

### 1. ⚡ OpenVINO™ NPU Acceleration
- Native execution on **Intel® AI Boost NPU** via `openvino_genai.LLMPipeline`.
- Zero CPU/dGPU power draw during generation with static KV-cache optimization (`MAX_PROMPT_LEN: 1024`, `GENERATE_HINT: BEST_PERF`).
- Support for Intel Core Ultra CPUs, Intel Graphics (iGPU), and NVIDIA RTX dGPUs.

### 2. 🦙 LM Studio Local Server Integration
- Connects automatically to local LM Studio instances on `http://127.0.0.1:1234/v1`.
- Auto-detects and enumerates loaded models (e.g. `deepseek-r1-0528-qwen3-8b`, `qwen3.5-9b`, `gemma-4-12b-qat`).
- Native streaming of **Reasoning / Thought Process** tokens into collapsible interactive UI cards.

### 3. 📁 Project-Scoped RAG Architecture
- **Zero Context Pollution**: Documents uploaded to Project A are strictly isolated from Project B.
- **Multi-Format Parsers**: PDF (binary & text), TXT, Markdown, CSV, JSON, Python, C++, and log files.
- **Clickable Citation Chips**: Displays exact source file matches, percentage similarity scores, and expandable snippet previews.

### 4. 🧠 Safe Conversation Memorization (Anti-Hallucination)
- Distills concluded chat sessions into clean Q&A insight documents.
- Strips conversational pleasantries and filler to avoid polluting the vector database.
- Provides an **editable review modal** before indexing memories into the project knowledge base.

### 5. 🕒 Persistent Multi-Session Chat History
- Chronological, searchable timeline of all previous conversations.
- 1-click continuation restores complete conversation threads, citations, and metrics.
- Automatic sliding-window compaction preserves system prompts and recent context without overflowing the NPU buffer.

---

## 🏗️ System Architecture

```mermaid
graph TD
    Client[Web UI Dashboard] -->|HTTP / SSE Stream| Server[Studio Backend app.py]
    
    subgraph Engine Routing
        Server -->|Engine Switch| Router{Active Engine?}
        Router -->|OpenVINO| NPU[Intel NPU / CPU / GPU Pipeline]
        Router -->|LM Studio| LMS[LM Studio REST API :1234]
    end
    
    subgraph Multi-Workspace Storage
        Server --> PM[Project Manager]
        PM --> P1[Project: Default]
        PM --> P2[Project: Andromeda]
        PM --> P3[Project: Hyperion]
        
        P2 --> DOCS[documents/]
        P2 --> VEC[rag_index.json TF-IDF]
        P2 --> SESS[sessions/ history]
    end
    
    subgraph RAG Pipeline
        DOCS --> Loader[Multi-Format Loader]
        Loader --> Chunker[Semantic Chunker]
        Chunker --> VectorStore[Vector Store]
        VectorStore -->|Top-K Match| PromptAugmenter[Context Injector]
        PromptAugmenter --> Router
    end
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **Processor** | Intel® Core™ Ultra (Meteor Lake / Lunar Lake / Arrow Lake) | Intel® Core™ Ultra 7 / 9 with Intel® AI Boost NPU |
| **RAM** | 16 GB | 32 GB LPDDR5x / DDR5 |
| **OS** | Windows 11 (64-bit) / Linux (Ubuntu 22.04+) | Windows 11 23H2+ with latest Intel NPU Drivers |
| **Optional dGPU** | NVIDIA RTX 3060+ | NVIDIA RTX 4070 / 5070 Ti (for dual-engine setups) |

---

## 📦 Installation & Quickstart

### Step 1: Clone Repository
```bash
git clone https://github.com/subhojitp76/antigravity-ai-studio.git
cd antigravity-ai-studio
```

### Step 2: Create Python Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Obtain OpenVINO Model (or use LM Studio)
To export Llama 3.2 3B to OpenVINO IR format:
```bash
pip install optimum-intel[openvino]
optimum-cli export openvino --model meta-llama/Llama-3.2-3B-Instruct --weight-format int4 llama-3.2-3b-ov
```
*(Or download a pre-quantized OpenVINO model folder directly into `./llama-3.2-3b-ov`)*

---

## 🎯 Running the Studio

### Windows (One-Click Batch)
Double-click `start_app.bat` or run:
```cmd
start_app.bat
```

### Linux / macOS
```bash
chmod +x start_app.sh
./start_app.sh
```

### Python CLI Options
```bash
# Start server on default port 7860
python app.py

# Custom port without auto-opening browser
python app.py --port 8080 --no-browser
```

Open your browser at **`http://localhost:7860`**.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
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

---

## 🧪 Automated Test Suite

Run the full automated test suite to verify NPU acceleration, LM Studio streaming, RAG vector retrieval, and session resumption:

```bash
# Test 1: Project-Scoped RAG, Sessions & Memorization
python test_projects_and_history.py

# Test 2: LM Studio Dual-Engine Streaming & Reasoning
python test_lmstudio_integration.py

# Test 3: OpenVINO NPU RAG Pipeline Verification
python test_rag_pipeline.py
```

---

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
