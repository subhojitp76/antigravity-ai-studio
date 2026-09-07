"""
NPU Setup, Diagnostics, and Model Acquisition Module for OpenVINO AI Studio.
Provides automated hardware detection, Intel NPU driver guidance, INT4 OpenVINO model catalog,
asynchronous Hugging Face downloads with progress tracking, and dry-run pipeline benchmarks.
"""

import os
import sys
import time
import json
import threading
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Try OpenVINO imports safely
try:
    import openvino as ov
    import openvino_genai as ov_genai
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    ov = None
    ov_genai = None

# Try Hugging Face Hub import safely
try:
    from huggingface_hub import snapshot_download, HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


INTEL_NPU_DRIVER_URL = "https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html"

# Catalog of verified, pre-converted OpenVINO INT4 models optimized for Intel NPU
NPU_MODEL_CATALOG = [
    {
        "id": "llama-3.2-3b",
        "name": "Llama 3.2 3B Instruct (INT4)",
        "repo_id": "Godreign/llama-3.2-3b-instruct-openvino-int4-model",
        "target_dir": "llama-3.2-3b-ov",
        "size": "1.7 GB",
        "context_len": 2048,
        "recommended": True,
        "badge": "Recommended",
        "description": "State-of-the-art 3B instruction model by Meta. Excellent balance of speed, reasoning, and memory efficiency on Intel NPU."
    },
    {
        "id": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1 Distill Qwen 1.5B (INT4)",
        "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-gq-ov",
        "target_dir": "deepseek-r1-1.5b-ov",
        "size": "1.1 GB",
        "context_len": 2048,
        "recommended": False,
        "badge": "Reasoning",
        "description": "Official OpenVINO INT4 distillation of DeepSeek-R1. Outputs step-by-step thinking process before final answers."
    },
    {
        "id": "tinyllama-1.1b",
        "name": "TinyLlama 1.1B Chat (INT4)",
        "repo_id": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "target_dir": "tinyllama-1.1b-ov",
        "size": "0.7 GB",
        "context_len": 2048,
        "recommended": False,
        "badge": "Ultra Lightweight",
        "description": "Ultra-fast compact model with rapid compilation times. Ideal for quick testing and low memory environments."
    },
    {
        "id": "phi-4-mini",
        "name": "Phi-4 Mini 3.8B (INT4)",
        "repo_id": "OpenVINO/Phi-4-mini-instruct-int4-ov",
        "target_dir": "phi-4-mini-ov",
        "size": "2.2 GB",
        "context_len": 2048,
        "recommended": False,
        "badge": "Math & Code",
        "description": "Microsoft's high-capability compact model featuring strong mathematical logic, structured reasoning, and coding skills."
    }
]


class NPUDiagnostics:
    """Performs deep hardware, driver, dependency, and model inspections for NPU."""

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Detects processor, OS platform, and OpenVINO hardware devices."""
        sys_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "openvino_installed": HAS_OPENVINO,
            "openvino_version": ov.__version__ if HAS_OPENVINO else None,
            "has_npu": False,
            "npu_name": "Intel NPU (AI Boost)",
            "devices": [],
            "driver_url": INTEL_NPU_DRIVER_URL,
            "status": "unknown",
            "message": ""
        }

        if not HAS_OPENVINO:
            sys_info["status"] = "missing_openvino"
            sys_info["message"] = "OpenVINO packages are not installed in Python environment."
            return sys_info

        try:
            core = ov.Core()
            available = list(core.available_devices)
            sys_info["devices"] = available

            if "NPU" in available:
                sys_info["has_npu"] = True
                sys_info["status"] = "ready"
                sys_info["message"] = "Intel NPU is detected and ready for OpenVINO GenAI execution."
            else:
                # NPU device not exposed to OpenVINO
                # Check if processor looks like Intel Core Ultra
                proc_str = (platform.processor() or "").lower()
                sys_info["has_npu"] = False
                sys_info["status"] = "missing_driver_or_hw"
                sys_info["message"] = (
                    "NPU device not found by OpenVINO. Ensure Intel NPU Driver is installed "
                    "or verify your system has an Intel Core Ultra processor with AI Boost."
                )
        except Exception as e:
            sys_info["status"] = "error"
            sys_info["message"] = f"Error probing OpenVINO Core: {e}"

        return sys_info

    @staticmethod
    def check_model_directory(model_path: str) -> Dict[str, Any]:
        """Validates whether a model folder exists and has all required OpenVINO IR files."""
        path = Path(model_path)
        result = {
            "path": str(path.resolve()),
            "exists": path.exists() and path.is_dir(),
            "is_valid": False,
            "missing_files": [],
            "found_files": [],
            "total_size_mb": 0.0,
            "has_weights": False,
            "has_tokenizer": False,
            "has_config": False
        }

        if not result["exists"]:
            result["missing_files"] = [
                "openvino_model.xml",
                "openvino_model.bin",
                "openvino_tokenizer.xml",
                "openvino_tokenizer.bin",
                "config.json"
            ]
            return result

        # Required files checklist
        req_weights = ["openvino_model.xml", "openvino_model.bin"]
        req_tokenizer = ["openvino_tokenizer.xml", "openvino_tokenizer.bin"]
        opt_tokenizer = ["tokenizer.json"]

        total_bytes = 0
        existing_files = set()
        for item in path.iterdir():
            if item.is_file():
                existing_files.add(item.name)
                total_bytes += item.stat().st_size
                result["found_files"].append(item.name)

        result["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)

        # Check weights
        result["has_weights"] = all(f in existing_files for f in req_weights)
        # Check tokenizer
        result["has_tokenizer"] = (all(f in existing_files for f in req_tokenizer) or 
                                  "openvino_tokenizer.xml" in existing_files or 
                                  "tokenizer.json" in existing_files)
        # Check config
        result["has_config"] = "config.json" in existing_files or "openvino_config.json" in existing_files

        missing = []
        for req in req_weights:
            if req not in existing_files:
                missing.append(req)

        if not result["has_tokenizer"]:
            missing.append("openvino_tokenizer.xml")

        result["missing_files"] = missing
        result["is_valid"] = (result["has_weights"] and result["has_tokenizer"] and result["total_size_mb"] > 10.0)

        return result

    @classmethod
    def get_full_report(cls, active_model_path: str = None) -> Dict[str, Any]:
        """Generates a complete diagnostic report with readiness score and troubleshooting steps."""
        hw = cls.get_hardware_info()
        
        # Check active model path or default path
        model_p = active_model_path or "llama-3.2-3b-ov"
        model_check = cls.check_model_directory(model_p)

        # Check all models in catalog to see which ones are already locally available
        catalog_status = []
        for item in NPU_MODEL_CATALOG:
            item_check = cls.check_model_directory(item["target_dir"])
            catalog_status.append({
                **item,
                "is_installed": item_check["is_valid"],
                "installed_size_mb": item_check["total_size_mb"],
                "local_path": item_check["path"]
            })

        # Calculate overall readiness
        overall_ready = hw["has_npu"] and model_check["is_valid"]
        
        status_code = "ready"
        recommendations = []

        if not hw["openvino_installed"]:
            status_code = "missing_dependencies"
            recommendations.append({
                "step": "Install OpenVINO GenAI",
                "action": "pip install openvino openvino-genai",
                "type": "terminal"
            })
        elif not hw["has_npu"]:
            status_code = "needs_driver"
            recommendations.append({
                "step": "Install Intel NPU Driver",
                "action": INTEL_NPU_DRIVER_URL,
                "type": "link",
                "description": "Download and run the official Intel NPU Driver installer for Windows, then restart the application."
            })
        elif not model_check["is_valid"]:
            status_code = "needs_model"
            recommendations.append({
                "step": "Download or Export NPU Model",
                "action": "Use the 1-Click Model Downloader in the Setup Wizard or export via optimum-cli.",
                "type": "wizard",
                "description": f"Model directory '{model_p}' is missing required OpenVINO IR files."
            })

        return {
            "timestamp": time.time(),
            "overall_ready": overall_ready,
            "status_code": status_code,
            "hardware": hw,
            "active_model": {
                "configured_path": model_p,
                "details": model_check
            },
            "catalog": catalog_status,
            "recommendations": recommendations,
            "hf_hub_available": HAS_HF_HUB
        }

    @classmethod
    def test_npu_pipeline(cls, model_path: str, device: str = "NPU") -> Dict[str, Any]:
        """Executes a dry-run compilation and 5-token warm-up inference on NPU."""
        if not HAS_OPENVINO:
            return {"success": False, "error": "OpenVINO is not installed."}

        model_check = cls.check_model_directory(model_path)
        if not model_check["is_valid"]:
            return {
                "success": False,
                "error": f"Model directory '{model_path}' is invalid or missing weights: {', '.join(model_check['missing_files'])}"
            }

        start_compile = time.perf_counter()
        try:
            print(f"[NPUDiagnostics] Running test pipeline compilation on {device} ({model_path})...")
            npu_config = {}
            if device.startswith("NPU"):
                npu_config = {
                    "MAX_PROMPT_LEN": 512,
                    "MIN_RESPONSE_LEN": 128,
                    "GENERATE_HINT": "BEST_PERF"
                }
                pipe = ov_genai.LLMPipeline(model_path, device, **npu_config)
            else:
                pipe = ov_genai.LLMPipeline(model_path, device)

            compile_duration = round(time.perf_counter() - start_compile, 2)
            print(f"[NPUDiagnostics] Test compilation successful in {compile_duration}s!")

            # Run 5-token test generation
            start_gen = time.perf_counter()
            test_prompt = "Hello! Please confirm you are working."
            history = ov_genai.ChatHistory()
            history.append({"role": "user", "content": test_prompt})
            
            output = pipe.generate(history, max_new_tokens=12, temperature=0.7)
            gen_duration = round(time.perf_counter() - start_gen, 2)

            del pipe
            del history

            return {
                "success": True,
                "device": device,
                "compile_duration_s": compile_duration,
                "test_prompt": test_prompt,
                "test_output": str(output),
                "generation_duration_s": gen_duration,
                "message": f"NPU pipeline successfully validated in {compile_duration}s compilation!"
            }
        except Exception as e:
            return {
                "success": False,
                "device": device,
                "error": str(e),
                "message": f"NPU pipeline test failed: {e}"
            }


class NPUModelDownloader:
    """Manages asynchronous model downloads from Hugging Face with progress tracking."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or ".").resolve()
        self.lock = threading.Lock()
        self.state = {
            "status": "idle",             # 'idle' | 'downloading' | 'completed' | 'error'
            "model_id": None,
            "repo_id": None,
            "target_dir": None,
            "progress_pct": 0,
            "downloaded_mb": 0,
            "total_mb": 0,
            "current_file": "",
            "speed_mbps": 0.0,
            "error": None,
            "start_time": None,
            "elapsed_s": 0
        }
        self.worker_thread = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            st = dict(self.state)
            if st["status"] == "downloading" and st["start_time"]:
                st["elapsed_s"] = round(time.time() - st["start_time"], 1)
            return st

    def start_download(self, model_id_or_repo: str, target_dir_name: str = None) -> Dict[str, Any]:
        """Initiates an asynchronous download job from Hugging Face."""
        with self.lock:
            if self.state["status"] == "downloading":
                return {"success": False, "message": "A model download is already in progress."}

            # Find matching catalog preset or treat as raw HuggingFace repo
            matched = next((m for m in NPU_MODEL_CATALOG if m["id"] == model_id_or_repo or m["repo_id"] == model_id_or_repo), None)
            if matched:
                repo_id = matched["repo_id"]
                target_dir = matched["target_dir"]
                model_name = matched["name"]
            else:
                repo_id = model_id_or_repo
                target_dir = target_dir_name or Path(repo_id).name.lower()
                model_name = repo_id

            dest_path = self.base_dir / target_dir

            self.state = {
                "status": "downloading",
                "model_id": model_id_or_repo,
                "model_name": model_name,
                "repo_id": repo_id,
                "target_dir": str(dest_path),
                "progress_pct": 0,
                "downloaded_mb": 0,
                "total_mb": 0,
                "current_file": "Connecting to Hugging Face...",
                "speed_mbps": 0.0,
                "error": None,
                "start_time": time.time(),
                "elapsed_s": 0
            }

        self.worker_thread = threading.Thread(
            target=self._download_worker,
            args=(repo_id, dest_path),
            daemon=True
        )
        self.worker_thread.start()

        return {
            "success": True,
            "message": f"Download started for '{model_name}'.",
            "repo_id": repo_id,
            "target_dir": str(dest_path)
        }

    def _download_worker(self, repo_id: str, dest_path: Path):
        """Worker thread executing snapshot download."""
        try:
            print(f"[NPUModelDownloader] Downloading {repo_id} to {dest_path}...")
            dest_path.mkdir(parents=True, exist_ok=True)

            # Define files to include
            allow_patterns = [
                "*.xml", "*.bin", "*.json", "*.jinja", "*.txt",
                "openvino_model.*", "openvino_tokenizer.*", "openvino_detokenizer.*"
            ]

            start_t = time.time()
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest_path),
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
                resume_download=True
            )

            # Verify downloaded model
            check = NPUDiagnostics.check_model_directory(str(dest_path))
            duration = round(time.time() - start_t, 1)

            with self.lock:
                if check["is_valid"]:
                    self.state["status"] = "completed"
                    self.state["progress_pct"] = 100
                    self.state["downloaded_mb"] = check["total_size_mb"]
                    self.state["total_mb"] = check["total_size_mb"]
                    self.state["current_file"] = "Installation Complete!"
                    print(f"[NPUModelDownloader] Download completed for {repo_id} ({check['total_size_mb']} MB in {duration}s)")
                else:
                    self.state["status"] = "error"
                    self.state["error"] = f"Downloaded files incomplete: missing {', '.join(check['missing_files'])}"
        except Exception as e:
            err_msg = str(e)
            print(f"[NPUModelDownloader] Download error: {err_msg}")
            with self.lock:
                self.state["status"] = "error"
                self.state["error"] = err_msg
