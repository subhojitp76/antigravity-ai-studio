"""
Unit and Integration Tests for Intel NPU Guided Setup and Diagnostics System.
Tests NPUDiagnostics, NPUModelDownloader, Catalog, and REST API Endpoints.
"""
import os
import sys
import time
import json
import socket
import threading
import unittest
import urllib.request
import urllib.parse
from pathlib import Path

from npu_setup import NPUDiagnostics, NPUModelDownloader, NPU_MODEL_CATALOG
from app import StudioHTTPHandler, ThreadingHTTPServer, model_mgr, project_mgr, npu_downloader

class TestNPUSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Pick a free random port for testing HTTP server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        cls.port = sock.getsockname()[1]
        sock.close()

        cls.server = ThreadingHTTPServer(('127.0.0.1', cls.port), StudioHTTPHandler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get_json(self, endpoint):
        req = urllib.request.Request(f"{self.base_url}{endpoint}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def post_json(self, endpoint, data=None):
        payload = json.dumps(data or {}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def test_01_npu_diagnostics_hardware_info(self):
        """Verify NPUDiagnostics returns valid hardware and driver info."""
        diag = NPUDiagnostics.get_hardware_info()
        self.assertIsInstance(diag, dict)
        self.assertIn("processor", diag)
        self.assertIn("devices", diag)
        self.assertIn("has_npu", diag)
        self.assertIn("openvino_installed", diag)
        self.assertIn("status", diag)
        self.assertIn("driver_url", diag)
        self.assertTrue(diag["has_npu"])
        self.assertEqual(diag["status"], "ready")
        print(f"\n[Test 01] NPU Hardware Diagnostics: status={diag['status']}, devices={diag['devices']}, has_npu={diag['has_npu']}")

    def test_02_npu_diagnostics_model_dir_check(self):
        """Verify model directory validation."""
        # Non-existent directory
        res_fake = NPUDiagnostics.check_model_directory("non_existent_dir_12345")
        self.assertFalse(res_fake["is_valid"])
        self.assertFalse(res_fake["exists"])
        self.assertGreater(len(res_fake["missing_files"]), 0)

        # Local llama-3.2-3b-ov directory if present
        if os.path.isdir("llama-3.2-3b-ov"):
            res_real = NPUDiagnostics.check_model_directory("llama-3.2-3b-ov")
            self.assertTrue(res_real["is_valid"])
            self.assertTrue(res_real["exists"])
            self.assertTrue(res_real["has_weights"])
            self.assertTrue(res_real["has_tokenizer"])
            self.assertTrue(res_real["has_config"])
            self.assertEqual(len(res_real["missing_files"]), 0)
            print(f"[Test 02] Verified local llama-3.2-3b-ov model directory integrity: is_valid=True ({res_real['total_size_mb']} MB)")

    def test_03_npu_catalog_metadata(self):
        """Verify model catalog structure and entries."""
        self.assertIsInstance(NPU_MODEL_CATALOG, list)
        self.assertGreater(len(NPU_MODEL_CATALOG), 0)
        for item in NPU_MODEL_CATALOG:
            self.assertIn("id", item)
            self.assertIn("name", item)
            self.assertIn("size", item)
            self.assertIn("repo_id", item)
            self.assertIn("recommended", item)
            self.assertIn("description", item)
            self.assertIn("target_dir", item)
        print(f"[Test 03] Verified {len(NPU_MODEL_CATALOG)} catalog presets.")

    def test_04_api_npu_diagnostics_endpoint(self):
        """Test GET /api/npu/diagnostics."""
        status, data = self.get_json('/api/npu/diagnostics')
        self.assertEqual(status, 200)
        self.assertIn("hardware", data)
        self.assertIn("active_model", data)
        self.assertIn("catalog", data)
        self.assertIn("overall_ready", data)
        self.assertIn("status_code", data)
        print(f"[Test 04] GET /api/npu/diagnostics returned overall_ready={data['overall_ready']}, status_code={data['status_code']}")

    def test_05_api_npu_catalog_endpoint(self):
        """Test GET /api/npu/catalog."""
        status, data = self.get_json('/api/npu/catalog')
        self.assertEqual(status, 200)
        self.assertIn("catalog", data)
        self.assertIsInstance(data["catalog"], list)
        print(f"[Test 05] GET /api/npu/catalog returned {len(data['catalog'])} models.")

    def test_06_api_npu_download_status_endpoint(self):
        """Test GET /api/npu/download_status."""
        status, data = self.get_json('/api/npu/download_status')
        self.assertEqual(status, 200)
        self.assertIn("status", data)
        self.assertIn("progress_pct", data)
        print(f"[Test 06] GET /api/npu/download_status returned status={data['status']}, progress_pct={data['progress_pct']}")

    def test_07_api_npu_set_model_path_validation(self):
        """Test POST /api/npu/set_model_path."""
        # Test invalid path
        status_invalid, data_invalid = self.post_json('/api/npu/set_model_path', {"model_path": "invalid_path_xyz"})
        self.assertEqual(status_invalid, 400)
        self.assertFalse(data_invalid.get("success"))

        # Test valid path
        if os.path.isdir("llama-3.2-3b-ov"):
            status_valid, data_valid = self.post_json('/api/npu/set_model_path', {"model_path": "llama-3.2-3b-ov"})
            self.assertEqual(status_valid, 200)
            self.assertTrue(data_valid.get("success"))
            print("[Test 07] POST /api/npu/set_model_path successfully validated model directory.")

if __name__ == '__main__':
    unittest.main()
