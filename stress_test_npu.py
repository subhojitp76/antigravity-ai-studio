"""
Stress Test Script for Intel NPU under OpenVINO GenAI.
Tests throughput (tokens/sec), latency, memory stability, and consecutive inference calls.
"""

import time
import os
import psutil
import openvino_genai as ov_genai

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_stress_test(num_iterations=5, prompt_len="medium"):
    model_path = os.path.abspath("llama-3.2-3b-ov")
    device = "NPU"

    prompts = {
        "short": "What is 2+2?",
        "medium": "Explain the concept of quantum computing in simple terms for high school students.",
        "long": "Draft a comprehensive design document for a high-throughput microservices architecture handling 100k requests per second with distributed caching and failover."
    }

    selected_prompt = prompts.get(prompt_len, prompts["medium"])

    print("=" * 60)
    print(f" INTEL NPU STRESS TEST ({num_iterations} Iterations)")
    print(f" Model: {model_path}")
    print(f" Prompt Mode: {prompt_len} ({len(selected_prompt)} chars)")
    print("=" * 60)

    config = {
        "MAX_PROMPT_LEN": 1024,
        "MIN_RESPONSE_LEN": 128,
        "GENERATE_HINT": "BEST_PERF"
    }

    print("\n[1/3] Compiling Model to NPU...")
    compile_start = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model_path, device, **config)
    compile_time = time.perf_counter() - compile_start
    print(f"  -> Model compiled in {compile_time:.2f} seconds.")
    print(f"  -> Baseline Memory: {get_process_memory():.1f} MB\n")

    print("[2/3] Executing Stress Iterations...")
    results = []

    for i in range(1, num_iterations + 1):
        token_count = 0
        first_token_time = None
        gen_start = time.perf_counter()

        def stress_streamer(word):
            nonlocal token_count, first_token_time
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            return False

        pipe.generate(selected_prompt, max_new_tokens=150, streamer=stress_streamer)
        gen_end = time.perf_counter()

        total_time = gen_end - gen_start
        ttft = (first_token_time - gen_start) * 1000 if first_token_time else 0
        tps = token_count / total_time if total_time > 0 else 0
        mem = get_process_memory()

        results.append({
            "iteration": i,
            "tokens": token_count,
            "total_time": total_time,
            "ttft_ms": ttft,
            "tps": tps,
            "mem_mb": mem
        })

        print(f"  Iteration {i:02d}: {token_count:3d} tokens in {total_time:.2f}s | TTFT: {ttft:.1f}ms | TPS: {tps:.2f} tok/s | RAM: {mem:.1f} MB")

    print("\n[3/3] Performance Summary:")
    print("-" * 60)
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    final_mem = get_process_memory()

    print(f"  Total Generated Tokens: {total_tokens}")
    print(f"  Average Throughput:     {avg_tps:.2f} tokens/second")
    print(f"  Average Time-to-First:  {avg_ttft:.2f} ms")
    print(f"  Memory Stability Delta: {final_mem - results[0]['mem_mb']:+.1f} MB")
    print("=" * 60)

if __name__ == "__main__":
    run_stress_test(num_iterations=5, prompt_len="medium")
