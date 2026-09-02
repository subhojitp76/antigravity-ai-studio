import os
import openvino_genai as ov_genai

# Path to the exported OpenVINO model
model_path = os.path.abspath("llama-3.2-3b-ov")
device = "NPU"  # Target device: Intel NPU

print(f"Loading OpenVINO model from: {model_path}")
print(f"Target device: {device}")

# Pipeline configuration
config = {
    "MAX_PROMPT_LEN": 1024,
    "MIN_RESPONSE_LEN": 128,
    "GENERATE_HINT": "BEST_PERF"
}

try:
    print("Compiling model for NPU...")
    pipe = ov_genai.LLMPipeline(model_path, device, **config)
    print("Model loaded successfully on NPU!")

    prompt = "Tell me a short fun fact about space."
    print(f"\nUser: {prompt}\nAI: ", end="", flush=True)

    def streamer(subword):
        print(subword, end="", flush=True)
        return False

    pipe.generate(prompt, max_new_tokens=100, streamer=streamer)
    print("\n\nDone!")

except Exception as e:
    print(f"\nError running model on {device}: {e}")
