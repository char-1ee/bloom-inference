# Run bigscience/bloom inference with vLLM vram optimization with Page Attention:
#   - inference and monitor GPU usage on single GPU
#   - TODO monitor multiple devices
#   - TODO timing
#   - TODO adapt distributed inference
# Prerequsite:
#   conda activate vllm/p38
#   pip install ray, vllm

import torch
import subprocess
import threading
import time
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "DeepSpeed is a machine learning framework",
    "He is working on",
    "He has a",
    "He got all",
    "Everyone is happy and I can",
    "The new movie that got Oscar this year",
    "In the far far distance from our galaxy,",
    "Peace is the only way",
]

model_path = "/home/users/industry/ai-hpc/apacsc14/scratch/MyBloom/offline_models/models--bigscience--bloom"

torch.cuda.empty_cache()
device0 = torch.device("cuda:0")

def get_gpu_memory_usage(device):
    """
    Get the current GPU memory usage.
    """
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2  # Memory usage in MB

def monitor_gpu_memory(device):
    while True:
        memory_usage = get_gpu_memory_usage(device)
        print(f"GPU Memory Usage: {memory_usage:.2f} MB")
        subprocess.run(["nvidia-smi"])
        time.sleep(1)  # Adjust the interval as needed

# Start the GPU memory monitoring thread
monitoring_thread = threading.Thread(target=monitor_gpu_memory, args=(device0,))
monitoring_thread.start()

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Create an LLM.
llm = LLM(model=model_path, trust_remote_code=True)

# For distributed inference
# llm = LLM(model="bigscience/bloom", tensor_parallel_size=8)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Ensure the monitoring thread exits
monitoring_thread.join()

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# Clean up
torch.cuda.empty_cache()
