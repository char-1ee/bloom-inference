# vLLM

vLLM is a fast and easy-to-use library for LLM inference and serving.

## Featured

* Efficient management of attention key and value memory with PagedAttention
* Continuous batching of incoming request and optimized CUDA kernels
* **24x better than HuggingFace Transformers**

## vLLM single card batch inference

Refer to the `vLLM_inference.py`

## vLLM distributed batch inference

0. Install Dependency:

```bash
pip install vllm ray
```

1. Start ray server on one node:

```bash
ray start --haed
```

2. SSH to other nodes and run ray worker:

```bash
ray start --address=<ray-head-address>
```

3. Run Inference

```bash
from vllm import LLM
llm = LLM("bigscience/bloom", tensor_parallel_size=8)
output = llm.generate("San Franciso is a")
```

Notice that 

- For single card (V100, 32GB) cannot serve full parameter model (OOM)

* For multiple card, vLLM seems only support client-server inference pattern (OpenAI API service) to do distributed inference, which not works in Aspire 2A or Gadi offline environment.

## vLLM online inference

vLLM can be deployed to server and provides OpenAI style online inference service

```bash
python -m vllm.entrypoints.openai.api_server --model /mnt/offline_model/bloom-7b1
```

Send a request

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/mnt/offline_model/bloom-7b1",
        "prompt": "vLLM is",
        "max_tokens": 10,
        "temperature": 0.8
    }'
```

Then we got the response

```bash
{"id":"cmpl-6a971f9a46ce4870bcf965611b2c0508","object":"text_completion","created":1694439858,"model":"/mnt/offline_model/bloom-7b1","choices":[{"index":0,"text":"vLLM is efficient libaray for LLM inference","logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":39,"total_tokens":45,"completion_tokens":6}}。
```

## Reference

[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/)
