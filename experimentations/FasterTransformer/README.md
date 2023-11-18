# FasterTransformers Benchmark

FasterTransformers rewrites the CUDA libraries used in common Transformer model. We convert bigscience/bloom-7b1 model into FasterTransformer model and benchmark its inference performance with original Transformer structure.

## Preparation

Prepare a conda environment before start

```bash
export BLOOM_DIR=/home/users/industry/ai-hpc/apacsc14/scratch/MyBloom
export MODEL_DIR=/home/users/industry/ai-hpc/apacsc14/scratch/MyBloom/offline_models
mkdir -p ${BLOOM_DIR}
mkdir -p ${MODEL_DIR}

# Download bigscience/bloom-7b1 model
time ${BLOOM_DIR}/deepspeed076/p38/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bigscience/bloom-7b1',local_files_only=False,cache_dir='/home/users/industry/ai-hpc/apacsc14/scratch/MyBloom/offline_models',ignore_patterns=['*.safetensors'],)" 
```

```bash
# Pull FasterTransformer source code
mkdir FasterTransformers-fork
git clone https://github.com/tingshua-yts/FasterTransformer.git
cd FasterTransformer-fork/FasterTransformer
```

## Model conversion

```bash
# Convert model with FasterTransformers script
python  examples/pytorch/gpt/utils/huggingface_bloom_convert.py  \
        -i ${MODEL_DIR}/bloom-7b1 \
        -o ${MODEL_DIR}/bloom-7b1-ft-fp16 \
        -tp 8 \
        -dt fp16 \
        -p 64 -v
```

## Build FasterTransformers

```bash
cd ${BLOOM_DIR}
docker run -ti --shm-size 5g --rm nvcr.io/nvidia/tensorflow:22.09-tf1-py3 bash
git clone https://github.com/NVIDIA/FasterTransformer.git
mkdir -p FasterTransformer/build
cd FasterTransformer/build
git submodule init && git submodule update
```

```bash
cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_MULTI_GPU=ON -DBUILD_PYT=ON -DENABLE_FP8=OFF ..
make -j 48
```

## Benchmarking

Before start benchmarking, change the `FasterTransformer/examples/pytorh/gpt/bloom_lambda.py` to support distributed inference.

```python
if args.test_hf:
        # Load HF's pretrained model for testing.
        if args.multi_gpu:
            model = transformers.AutoModelForCausalLM.from_pretrained(
            args.tokenizer_path, device_map="balanced_low_0", torch_dtype=torch.float16)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                args.tokenizer_path, torch_dtype=torch.float16).cuda()

        return model, tokenizer
```

Start benchmarking HuggingFace

```bash
export CUDA_VISIBLE_DEVICES=0,1
python ../../examples/pytorch/gpt/bloom_lambada.py \
    --tokenizer-path ${MODEL_DIR}/bloom-7b1 \
    --dataset-path ${BLOOM_DIR}/data/lambada/lambada_test.jsonl \
    --batch-size 16 \
    --test-hf \
    --multi_gpu \
    --show-progress
```

Start benchmarking FasterTransformers

```bash
ckp=${MODEL_DIR}/bloom-7b1-ft-fp16/8-gpu
workd_size=2
mpirun --allow-run-as-root -n $workd_size python ../../examples/pytorch/gpt/bloom_lambada.py \
    --lib-path ${BLOOM_DIR}/FasterTransformer/build/lib/libth_transformer.so \
    --checkpoint-path $ckp \
    --batch-size 16 \
    --tokenizer-path ${MODEL_DIR}/bloom-7b1 \
    --dataset-path ${BLOOM_DIR}/data/lambada/lambada_test.jsonl \
    --show-progress
```
