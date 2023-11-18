# Flash Attention

This contains script to setup and run the bloom model.

## Setup

### Prepare Environment

``` bash
# Create working directory
export BLOOM_DIR=/scratch/zs99/dz9214/BLOOM
export WORK_DIR=/scratch/zs99/dz9214/hpcai-2023-scripts # WORK_DIR can be the same as BLOOM_DIR
mkdir -p ${BLOOM_DIR}
mkdir -p ${WORK_DIR}
```

``` bash
time wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${BLOOM_DIR}/miniconda.sh
time bash ${BLOOM_DIR}/miniconda.sh -b -p ${HOME}/miniconda3
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda init
conda config --set auto_activate_base false
```

``` bash
module load cuda/11.7.0
```

### Setup Python Environment

``` bash
mkdir -p ${WORK_DIR}/pyenv
time conda create -p ${WORK_DIR}/pyenv/flash python=3.8 -y
export PYTHONPATH=${WORK_DIR}/pyenv/flash/lib/python3.8/site-packages
```

``` bash
time ${WORK_DIR}/pyenv/flash/bin/pip install torch==2.0.1
time ${WORK_DIR}/pyenv/flash/bin/pip install huggingface_hub==0.12.1
time ${WORK_DIR}/pyenv/flash/bin/pip install pydantic==1.10.2
time ${WORK_DIR}/pyenv/flash/bin/pip install transformers==4.26.1
time ${WORK_DIR}/pyenv/flash/bin/pip install deepspeed==0.7.6
time ${WORK_DIR}/pyenv/flash/bin/pip install accelerate==0.16.0
time ${WORK_DIR}/pyenv/flash/bin/pip install grpcio-tools==1.50.0 bitsandbytes flask flask_api fastapi==0.89.1 uvicorn==0.19.0 jinja2==3.1.2
time ${WORK_DIR}/pyenv/flash/bin/pip install flash-attn --no-build-isolation
```

### Prepare BLOOM Model

``` bash
mkdir -p ${BLOOM_DIR}/offline_models
```

Rsync from disk or download the model from the internet.

``` bash
# rsync the downloaded model from /scratch/public
time rsync -avP /scratch/public/HPC-AI-BLOOM/models--microsoft--bloom-deepspeed-inference-int8 ${BLOOM_DIR}/offline_models
```

``` bash
# download microsoft/bloom-deepspeed-inference-int8
${WORK_DIR}/pyenv/flash/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="microsoft/bloom-deepspeed-inference-int8",local_files_only=False,cache_dir="/scratch/zs99/dz9214/BLOOM/offline_models",ignore_patterns=["*.safetensors"],)'
```

Create symbolic link to the offline models

``` bash
rm -rf ${HOME}/.cache/huggingface/hub
ln -sf ${BLOOM_DIR}/offline_models ${HOME}/.cache/huggingface/hub
```

### Download and Build pdsh

```bash
# Clone the BLOOM Inference project
mkdir ${BLOOM_DIR}/github -p
cd ${BLOOM_DIR}/github
git clone https://github.com/huggingface/transformers-bloom-inference

# Install pdsh with ssh support
cd ${BLOOM_DIR}/github
git clone https://github.com/chaos/pdsh
cd pdsh
./bootstrap
./configure --prefix=${HOME}/.local --with-ssh
make install -j $(nproc)
```

### Setup DeepSpeed Environment

``` bash
# Add variables to ${HOME}/.deepspeed_env, we will start all training's from $HOME directory
echo "CUDA_HOME=/apps/cuda/11.6.1" > ${HOME}/.deepspeed_env
echo "NCCL_DEBUG=INFO" >> ${HOME}/.deepspeed_env
echo "TORCH_DISTRIBUTED_DETAIL=DEBUG" >> ${HOME}/.deepspeed_env
echo "TRANSFORMERS_OFFLINE=1" >> ${HOME}/.deepspeed_env
echo "LD_LIBRARY_PATH=/apps/cuda/11.6.1/extras/CUPTI/lib64:/apps/cuda/11.6.1/lib64" >> ${HOME}/.deepspeed_env
echo "PDSH_SSH_ARGS='-o StrictHostKeyChecking=no'" >> ${HOME}/.deepspeed_env
```

### Configure SSH authentication

``` bash
yes | ssh-keygen -t ecdsa -f ${HOME}/.ssh/id_ecdsa -N "" -vvv
ssh-keygen -y -f ${HOME}/.ssh/id_ecdsa >> ${HOME}/.ssh/authorized_keys
```

## Run Inference

``` bash
qsub flash_inference.sh
```
