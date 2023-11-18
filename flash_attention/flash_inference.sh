#!/bin/bash
#PBS -P zs99
#PBS -N Bloom-FlashAtn
#PBS -q gpuvolta
#PBS -l ncpus=96
#PBS -l ngpus=8
#PBS -l mem=300G
#PBS -l jobfs=800GB
#PBS -l storage=scratch/zs99
#PBS -l wd
#PBS -l walltime=00:10:00
#PBS -j oe
#PBS -o bloom-flashatn.txt
#PBS -M dzhang022@e.ntu.edu.sg
#PBS -m abe

date

module purge
module load cuda/11.7.0

export PATH="${HOME}/.local/bin:$PATH"
export BLOOM_DIR=/scratch/zs99/dz9214/BLOOM
export WORK_DIR=/scratch/zs99/dz9214/hpcai-2023-scripts

# export SCRIPT=${WORK_DIR}/00-submission/flash_attention/flash_inference.py
export SCRIPT=${WORK_DIR}/run.py
export LOG_DIR=${WORK_DIR}/flash_attention/logs

export PYTHONPATH=${WORK_DIR}/pyenv/flash/lib/python3.8/site-packages

cat ${PBS_NODEFILE} | cut -f 1 -d . | sed -e 's/$/ slots=48/' | sort -u >${HOME}/hostfile

cat ${HOME}/hostfile

mkdir -p ${LOG_DIR}

do_inference() {
    local batch_size="$1"
    local log_file="$2"

    time ${WORK_DIR}/pyenv/flash/bin/deepspeed --hostfile ${HOME}/hostfile --num_gpus 4 \
        ${SCRIPT} --name microsoft/bloom-deepspeed-inference-int8 \
        --batch_size ${batch_size} --dtype int8 --benchmark 2>&1 | tee ${LOG_DIR}/${log_file}
}

echo "======================================================================================"
echo "Running Warm Up..."
echo "======================================================================================"

# do_inference "1" "warmup.log"

echo "======================================================================================"
echo "Running Actual Run..."
echo "======================================================================================"

# for i in {1..3}; do
#     do_inference "13"  "bs13-run$i.log"
# done
do_inference "12" "bs-12.log"