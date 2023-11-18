# BLOOM

This folder contains scripts and outputs for BLOOM batch inference on cluster environment (2 nodes, each with 4 V100 GPUs). The goal is to increase BLOOM offline inference throughput under GPU memory restriction.

## File Structure

``` text
.
├── baseline/                   scripts for baseline inference
├── best_results/               output from the best running results
├── extract_benchmark.pl        a useful script to extract benchmark from output files
├── faster_transformer/         scripts for Faster Transformer (Experiment on Aspire2A)
├── flash_attention/            scripts for FlashAttention
└── README.md                   this file
```

## Best Output

The best configuration's output is in `best_results` folder. To extract throughput, run the `extract_benchmark.pl` script:

``` bash
$ perl extract_benchmark.pl best_results/best_log1.txt best_results/best_log2.txt best_results/best_log3.txt best_results/best_log.txt

File 1: Throughput per token including tokenize: 17.12 msecs
File 1: Start to ready to generate: 48.803 secs
File 1: Tokenize and generate 6500 tokens: 22.266 secs
File 1: Start to finish: 71.069 secs
File 2: Throughput per token including tokenize: 17.12 msecs
File 2: Start to ready to generate: 49.731 secs
File 2: Tokenize and generate 6500 tokens: 22.260 secs
File 2: Start to finish: 71.991 secs
File 3: Throughput per token including tokenize: 17.10 msecs
File 3: Start to ready to generate: 48.865 secs
File 3: Tokenize and generate 6500 tokens: 22.235 secs
File 3: Start to finish: 71.100 secs
File 4: Throughput per token including tokenize: 17.06 msecs
File 4: Start to ready to generate: 210.822 secs
File 4: Tokenize and generate 6500 tokens: 22.176 secs
File 4: Start to finish: 232.999 secs
***
Average Throughput per token including tokenize: 17.1 msecs
```
