# ChronoVision
Seeing Time: Benchmarking Chronological Reasoning and Shortcut Biases in Vision-Language Models.

![main_image](./assets/main_github.svg)


The datasets of images are available at: https://huggingface.co/datasets/Q1anK/ChronoVision

use vllm to quick start:

(a sample of Qwen3-VL-4B-Instruct)

CUDA_VISIBLE_DEVICES=1,2 vllm serve ./Qwen3-VL-4B-Instruct\
  --served-model-name Qwen3-VL-4B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 32768 \
  --max-num-seqs 512
