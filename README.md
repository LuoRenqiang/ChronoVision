# 🎉Seeing Time
## Benchmarking Chronological Reasoning and Shortcut Biases in Vision-Language Models.

This repository contains the official code and data for the paper 'Seeing Time: Benchmarking Chronological Reasoning and Shortcut Biases in Vision-Language Models.'. We introduce a novel benchmark to evaluate chronological reasoning in VLMs across historical artifacts, modern events, and cross-modal news alignment. We also expose the 'grayscale equals old' shortcut bias.

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
