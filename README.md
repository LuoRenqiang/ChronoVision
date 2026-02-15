<div align="center">

# ⏳ Seeing Time: Benchmarking Chronological Reasoning and Shortcut Biases in VLMs

<a href="https://huggingface.co/datasets/Q1anK/ChronoVision"><img src="https://img.shields.io/badge/Data-HuggingFace-yellow.svg" alt="Data"></a>
<a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/Inference-vLLM-blue.svg" alt="vLLM"></a>

![main_image](./assets/main_github.svg)

</div>

## 📖 Introduction

This repository contains the official code and evaluation data for the paper **"Seeing Time: Benchmarking Chronological Reasoning and Shortcut Biases in Vision-Language Models"**.

[cite_start]We introduce **ChronoVision**, a novel benchmark designed to evaluate how Vision-Language Models (VLMs) perceive and reason about time[cite: 57]. Unlike existing benchmarks that focus on simple frame sequencing, our work delves into:
- [cite_start]**🕵️‍♂️ Chronological Logic:** Reasoning about historical artifacts and object evolution[cite: 58].
- [cite_start]**📰 Cross-Modal Alignment:** Synchronizing visual evidence with time-sensitive news text[cite: 59].
- [cite_start]**⚠️ Shortcut Bias Detection:** Exposing the "grayscale equals old" heuristic that many SOTA models suffer from[cite: 60, 61].

---

## 📚 Datasets

[cite_start]Our benchmark consists of three specialized datasets, meticulously curated to test different aspects of temporal reasoning[cite: 59, 244].

**📥 Download:** [HuggingFace Dataset](https://huggingface.co/datasets/Q1anK/ChronoVision)

| Dataset | Full Name | Focus | Scope |
| :--- | :--- | :--- | :--- |
| **CHA** | Chinese Historical Artifacts | Fine-grained artifact evolution | [cite_start]Tang to Qing Dynasties (887 images) [cite: 268] |
| **SPEED** | Sports, Politics, Electronics, Emergency, Diversity | Modern event chronology | [cite_start]1952 - 2025 (1,028 images) [cite: 426, 427] |
| **HistNews**| Historical News | Text-Image chronological alignment | [cite_start]1946 - 2025 (400 events) [cite: 450] |

---

## 🏆 Leaderboard

[cite_start]Here is a summary of the zero-shot performance of representative VLMs on our benchmark (Score range: 0-100)[cite: 846].

| Model | Type | Overall Score | Artifacts Acc | Shortcut Score | News Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Gemini-2.5-Pro** | Closed | **67.17** 🥇 | 44.23 | 78.75 | 55.58 |
| **GPT-5.2** | Closed | 49.96 🥈 | 40.14 | 56.83 | 49.73 |
| **Qwen3-VL-235B** | Open | 49.92 🥉 | 49.94 | 62.36 | 33.80 |
| **Qwen3-VL-8B** | Open | 44.47 | 38.43 | 61.32 | 26.13 |
| **InternVL3.5-8B** | Open | 29.06 | 24.86 | 24.92 | 12.43 |

> [cite_start]**Note:** We identify a significant performance gap between closed-source and open-source models, though large-scale open models are catching up[cite: 1051, 1053].

---

## 🚀 Quick Start with vLLM

We recommend using [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference. Below is an example script to serve the **Qwen3-VL-4B-Instruct** model.

### 1. Install Requirements
```bash
pip install vllm
