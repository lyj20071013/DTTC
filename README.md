# DTTC: Dynamic Test-Time Computing for Lightweight Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: DeepSeek-R1-Distill-Qwen-1.5B](https://img.shields.io/badge/Model-DeepSeek--R1--1.5B-blue)](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
[![Task: Math Reasoning](https://img.shields.io/badge/Task-Mathematical_Reasoning-green)]()

> **Achievement:** Achieving **88.23%** accuracy on GSM8K using a single **1.5B parameter model** (DeepSeek-R1-Distill-Qwen-1.5B) via dynamic test-time computation strategies, without any fine-tuning.

## 📄 Abstract
This repository contains the official implementation and technical report for **DTTC (Dynamic Test-Time Computing)**. 

DTTC is a lightweight framework designed to enhance the reasoning capabilities of small language models (SLMs). By introducing a **Dynamic Parameter Pool (DPP)**, **Ambiguity Statement Mapping (ASM)**, and **Time-enhanced Penalty Decoding (TPD)**, we significantly bridge the gap between 1.5B models and larger state-of-the-art LLMs.

[**📄 Read the Technical Report (PDF)**](assets/DTTC_Technical_Report.pdf)

## 🚀 Key Results

Experiments were conducted on a single RTX 4090D (24GB).

| Benchmark | Baseline (CoT) | **DTTC (Ours)** | Improvement |
| :--- | :---: | :---: | :---: |
| **GSM8K** | 76.04% | **88.23%** | 🔺 +12.19% |
| **SVAMP** | 85.33% | **94.67%** | 🔺 +9.34% |
| **ASDiv** | 93.92% | **98.68%** | 🔺 +4.76% |
| **AQUA** | 63.39% | **81.49%** | 🔺 +18.10% |

## 🧩 Methodology

The framework consists of three synergistic components:

1.  **Dynamic Parameter Pool (DPP):** * Instead of a fixed temperature, we utilize a pool of diverse decoding configurations (e.g., balanced $T=0.6$, deterministic $T=0.5$, exploratory $T=0.8$).
    * The system dynamically switches strategies via a queue-based mechanism until a valid answer format is generated.

2.  **Ambiguity Statement Mapping (ASM):**
    * A robust regex-based module to handle linguistic ambiguities in math word problems (e.g., "3 sprints 3 times a week").
    * See `inference.py` for the extensive pattern matching rules.

3.  **Time-enhanced Penalty Decoding (TPD):**
    * A custom `LogitsProcessor` that suppresses thought-switching tokens (e.g., "Alternatively", "However") during early generation stages to enforce reasoning consistency.
