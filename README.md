# Civil_Engineering_LLM_Assistant
A domain-specific AI assistant for Civil Engineering, fine-tuned on Mistral 7B using QLoRA optimization on a T4 GPU.

# Civil Engineering AI Assistant (Mistral 7B + QLoRA)

![Project Status](https://img.shields.io/badge/Status-Complete-green)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![GPU](https://img.shields.io/badge/GPU-NVIDIA_T4-orange)

## 🏗️ Project Overview
This project adapts the **Mistral 7B** Large Language Model (LLM) to the specialized domain of **Civil Engineering**. By utilizing **Parameter-Efficient Fine-Tuning (PEFT)** and **QLoRA (4-bit quantization)**, I successfully fine-tuned this 7-billion parameter model on a resource-constrained **NVIDIA T4 GPU** (15GB VRAM).

The resulting model demonstrates improved factual recall on technical topics such as **contract management**, **structural analysis**, and **geotechnical engineering codes**, bridging the gap between general-purpose AI and specialized engineering requirements.

## 🎯 Key Technical Features
* **Model:** Mistral 7B (v0.1) Base Model.
* **Technique:** QLoRA (Quantized Low-Rank Adaptation) for memory efficiency.
* **Optimization:** Implemented **Gradient Checkpointing** and **Paged Optimizers** to prevent OOM errors on the T4 GPU.
* **Data:** Custom-curated dataset of Civil Engineering Q&A pairs formatted for instruction tuning.

## 🛠️ Installation & Usage

### 1. Open in Google Colab
The easiest way to replicate this training is to run the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](LINK_TO_YOUR_NOTEBOOK_HERE)

*(Note: Replace `LINK_TO_YOUR_NOTEBOOK_HERE` with the actual URL of your file in the repo)*

### 2. Local Inference
To run the model locally (requires GPU):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Base Model
model_id = "mistralai/Mistral-7B-v0.1"
base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

# Load Fine-Tuned Adapter
adapter_path = "path/to/downloaded/adapter"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Run Inference...
