# ------------------------------------ PEFT: LoRA, QLoRA, & Hyperparameters ------------------------------------
"""
Mastering Parameter-Efficient Fine-Tuning (PEFT) with a focus on:
- LoRA (Low-Rank Adaptation): 
    ✔ Problem - Many LLMs have billions of parameters, making them expensive to fine-tune.  These models are 
        made up of many layers (32 groups of modules, etc.), often called decoder layers in transformer 
        architectures.  Each layer includes:
        * Self-Attention: Learns relationships between words.
        * MLP (multi-layer perceptron): Learns complex patterns.
        * SiLU activation: A smooth activation function.
        * LayerNorm: Keeps values in a stable range during training.
    ✔ Solution - LoRA allows fine-tuning by adding small trainable matrices to the frozen base model weights.
        * You freeze the original model weights and only train these small matrices.
        * This drastically reduces the number of trainable parameters.
        * LoRA adapters can be added to any transformer model.
    ✔ How it works - It decomposes weight updates into low-rank matrices

- QLoRA: 
    ✔ Problem - Fine-tuning large models requires significant memory and compute resources.
    ✔ Solution - QLoRA combines 2 Major Tricks:
        1) Quantization - To reduce memory footprint. Ex. compresses 16-bit weights to 4-bit, reducing memory by 75%.
        2) LoRA - To enable large model tuning on consumer GPUs.
    ✔ How it works - Load the base model with less bit precision, freeze the model weights, then apply LoRA adapters to 
        fine-tune it.  Note - This quanitizes the base model weights, not the LoRA adapters.  The LoRA adapters are still
        trained in full precision (32-bit floats), allowing them to learn effectively while keeping the base model
        memory-efficient.
- Hyperparameters: 
    ✔ Problem - Fine-tuning LLMs requires careful tuning of hyperparameters to achieve optimal performance.
    ✔ Solution - Hyperparameters are tunable settings that control the training process - Not the model itself.  They 
        affect how the model learns, adapts, or behaves during training.  You can set them manually before training
        starts.  
    ✔ How it works - Key QLoRA hyperparamenters (LoRA-Specific):
        - r (rank):         Controls the size of A & B matrices (bottleneck dimensions).  e.g. 5, 8, 16
        - lora_alpha:       Scaling factor for the LoRA updates.  e.g. 16, 32, 64
        - lora_dropout: 	Randomly drops adapter weights during training for regularization.  e.g. 0.05, 0.1
        - target_modules: 	Specifies which layers to inject LoRA into.  e.g., q_proj, v_proj
        - bias:             Whether to include or freeze bias parameters  e.g. "none", "lora_only", "all"
    ✔ How Hyperparmeters affect performance:
        - Want faster training?	    Increase learning_rate, but carefully
        - Model is overfitting? 	Add lora_dropout, reduce num_train_epochs
        - Model is underfitting?	Increase r, lora_alpha, or epochs
        - Running out of memory?	Reduce r, batch_size, or increase gradient_accumulation_steps

Benefits:
✔ Drastically reduces memory and compute needs
✔ Enables scalable, low-cost fine-tuning for LLMs
✔ Great for task-specific adaptation (e.g., domain-specific QA, LCMS lab data)

Use PEFT to:
- Customize LLMs for niche domains
- Deploy powerful models in constrained environments
- Fine-tune multiple adapters for different use cases

| **Concept**           | **What It Does**                                 | **Relationship**                          |
|-----------------------|--------------------------------------------------|-------------------------------------------|
| **PEFT**              | Strategy for fine-tuning large models efficiently| LoRA and QLoRA are specific PEFT methods  |
| **LoRA**              | Adds small trainable adapter matrices (A & B)    | A type of PEFT                            |
| **QLoRA**             | LoRA + 4-bit quantized base model                | A memory-efficient extension of LoRA      |
| **Hyperparameters**   | Settings that control training behavior          | Applied to both LoRA and QLoRA setups     |

"""


# ------------------------------------ Activating Virtual Environment w/Jupyter Lab ----------------------------------
# From Anaconda PowerShell Prompt:
#     cd "C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering"
#     conda activate llms
#     Juptyer lab


# ------------------------------------ Package Install Instructions ----------------------------------
# pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# pip install -q requests bitsandbytes==0.43.0 transformers==4.48.3 accelerate==1.3.0
# pip install -q datasets requests peft


# ------------------------------------ Imports ----------------------------------
import os
import dotenv
from dotenv import load_dotenv
import re
import math
from tqdm import tqdm
# from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime


# ------------------------------------ Log In to LLM API Platforms ----------------------------------
# Load API Keys from local .env file (for LLM fine-tuning later)
env_path = r"C:\Users\Laptop\Desktop\Coding\LLM\Projects\llm_engineering\.env"
load_dotenv(dotenv_path=env_path, override=True)

# Print out available API keys for safety check
# openai_api_key = os.getenv('OPENAI_API_KEY')
# anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
# google_api_key = os.getenv('GOOGLE_API_KEY')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

print("Checking API Keys...\n")
# if openai_api_key: print(f"OpenAI Key found: {openai_api_key[:10]}...")
# if anthropic_api_key: print(f"Anthropic Key found: {anthropic_api_key[:10]}...")
# if google_api_key: print(f"Google Key found: {google_api_key[:10]}...")
if huggingface_token: print(f"HuggingFace Token found: {huggingface_token[:10]}...")

print("\n------------------------------------\n")
# Log into Hugging Face (necessary for items.py tokenizer)
login(huggingface_token, add_to_git_credential=True)

# openai = OpenAI()
# claude = Anthropic()


# ------------------------------------ Model Config ----------------------------------
# Base model and fine-tuned model identifiers
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = f"ed-donner/pricer-2024-09-13_13.04.39"

# Define LoRA config hyperparameters
LORA_R = 32
LORA_ALPHA = 64
# Attention Layers
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"] 


# ------------------------------------ Load (No Quantization) Base Model (initial test, full precision) ----------------------------------
# QLoRA loads base model in 4-bit to save memory
print("\n📦 Loading 4-bit Quantized Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
print(f"🧠 4-bit Quantized Base Model Memory Footprint: {base_model.get_memory_footprint() / 1e9:.2f} GB")



# ------------------------------------ Load 8-bit Quantized Model (first optimization) ----------------------------------
print("\n📦 Loading 8-bit Quantized Base Model...")
quant_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
print(f"🧠 8-bit Quantized Base Model Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")


# ------------------------------------ Load 4-bit Quantized Model (QLoRA) ----------------------------------
print("\n📦 Loading 4-bit Quantized Model (QLoRA) Model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"  # NormalFloat4 format: more accurate for LLMs
)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant_config, device_map="auto")
print(f"🧠 4-bit Quantized Model (QLoRA) Model Memory footprint: {base_model.get_memory_footprint() / 1e9:,.2f} GB")


# ------------------------------------ Load Fine-Tuned LoRA Adapters ----------------------------------
print("\n📦 Loading Fine-Tuned LoRA Adapter...")
fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
print(f"🧠 Fine-Tuned Model Memory Footprint: {fine_tuned_model.get_memory_footprint() / 1e9:.2f} GB")


# ------------------------------------ LoRA Parameter Analysis ----------------------------------
print()
# Estimate LoRA adapter parameter counts for each attention projection
# LoRA introduces 2 matrices (A and B) per target module
lora_q_proj = 4096 * 32 + 4096 * 32
lora_k_proj = 4096 * 32 + 1024 * 32
lora_v_proj = 4096 * 32 + 1024 * 32
lora_o_proj = 4096 * 32 + 4096 * 32

# Total parameters for one transformer block (layer)
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj

# Total layers in LLaMA 8B = 32 transformer blocks
params = lora_layer * 32

# Estimate total adapter size in MB (4 bytes per FP32 parameter)
size = (params * 4) / 1_000_000
print(f"\n📊 LoRA Adapter Params: {params:,}")
print(f"💾 Approx. LoRA Adapter Size: {size:.1f} MB")
