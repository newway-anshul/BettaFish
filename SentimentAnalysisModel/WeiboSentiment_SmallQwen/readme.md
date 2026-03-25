# Fine-Tuning Small Qwen3 Models for Sentiment Analysis

<img src="https://github.com/666ghj/Weibo_PublicOpinion_AnalysisSystem/blob/main/static/image/logo_Qweb3.jpg" alt="Weibo Sentiment Analysis Example" width="25%" />

## Project Background

This folder is dedicated to Weibo sentiment analysis based on Alibaba's Qwen3 model family. According to recent evaluations, small Qwen3 models (0.6B, 4B, 8B) perform strongly on relatively straightforward NLP tasks such as topic detection and sentiment analysis, often outperforming classic baseline models like BERT.

Using Qwen3-0.6B with a linear classifier for domain-specific text classification and sequence labeling can outperform BERT, and may even outperform Qwen3 235B few-shot learning for this type of task. Under limited compute budgets, this setup can offer excellent cost-performance.

After some practical investigation, using small Qwen3 variants in this system turned out to be a solid choice.

Even though these are considered "small" in the LLM era, fine-tuning still requires substantial resources for individual developers. Training on a single A100 took four full days.

## Research Question

An interesting comparison is the following:
For Qwen3-Embedding-0.6B vs Qwen3-0.6B, if the former is used with an external classification head and the latter is fine-tuned with LoRA on the same dataset, which performs better and what are their trade-offs?

**In most cases, LoRA fine-tuning on Qwen3-0.6B gives significantly better quality than using Qwen3-Embedding-0.6B with an external classifier head, but inference speed is usually slower than the classifier-head approach.**

For this reason, this module provides both **fine-tuning** and **embedding + classifier head** versions across model sizes.

The following table summarizes the differences:

| Feature / Dimension | Method A: `Qwen3-Embedding-0.6B` + Classifier Head | Method B: `Qwen3-0.6B` + LoRA Fine-Tuning |
| ------------------- | --------------------------------------------------- | ------------------------------------------ |
| **Core Idea** | **Representation Learning** | **Instruction Following** |
| **How the Model Learns** | Freeze the embedding model and train only a tiny classifier head (for example, `nn.Linear`) to map fixed text vectors to sentiment labels. | Freeze most base-model parameters and train LoRA adapters to adjust **internal attention behavior and knowledge representation**, so the model learns to output task-specific answers under instructions. |
| **Performance Ceiling** | **Lower**. The model is bounded by the general semantic representation of `Qwen3-Embedding-0.6B`, and cannot adapt deeply to subtle dataset-specific sentiment patterns. | **Higher**. Fine-tuning adapts the model's language understanding to your task distribution, better capturing sarcasm, internet slang, and nuanced sentiment. |
| **Flexibility** | **Low**. The model is limited to outputting classification labels. | **High**. The model learns a reusable task skill. You can adjust prompts to output "positive/negative/neutral" or even explanatory responses. |
| **Training Resource Cost** | **Very low**. Only a KB-to-MB scale classifier head is trained; can often run on CPU. VRAM usage is minimal. | **Higher**. Although LoRA is parameter-efficient, training still requires GPU and loading the full 0.6B base model plus adapters for backpropagation. |
| **Inference Speed / Cost** | **Very fast / very low**. One forward pass gives embeddings, and classifier-head cost is negligible. Great for large-scale low-latency production. | **Slower / higher cost**. Requires autoregressive generation token by token; even short answers are typically much slower than a single forward pass. |
| **Implementation Complexity** | **Simple**. Classic BERT-style workflow, mature and straightforward. | **Medium**. Requires instruction templates, LoRA configuration, and trainer setup (for example, SFTTrainer), but ecosystem support is mature. |

## Usage Guide

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Activate your PyTorch environment
conda activate your_env_name
```

### Train Models

**Embedding + Classifier Head:**
```bash
python qwen3_embedding_universal.py
# The program will ask you to choose model size (0.6B/4B/8B)
```

**LoRA Fine-Tuning:**
```bash
python qwen3_lora_universal.py
# The program will ask you to choose model size (0.6B/4B/8B)
```

**Command-Line Options:**
```bash
# Specify model directly
python qwen3_embedding_universal.py --model_size 0.6B
python qwen3_lora_universal.py --model_size 4B

# Custom parameters
python qwen3_embedding_universal.py --model_size 8B --epochs 10 --batch_size 16
```

### Run Prediction

**Interactive Mode:**
```bash
python predict_universal.py
# The program will prompt you to choose model method and size
```

**Command-Line Prediction:**
```bash
# Predict with a specified model
python predict_universal.py --model_type embedding --model_size 0.6B --text "The weather is great today"

# Load all available models
python predict_universal.py --load_all --text "This movie is amazing"
```

### Notes

1. **VRAM requirements**:
   - 0.6B: at least 4GB
   - 4B: at least 16GB
   - 8B: at least 32GB

2. **Data format**: each line should be `text content\tlabel`, where label is 0 (negative) or 1 (positive)

3. **Model selection**: for first-time testing, start with the 0.6B model

4. **Training time**: LoRA fine-tuning is slower than the embedding approach; GPU acceleration is recommended