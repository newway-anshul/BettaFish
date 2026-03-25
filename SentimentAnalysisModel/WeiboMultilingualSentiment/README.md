# Multilingual Sentiment Analysis

This module uses a multilingual sentiment analysis model from HuggingFace to perform sentiment analysis, supporting 22 languages.

## Model Information

- **Model Name**: tabularisai/multilingual-sentiment-analysis  
- **Base Model**: distilbert-base-multilingual-cased
- **Supported Languages**: 22 languages, including:
  - 中文 (Chinese)
  - English
  - Español (Spanish)
  - 日本語 (Japanese)
  - 한국어 (Korean)
  - Français (French)
  - Deutsch (German)
  - Русский (Russian)
  - العربية (Arabic)
  - हिन्दी (Hindi)
  - Português (Portuguese)
  - Italiano (Italian)
  - and more...

- **Output Classes**: 5-level sentiment classification
  - Very Negative
  - Negative
  - Neutral
  - Positive
  - Very Positive

## Quick Start

1. Ensure dependencies are installed:
```bash
pip install transformers torch
```

2. Run the prediction program:
```bash
python predict.py
```

3. Enter text in any language for analysis:
```
Please enter text: I love this product!
Prediction: Very Positive (confidence: 0.9456)
```

4. View multilingual examples:
```
Please enter text: demo
```

## Code Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_name = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Predict
texts = [
    "今天心情很好",  # Chinese
    "I love this!",  # English
    "¡Me encanta!"   # Spanish
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    print(f"{text} -> {sentiment_map[prediction]}")
```

## Key Features

- **Multilingual support**: No need to specify language — automatically detects 22 languages
- **5-level fine-grained classification**: More nuanced sentiment analysis than traditional binary classification
- **High accuracy**: Built on the advanced DistilBERT architecture
- **Local caching**: Saved locally after first download for faster subsequent use

## Use Cases

- International social media monitoring
- Multilingual customer feedback analysis
- Global product review sentiment classification
- Cross-language brand sentiment tracking
- Multilingual customer service optimization
- International market research

## Model Storage

- On first run, the model is automatically downloaded to the `model` folder in the current directory
- Subsequent runs load directly from local cache — no re-download needed
- Model size is approximately 135MB; a network connection is required for the first download

## File Description

- `predict.py`: Main prediction program using direct model inference
- `README.md`: Usage instructions

## Notes

- A network connection is required for the first run to download the model
- The model is saved to the current directory for convenient reuse
- GPU acceleration is supported and the available device is detected automatically
- To clean up model files, simply delete the `model` folder
- This model is trained on synthetic data; validation in real-world applications is recommended