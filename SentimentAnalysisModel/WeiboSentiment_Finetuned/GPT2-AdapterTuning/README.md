# Weibo Sentiment Recognition Model - GPT2-Adapter Fine-tuning

## Project Description
This is a binary sentiment classification model for Weibo based on GPT2, using Adapter fine-tuning technique. Through Adapter fine-tuning, only a small number of parameters need to be trained to adapt the model to the sentiment analysis task, greatly reducing computational resource requirements and model size.

## Dataset
Uses the Weibo sentiment dataset (weibo_senti_100k), which contains approximately 100,000 Weibo content with sentiment annotations, with about 50,000 positive and 50,000 negative comments. Dataset labels:
- Label 0: Negative sentiment
- Label 1: Positive sentiment

## File Structure
```
GPT2-Adpter-tuning/
├── adapter.py              # Adapter layer implementation
├── gpt2_adapter.py         # Adapter implementation for GPT2 model
├── train.py                # Training script
├── predict.py              # Simplified prediction script (interactive use)
├── models/                 # Directory for storing local pre-trained models
│   └── gpt2-chinese/       # Chinese GPT2 model and configuration
├── dataset/                # Dataset directory
│   └── weibo_senti_100k.csv  # Weibo sentiment dataset
└── best_weibo_sentiment_model.pth  # Trained best model
```

## Technical Features

1. **Parameter-Efficient Fine-tuning**: Compared to full parameter fine-tuning, only about 3% of parameters are trained
2. **Model Performance Preservation**: Maintains good classification performance while training only a small number of parameters
3. **Suitable for Resource-Constrained Environments**: Small model size and fast inference speed

## Environment Dependencies
- Python 3.6+
- PyTorch
- Transformers
- Pandas
- NumPy
- Scikit-learn
- Tqdm

## Usage

### Train Model
```bash
python train.py
```
The training process will automatically:
- Download and save the Chinese GPT2 pre-trained model locally
- Load the Weibo sentiment dataset
- Train the model and save the best model

### Sentiment Analysis Prediction
```bash
python predict.py
```
After running, you will enter interactive mode:
- Enter the Weibo text to be analyzed in the console
- The system will return sentiment analysis results (positive/negative) and confidence score
- Enter 'q' to exit the program

## Model Structure
- Base Model: `uer/gpt2-chinese-cluecorpussmall` Chinese pre-trained model
- Local Model Path: `./models/gpt2-chinese/`
- Fine-tune by adding Adapter layers after each GPT2Block
- Freeze original GPT2 parameters, only train classifier and Adapter layer parameters

## Adapter Technology
Adapter is a parameter-efficient fine-tuning technique that enables adaptation to downstream tasks with only a limited number of parameters by inserting small bottleneck layers into Transformer layers. Main features:

1. **Parameter Efficiency**: Compared to full parameter fine-tuning, Adapter only needs to train a small fraction of parameters
2. **Forgetting Prevention**: Keeps the original pre-trained model parameters unchanged, avoiding catastrophic forgetting
3. **Multi-task Adaptation**: Different Adapters can be trained for different tasks, sharing the same base model

In this project, we added an Adapter layer after each GPT2Block. The hidden layer size of the Adapter is 64, much smaller than the original model's hidden layer size (typically 768 or 1024).

## Usage Example
```
Using device: cuda
Loading model: best_weibo_sentiment_model.pth

============= Weibo Sentiment Analysis =============
Enter Weibo content for analysis (input 'q' to exit):

Please enter Weibo content: This movie is so good, I really love it!
Prediction result: Positive sentiment (Confidence: 0.9876)

Please enter Weibo content: Poor service, expensive prices, not recommended at all
Prediction result: Negative sentiment (Confidence: 0.9742)
```

## Notes
- The prediction script uses local model paths and does not need to download models online
- Ensure the `models/gpt2-chinese/` directory contains model files saved from the training process
- When running train.py for the first time, it will automatically download and save the model, please ensure a stable network connection 