# Weibo Sentiment Analysis - Traditional Machine Learning Methods

## Project Overview

This project uses five traditional machine learning methods for binary sentiment classification (positive/negative) on Chinese Weibo text:

- **Naive Bayes**: Probabilistic classification based on bag-of-words features
- **SVM**: Support Vector Machine based on TF-IDF features
- **XGBoost**: Gradient-boosted decision trees
- **LSTM**: Recurrent neural network with Word2Vec embeddings
- **BERT + Classification Head**: Pretrained language model with a task-specific classifier

## Model Performance

Performance on the Weibo sentiment dataset (10,000 training samples, 500 test samples):

| Model | Accuracy | AUC | Notes |
|------|--------|-----|------|
| Naive Bayes | 85.6% | - | Fast and lightweight |
| SVM | 85.6% | - | Strong generalization |
| XGBoost | 86.0% | 90.4% | Stable performance, supports feature importance |
| LSTM | 87.0% | 93.1% | Captures sequence and context information |
| BERT + Head | 87.0% | 92.9% | Strong semantic understanding |

## Environment Setup

```bash
pip install -r requirements.txt
```

Data directory structure:
```
data/
├── weibo2018/
│   ├── train.txt
│   └── test.txt
└── stopwords.txt
```

## Train Models (can run directly without extra args)

### Naive Bayes
```bash
python bayes_train.py
```

### SVM
```bash
python svm_train.py --kernel rbf --C 1.0
```

### XGBoost
```bash
python xgboost_train.py --max_depth 6 --eta 0.3 --num_boost_round 200
```

### LSTM
```bash
python lstm_train.py --epochs 5 --batch_size 100 --hidden_size 64
```

### BERT
```bash
python bert_train.py --epochs 10 --batch_size 100 --learning_rate 1e-3
```

Note: The BERT script will automatically download the pretrained model (bert-base-chinese).

## Prediction

### Interactive Prediction (Recommended)
```bash
python predict.py
```

### Command-Line Prediction
```bash
# Single-model prediction
python predict.py --model_type bert --text "The weather is great today, I feel awesome."

# Multi-model ensemble prediction
python predict.py --ensemble --text "This movie was so boring."
```

## File Structure

```
WeiboSentiment_MachineLearning/
├── bayes_train.py           # Naive Bayes training
├── svm_train.py             # SVM training
├── xgboost_train.py         # XGBoost training
├── lstm_train.py            # LSTM training
├── bert_train.py            # BERT training
├── predict.py               # Unified prediction program
├── base_model.py            # Base model class
├── utils.py                 # Utility functions
├── requirements.txt         # Dependencies
├── model/                   # Model output directory
└── data/                    # Data directory
```

## Notes

1. **BERT model** downloads the pretrained files on first run (about 400MB).
2. **LSTM model** takes longer to train; GPU is recommended.
3. **Model outputs** are saved in `model/`; ensure enough disk space.
4. **Memory usage**: BERT > LSTM > XGBoost > SVM > Naive Bayes.
