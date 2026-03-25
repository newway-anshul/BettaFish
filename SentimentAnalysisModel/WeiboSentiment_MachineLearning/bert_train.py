# -*- coding: utf-8 -*-
"""
BERT sentiment analysis training script.
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from typing import List, Tuple
import warnings
import requests
from pathlib import Path

from base_model import BaseModel
from utils import load_corpus_bert

# Ignore transformers warnings.
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BertDataset(Dataset):
    """BERT dataset."""
    
    def __init__(self, data: List[Tuple[str, int]]):
        self.data = [item[0] for item in data]
        self.labels = [item[1] for item in data]
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)


class BertClassifier(nn.Module):
    """BERT classifier network."""
    
    def __init__(self, input_size):
        super(BertClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class BertModel_Custom(BaseModel):
    """BERT sentiment analysis model."""
    
    def __init__(self, model_path: str = "./model/chinese_wwm_pytorch"):
        super().__init__("BERT")
        self.model_path = model_path
        self.tokenizer = None
        self.bert = None
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _download_bert_model(self):
        """Automatically download the BERT pretrained model."""
        print(f"BERT model not found. Downloading Chinese BERT pretrained model...")
        print("Source: bert-base-chinese (Hugging Face)")
        
        try:
            # Create model directory.
            os.makedirs(self.model_path, exist_ok=True)
            
            # Use the Chinese BERT model from Hugging Face.
            model_name = "bert-base-chinese"
            print(f"Downloading {model_name} from Hugging Face...")
            
            # Download tokenizer.
            print("Downloading tokenizer...")
            tokenizer = BertTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(self.model_path)
            
            # Download model.
            print("Downloading BERT model...")
            bert_model = BertModel.from_pretrained(model_name)
            bert_model.save_pretrained(self.model_path)
            
            print(f"BERT model download complete, saved at: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to download BERT model: {e}")
            print("\nYou can also download BERT manually:")
            print("1. Visit https://huggingface.co/bert-base-chinese")
            print("2. Or use HIT Chinese BERT: https://github.com/ymcui/Chinese-BERT-wwm")
            print(f"3. Extract model files to: {self.model_path}")
            return False
    
    def _load_bert(self):
        """Load BERT model and tokenizer."""
        print(f"Loading BERT model: {self.model_path}")
        
        # If model does not exist, try auto-download.
        if not os.path.exists(self.model_path) or not any(os.scandir(self.model_path)):
            print("BERT model does not exist, trying auto-download...")
            if not self._download_bert_model():
                raise FileNotFoundError(f"BERT model download failed. Please download manually to: {self.model_path}")
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.bert = BertModel.from_pretrained(self.model_path).to(self.device)
            
            # Freeze BERT parameters.
            for param in self.bert.parameters():
                param.requires_grad = False
                
            print("BERT model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            print("Trying online model...")
            
            # If local loading fails, try online model directly.
            try:
                model_name = "bert-base-chinese"
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
                self.bert = BertModel.from_pretrained(model_name).to(self.device)
                
                # Freeze BERT parameters.
                for param in self.bert.parameters():
                    param.requires_grad = False
                    
                print("Online BERT model loaded successfully")
                
            except Exception as e2:
                print(f"Online model also failed: {e2}")
                raise FileNotFoundError(f"Unable to load BERT model. Please check network or download manually to: {self.model_path}")
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the BERT model."""
        print(f"Starting training for {self.model_name} model...")
        
        # Load BERT.
        self._load_bert()
        
        # Hyperparameters.
        learning_rate = kwargs.get('learning_rate', 1e-3)
        num_epochs = kwargs.get('num_epochs', 10)
        batch_size = kwargs.get('batch_size', 100)
        input_size = kwargs.get('input_size', 768)
        decay_rate = kwargs.get('decay_rate', 0.9)
        
        print(f"BERT超参数: lr={learning_rate}, epochs={num_epochs}, "
              f"batch_size={batch_size}, input_size={input_size}")
        
        # Create dataset.
        train_dataset = BertDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create classifier.
        self.classifier = BertClassifier(input_size).to(self.device)
        
        # Loss function and optimizer.
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
        
        # Training loop.
        self.bert.eval()  # Keep BERT in evaluation mode.
        self.classifier.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i, (words, labels) in enumerate(train_loader):
                # Tokenize and encode.
                tokens = self.tokenizer(words, padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                labels = torch.tensor(labels, dtype=torch.float32).to(self.device)
                
                # Get BERT output with frozen parameters.
                with torch.no_grad():
                    bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                    bert_output = bert_outputs[0][:, 0]  # [CLS] token output.
                
                # Classifier forward pass.
                optimizer.zero_grad()
                outputs = self.classifier(bert_output)
                logits = outputs.view(-1)
                loss = criterion(logits, labels)
                
                # Backpropagation.
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
                    total_loss = 0
                    num_batches = 0
            
            # Learning rate decay.
            scheduler.step()
            
            # Save model after each epoch.
            if kwargs.get('save_each_epoch', False):
                epoch_model_path = f"./model/bert_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(self.classifier.state_dict(), epoch_model_path)
                print(f"Saved model: {epoch_model_path}")
        
        self.is_trained = True
        print(f"{self.model_name} model training completed.")
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels for texts."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained. Please call train() first.")
        
        predictions = []
        batch_size = 32
        
        self.bert.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and encode.
                tokens = self.tokenizer(batch_texts, padding=True, truncation=True,
                                      max_length=512, return_tensors='pt')
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                
                # Get BERT output.
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                bert_output = bert_outputs[0][:, 0]
                
                # Classifier prediction.
                outputs = self.classifier(bert_output)
                outputs = outputs.view(-1)
                
                # Convert to class labels.
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment for a single text."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained. Please call train() first.")
        
        self.bert.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            # Tokenize and encode.
            tokens = self.tokenizer([text], padding=True, truncation=True,
                                  max_length=512, return_tensors='pt')
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            
            # Get BERT output.
            bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
            bert_output = bert_outputs[0][:, 0]
            
            # Classifier prediction.
            output = self.classifier(bert_output)
            prob = output.item()
            
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """Save model."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained and cannot be saved.")
        
        if model_path is None:
            model_path = f"./model/{self.model_name.lower()}_model.pth"
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save classifier and related metadata.
        model_data = {
            'classifier_state_dict': self.classifier.state_dict(),
            'model_path': self.model_path,
            'input_size': 768,
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # Set BERT model path.
        self.model_path = model_data['model_path']
        
        # Load BERT.
        self._load_bert()
        
        # Rebuild classifier.
        input_size = model_data['input_size']
        self.classifier = BertClassifier(input_size).to(self.device)
        
        # Load classifier weights.
        self.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        self.is_trained = True
        print(f"Loaded model: {model_path}")
    
    @staticmethod
    def load_data(train_path: str, test_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Load data in BERT format."""
        print("Loading training data...")
        train_data = load_corpus_bert(train_path)
        print(f"Training samples: {len(train_data)}")
        
        print("Loading test data...")
        test_data = load_corpus_bert(test_path)
        print(f"Test samples: {len(test_data)}")
        
        return train_data, test_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='BERT sentiment analysis training')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, default='./model/bert_model.pth',
                        help='Model save path')
    parser.add_argument('--bert_path', type=str, default='./model/chinese_wwm_pytorch',
                        help='BERT pretrained model path')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluate an existing model only, without training')
    
    args = parser.parse_args()
    
    # Create model.
    model = BertModel_Custom(args.bert_path)
    
    if args.eval_only:
        # Evaluation-only mode.
        print("Evaluation mode: load an existing model for evaluation")
        model.load_model(args.model_path)
        
        # Load test data.
        _, test_data = model.load_data(args.train_path, args.test_path)
        
        # Evaluate model.
        model.evaluate(test_data)
    else:
        # Training mode.
        # Load data.
        train_data, test_data = model.load_data(args.train_path, args.test_path)
        
        # Train model.
        model.train(
            train_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Evaluate model.
        model.evaluate(test_data)
        
        # Save model.
        model.save_model(args.model_path)
        
        # Sample predictions.
        print("\nSample predictions:")
        test_texts = [
            "The weather is great today, I feel awesome.",
            "This movie was so boring, what a waste of time.",
            "Hahaha, this is really fun."
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "Positive" if pred == 1 else "Negative"
            print(f"Text: {text}")
            print(f"Prediction: {sentiment} (confidence: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()