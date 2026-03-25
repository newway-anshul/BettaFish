# -*- coding: utf-8 -*-
"""
Universal Qwen3-Embedding training script
Supports three model sizes: 0.6B, 4B, and 8B
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class SentimentDataset(Dataset):
    """Sentiment Analysis Dataset"""
    
    def __init__(self, data: List[Tuple[str, int]], tokenizer, max_length=512):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


class SentimentClassifier(nn.Module):
    """Sentiment Classifier"""
    
    def __init__(self, embedding_model, embedding_dim, hidden_dim=256):
        super(SentimentClassifier, self).__init__()
        self.embedding_model = embedding_model
        
        # Freeze embedding model parameters
        for param in self.embedding_model.parameters():
            param.requires_grad = False
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classification head
        logits = self.classifier(embeddings)
        return logits.squeeze()


class Qwen3EmbeddingUniversal(BaseQwenModel):
    """Universal Qwen3-Embedding model"""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"Unsupported model size: {model_size}")
            
        super().__init__(f"Qwen3-Embedding-{model_size}")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["embedding_model"]
        self.embedding_dim = self.config["embedding_dim"]
        
        self.tokenizer = None
        self.embedding_model = None
        self.classifier_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_embedding_model(self):
        """Load the Qwen3 Embedding model"""
        print(f"Loading {self.model_size} model: {self.model_name_hf}")
        
        # Step 1: Check models directory in current folder
        local_model_dir = f"./models/qwen3-embedding-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"Found local model, loading from local path: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.embedding_model = AutoModel.from_pretrained(local_model_dir).to(self.device)
                print(f"Successfully loaded {self.model_size} model from local path")
                return
                
            except Exception as e:
                print(f"Local model loading failed: {e}")
        
        # Step 2: Check HuggingFace cache
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"Checking HuggingFace cache: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.embedding_model = AutoModel.from_pretrained(self.model_name_hf).to(self.device)
            print(f"Successfully loaded {self.model_size} model from HuggingFace cache")
            
            # Save to local models directory
            print(f"Saving model locally to: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.embedding_model.save_pretrained(local_model_dir)
            print(f"Model saved to: {local_model_dir}")
            
        except Exception as e:
            print(f"Loading from HuggingFace cache failed: {e}")
            
            # Step 3: Download from HuggingFace
            try:
                print(f"Downloading {self.model_size} model from HuggingFace...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.embedding_model = AutoModel.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                ).to(self.device)
                
                # Save to local models directory
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.embedding_model.save_pretrained(local_model_dir)
                print(f"{self.model_size} model downloaded and saved to: {local_model_dir}")
                
            except Exception as e2:
                print(f"Downloading from HuggingFace also failed: {e2}")
                raise RuntimeError(f"Unable to load {self.model_size} model, all methods failed")
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the model"""
        print(f"Starting training for Qwen3-Embedding-{self.model_size} model...")
        
        # Load embedding model
        self._load_embedding_model()
        
        # Hyperparameters (recommended values or user-provided overrides)
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'])
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'])
        num_epochs = kwargs.get('num_epochs', 5)
        max_length = kwargs.get('max_length', 512)
        
        print(f"Hyperparameters: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Create dataset
        train_dataset = SentimentDataset(train_data, self.tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create classifier
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            self.embedding_dim
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.classifier_model.classifier.parameters(), lr=learning_rate)
        
        # Training loop
        self.classifier_model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.classifier_model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': total_loss / num_batches})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"Qwen3-Embedding-{self.model_size} model training completed!")
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment for multiple texts"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        predictions = []
        batch_size = 32
        
        self.classifier_model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                outputs = self.classifier_model(input_ids, attention_mask)
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment for a single text"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        self.classifier_model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            output = self.classifier_model(input_ids, attention_mask)
            prob = output.item()
            prediction = int(prob > 0.5)
            confidence = prob if prediction == 1 else 1 - prob
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """Save model"""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        if model_path is None:
            model_path = MODEL_PATHS["embedding"][self.model_size]
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'classifier_state_dict': self.classifier_model.classifier.state_dict(),
            'model_size': self.model_size,
            'model_name_hf': self.model_name_hf,
            'embedding_dim': self.embedding_dim,
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Load model data
        model_data = torch.load(model_path, map_location=self.device)
        
        # Verify model size matches
        if model_data['model_size'] != self.model_size:
            raise ValueError(f"Model size mismatch: expected {model_data['model_size']}, got {model_data['model_size']}")
        
        # Load embedding model
        self._load_embedding_model()
        
        # Rebuild classifier
        self.classifier_model = SentimentClassifier(
            self.embedding_model, 
            model_data['embedding_dim']
        ).to(self.device)
        self.classifier_model.classifier.load_state_dict(model_data['classifier_state_dict'])
        
        self.model = self.classifier_model
        self.is_trained = True
        print(f"Loaded Qwen3-Embedding-{self.model_size} model: {model_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Universal Qwen3-Embedding training script')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='Model size')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, help='Model save path (optional)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size (optional, uses recommended value)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (optional, uses recommended value)')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation-only mode')
    
    args = parser.parse_args()
    
    # If model size is not specified, ask user interactively
    if not args.model_size:
        print("Qwen3-Embedding model training")
        print("="*40)
        print("Available model sizes:")
        print("  1. 0.6B - Lightweight, fast training, about 4GB VRAM")
        print("  2. 4B  - Medium scale, balanced performance, about 16GB VRAM") 
        print("  3. 8B  - Large scale, best performance, about 32GB VRAM")
        
        while True:
            choice = input("\nPlease choose model size (1/2/3): ").strip()
            if choice == '1':
                args.model_size = '0.6B'
                break
            elif choice == '2':
                args.model_size = '4B'
                break
            elif choice == '3':
                args.model_size = '8B'
                break
            else:
                print("Invalid choice, please enter 1, 2, or 3")
        
        print(f"Selected: Qwen3-Embedding-{args.model_size}")
        print()
    
    # Ensure models directory exists
    os.makedirs('./models', exist_ok=True)
    
    # Create model
    model = Qwen3EmbeddingUniversal(args.model_size)
    
    # Determine model save path
    model_path = args.model_path or MODEL_PATHS["embedding"][args.model_size]
    
    if args.eval_only:
        # Evaluation-only mode
        print(f"Evaluation mode: loading Qwen3-Embedding-{args.model_size} model")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        model.evaluate(test_data)
    else:
        # Training mode
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # Prepare training parameters
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        
        # Train model
        model.train(train_data, **train_kwargs)
        
        # Evaluate model
        model.evaluate(test_data)
        
        # Save model
        model.save_model(model_path)
        
        # Example predictions
        print(f"\nQwen3-Embedding-{args.model_size} example predictions:")
        test_texts = [
            "The weather is great today, I feel awesome.",
            "This movie is so boring, a complete waste of time.", 
            "Hahaha, this is so funny."
        ]
        
        for text in test_texts:
            pred, conf = model.predict_single(text)
            sentiment = "positive" if pred == 1 else "negative"
            print(f"Text: {text}")
            print(f"Prediction: {sentiment} (confidence: {conf:.4f})")
            print()


if __name__ == "__main__":
    main()