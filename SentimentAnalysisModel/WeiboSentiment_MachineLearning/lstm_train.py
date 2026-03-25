# -*- coding: utf-8 -*-
"""
LSTM sentiment analysis training script.
"""
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from gensim import models
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from typing import List, Tuple, Dict, Any
import numpy as np

from base_model import BaseModel


class LSTMDataset(Dataset):
    """LSTM dataset."""
    
    def __init__(self, data: List[Tuple[str, int]], word2vec_model):
        self.data = []
        self.label = []
        
        for text, label in data:
            vectors = []
            for word in text.split(" "):
                if word in word2vec_model.wv.key_to_index:
                    vectors.append(word2vec_model.wv[word])
            
            if len(vectors) > 0:  # Ensure there are valid word vectors.
                vectors = torch.Tensor(vectors)
                self.data.append(vectors)
                self.label.append(label)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.label)


def collate_fn(data):
    """Batch collation function."""
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    data = pad_sequence(x, batch_first=True, padding_value=0)
    return data, torch.tensor(y, dtype=torch.float32), data_length


class LSTMNet(nn.Module):
    """LSTM network architecture."""
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Bidirectional LSTM.
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, lengths):
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        packed_input = pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))
        
        # Bidirectional LSTM: concatenate the final hidden states.
        lstm_out = torch.cat([h_n[-2], h_n[-1]], 1)
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out


class LSTMModel(BaseModel):
    """LSTM sentiment analysis model."""
    
    def __init__(self):
        super().__init__("LSTM")
        self.word2vec_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _train_word2vec(self, train_data: List[Tuple[str, int]], **kwargs):
        """Train Word2Vec embeddings."""
        print("Training Word2Vec embeddings...")
        
        # Prepare Word2Vec input data.
        wv_input = [text.split(" ") for text, _ in train_data]
        
        vector_size = kwargs.get('vector_size', 64)
        min_count = kwargs.get('min_count', 1)
        epochs = kwargs.get('epochs', 1000)
        
        # Train Word2Vec.
        self.word2vec_model = models.Word2Vec(
            wv_input,
            vector_size=vector_size,
            min_count=min_count,
            epochs=epochs
        )
        
        print(f"Word2Vec training completed, vector size: {vector_size}")
        
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the LSTM model."""
        print(f"Starting training for {self.model_name} model...")
        
        # Train Word2Vec.
        self._train_word2vec(train_data, **kwargs)
        
        # Hyperparameters.
        learning_rate = kwargs.get('learning_rate', 5e-4)
        num_epochs = kwargs.get('num_epochs', 5)
        batch_size = kwargs.get('batch_size', 100)
        embed_size = kwargs.get('embed_size', 64)
        hidden_size = kwargs.get('hidden_size', 64)
        num_layers = kwargs.get('num_layers', 2)
        
        print(f"LSTM hyperparameters: lr={learning_rate}, epochs={num_epochs}, "
              f"batch_size={batch_size}, hidden_size={hidden_size}")
        
        # Create dataset.
        train_dataset = LSTMDataset(train_data, self.word2vec_model)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 collate_fn=collate_fn, shuffle=True)
        
        # Create model.
        self.model = LSTMNet(embed_size, hidden_size, num_layers).to(self.device)
        
        # Loss function and optimizer.
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop.
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i, (x, labels, lengths) in enumerate(train_loader):
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass.
                outputs = self.model(x, lengths)
                logits = outputs.view(-1)
                loss = criterion(logits, labels)
                
                # Backpropagation.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (i + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {avg_loss:.4f}")
            
            # Save model after each epoch.
            if kwargs.get('save_each_epoch', False):
                epoch_model_path = f"./model/lstm_epoch_{epoch+1}.pth"
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), epoch_model_path)
                print(f"Saved model: {epoch_model_path}")
        
        self.is_trained = True
        print(f"{self.model_name} model training completed.")
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels for texts."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained. Please call train() first.")
        
        # Create dataset.
        test_data = [(text, 0) for text in texts]  # Labels are not used here.
        test_dataset = LSTMDataset(test_data, self.word2vec_model)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for x, _, lengths in test_loader:
                x = x.to(self.device)
                outputs = self.model(x, lengths)
                outputs = outputs.view(-1)
                
                # Convert to class labels.
                preds = (outputs > 0.5).cpu().numpy()
                predictions.extend(preds.astype(int).tolist())
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment for a single text."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained. Please call train() first.")
        
        # Convert input into word vectors.
        vectors = []
        for word in text.split(" "):
            if word in self.word2vec_model.wv.key_to_index:
                vectors.append(self.word2vec_model.wv[word])
        
        if len(vectors) == 0:
            return 0, 0.5  # Return a default value when no valid vectors exist.
        
        # Convert to tensor.
        x = torch.Tensor(vectors).unsqueeze(0).to(self.device)  # Add batch dimension.
        lengths = [len(vectors)]
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(x, lengths)
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
        
        # Save model state and Word2Vec.
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'word2vec_model': self.word2vec_model,
            'model_config': {
                'embed_size': 64,
                'hidden_size': 64,
                'num_layers': 2
            },
            'device': str(self.device)
        }
        
        torch.save(model_data, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        model_data = torch.load(model_path, map_location=self.device)
        
        # Load Word2Vec.
        self.word2vec_model = model_data['word2vec_model']
        
        # Rebuild LSTM network.
        config = model_data['model_config']
        self.model = LSTMNet(
            config['embed_size'],
            config['hidden_size'],
            config['num_layers']
        ).to(self.device)
        
        # Load model weights.
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.is_trained = True
        print(f"Loaded model: {model_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='LSTM sentiment analysis training')
    parser.add_argument('--train_path', type=str, default='./data/weibo2018/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./data/weibo2018/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, default='./model/lstm_model.pth',
                        help='Model save path')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM hidden size')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Evaluate an existing model only, without training')
    
    args = parser.parse_args()
    
    # Create model.
    model = LSTMModel()
    
    if args.eval_only:
        # Evaluation-only mode.
        print("Evaluation mode: load an existing model for evaluation")
        model.load_model(args.model_path)
        
        # Load test data.
        _, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # Evaluate model.
        model.evaluate(test_data)
    else:
        # Training mode.
        # Load data.
        train_data, test_data = BaseModel.load_data(args.train_path, args.test_path)
        
        # Train model.
        model.train(
            train_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
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