# -*- coding: utf-8 -*-
"""
Base model class providing a unified interface for all sentiment analysis models.
"""
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils import load_corpus


class BaseModel(ABC):
    """Base class for sentiment analysis models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
    @abstractmethod
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment labels for input texts."""
        pass
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment for a single text.
        
        Args:
            text: Input text.
            
        Returns:
            (predicted_label, confidence)
        """
        predictions = self.predict([text])
        return predictions[0], 0.0  # Default confidence value.
    
    def evaluate(self, test_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained. Please call train() first.")
            
        texts = [item[0] for item in test_data]
        labels = [item[1] for item in test_data]
        
        predictions = self.predict(texts)
        
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        print(f"\n{self.model_name} model evaluation results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        print("\nDetailed report:")
        print(classification_report(labels, predictions))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(labels, predictions)
        }
    
    def save_model(self, model_path: str = None) -> None:
        """Save the model to a file."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} is not trained and cannot be saved.")
            
        if model_path is None:
            model_path = f"model/{self.model_name}_model.pkl"
            
        # Create output directory.
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model data.
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load the model from a file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
            
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.vectorizer = model_data.get('vectorizer')
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        
        print(f"Loaded model: {model_path}")
    
    @staticmethod
    def load_data(train_path: str, test_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """Load training and test data."""
        print("Loading training data...")
        train_data = load_corpus(train_path)
        print(f"Training samples: {len(train_data)}")
        
        print("Loading test data...")
        test_data = load_corpus(test_path)
        print(f"Test samples: {len(test_data)}")
        
        return train_data, test_data