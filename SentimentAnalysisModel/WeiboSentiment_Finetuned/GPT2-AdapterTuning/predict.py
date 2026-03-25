import torch
from transformers import BertTokenizer
from train import GPT2ClassifierWithAdapter
import re

def preprocess_text(text):
    """Simple text preprocessing"""
    return text

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use local model path instead of online model names
    local_model_path = './models/gpt2-chinese'
    model_path = 'best_weibo_sentiment_model.pth'
    
    print(f"Loading model: {model_path}")
    # Load tokenizer from local path
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = '[PAD]'
    
    # Load model using local model path
    model = GPT2ClassifierWithAdapter(local_model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("\n============= Weibo Sentiment Analysis =============")
    print("Enter Weibo content for analysis (input 'q' to exit):")
    
    while True:
        text = input("\nPlease enter Weibo content: ")
        if text.lower() == 'q':
            break
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Encode text
        encoding = tokenizer(
            processed_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        
        # Output results
        confidence = probabilities[0][prediction].item()
        label = "Positive sentiment" if prediction == 1 else "Negative sentiment"
        
        print(f"Prediction result: {label} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main() 