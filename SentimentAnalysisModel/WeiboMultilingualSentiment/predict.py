import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

def preprocess_text(text):
    """Simple text preprocessing suitable for multilingual text"""
    return text

def main():
    print("Loading multilingual sentiment analysis model...")
    
    # Use multilingual sentiment analysis model
    model_name = "tabularisai/multilingual-sentiment-analysis"
    local_model_path = "./model"
    
    try:
        # Check if model already exists locally
        import os
        if os.path.exists(local_model_path):
            print("Loading model from local cache...")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            print("First time use, downloading model to local cache...")
            # Download and save locally
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Save to local cache
            tokenizer.save_pretrained(local_model_path)
            model.save_pretrained(local_model_path)
            print(f"Model saved to: {local_model_path}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"Model loaded successfully! Using device: {device}")
        
        # Sentiment label mapping (5-level classification)
        sentiment_map = {
            0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"
        }
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("Please check your network connection")
        return
    
    print("\n============= Multilingual Sentiment Analysis =============")
    print("Supported languages: Chinese, English, Spanish, Arabic, Japanese, Korean, and 22 more")
    print("Sentiment levels: Very Negative, Negative, Neutral, Positive, Very Positive")
    print("Enter text for analysis (enter 'q' to quit):")
    print("Enter 'demo' to see multilingual examples")
    
    while True:
        text = input("\nPlease enter text: ")
        if text.lower() == 'q':
            break
        
        if text.lower() == 'demo':
            show_multilingual_demo(tokenizer, model, device, sentiment_map)
            continue
        
        if not text.strip():
            print("Input cannot be empty, please try again")
            continue
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Tokenize and encode
            inputs = tokenizer(
                processed_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Run prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            # Output results
            confidence = probabilities[0][prediction].item()
            label = sentiment_map[prediction]
            
            print(f"Prediction: {label} (confidence: {confidence:.4f})")
            
            # Show probability for all classes
            print("Detailed probability distribution:")
            for i, (label_name, prob) in enumerate(zip(sentiment_map.values(), probabilities[0])):
                print(f"  {label_name}: {prob:.4f}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

def show_multilingual_demo(tokenizer, model, device, sentiment_map):
    """Show multilingual sentiment analysis examples"""
    print("\n=== Multilingual Sentiment Analysis Examples ===")
    
    demo_texts = [
        # Chinese
        ("今天天气真好，心情特别棒！", "Chinese"),
        ("这家餐厅的菜味道非常棒！", "Chinese"),
        ("服务态度太差了，很失望", "Chinese"),
        
        # English
        ("I absolutely love this product!", "English"),
        ("The customer service was disappointing.", "English"),
        ("The weather is fine, nothing special.", "English"),
        
        # Japanese
        ("このレストランの料理は本当に美味しいです！", "Japanese"),
        ("このホテルのサービスはがっかりしました。", "Japanese"),
        
        # Korean
        ("이 가게의 케이크는 정말 맛있어요！", "Korean"),
        ("서비스가 너무 별로였어요。", "Korean"),
        
        # Spanish
        ("¡Me encanta cómo quedó la decoración!", "Spanish"),
        ("El servicio fue terrible y muy lento.", "Spanish"),
    ]
    
    for text, language in demo_texts:
        try:
            inputs = tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            confidence = probabilities[0][prediction].item()
            label = sentiment_map[prediction]
            
            print(f"\n{language}: {text}")
            print(f"Result: {label} (confidence: {confidence:.4f})")
            
        except Exception as e:
            print(f"Error processing {text}: {e}")
    
    print("\n=== End of Examples ===")
    
    '''
    Loading multilingual sentiment analysis model...
Loading model from local cache...
Model loaded successfully! Using device: cuda

============= Multilingual Sentiment Analysis =============
Supported languages: Chinese, English, Spanish, Arabic, Japanese, Korean, and 22 more
Sentiment levels: Very Negative, Negative, Neutral, Positive, Very Positive
Enter text for analysis (enter 'q' to quit):
Enter 'demo' to see multilingual examples

Please enter text: 我喜欢你
C:\Users\67093\.conda\envs\pytorch_python11\Lib\site-packages\transformers\models\distilbert\modeling_distilbert.py:401: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:263.)
  attn_output = torch.nn.functional.scaled_dot_product_attention(
Prediction: Positive (confidence: 0.5204)
Detailed probability distribution:
  Very Negative: 0.0329
  Negative: 0.0263
  Neutral: 0.1987
  Positive: 0.5204
  Very Positive: 0.2216

Please enter text:
    '''

if __name__ == "__main__":
    main()