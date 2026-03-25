# -*- coding: utf-8 -*-
"""
Universal Qwen3-LoRA training script
Supports three model sizes: 0.6B, 4B, and 8B
"""
import argparse
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from typing import List, Tuple
import warnings
from tqdm import tqdm

from base_model import BaseQwenModel
from models_config import QWEN3_MODELS, MODEL_PATHS

warnings.filterwarnings("ignore")


class Qwen3LoRAUniversal(BaseQwenModel):
    """Universal Qwen3-LoRA model."""
    
    def __init__(self, model_size: str = "0.6B"):
        if model_size not in QWEN3_MODELS:
            raise ValueError(f"Unsupported model size: {model_size}")
            
        super().__init__(f"Qwen3-{model_size}-LoRA")
        self.model_size = model_size
        self.config = QWEN3_MODELS[model_size]
        self.model_name_hf = self.config["base_model"]
        
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_base_model(self):
        """Load the Qwen3 base model."""
        print(f"Loading {self.model_size} base model: {self.model_name_hf}")
        
        # Step 1: Check the local models directory in the current folder.
        local_model_dir = f"./models/qwen3-{self.model_size.lower()}"
        if os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json")):
            try:
                print(f"Found local model, loading from local path: {local_model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    local_model_dir,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Set pad token.
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                print(f"Successfully loaded {self.model_size} base model from local path")
                return
                
            except Exception as e:
                print(f"Local model loading failed: {e}")
        
        # Step 2: Check Hugging Face cache.
        try:
            from transformers.utils import default_cache_path
            cache_path = default_cache_path
            print(f"Checking Hugging Face cache: {cache_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_hf,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Set pad token.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            print(f"Successfully loaded {self.model_size} base model from Hugging Face cache")
            
            # Save to local models directory.
            print(f"Saving model locally to: {local_model_dir}")
            os.makedirs(local_model_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_model_dir)
            self.base_model.save_pretrained(local_model_dir)
            print(f"Model saved to: {local_model_dir}")
            
        except Exception as e:
            print(f"Loading from Hugging Face cache failed: {e}")
            
            # Step 3: Download from Hugging Face.
            try:
                print(f"Downloading {self.model_size} model from Hugging Face...")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_hf,
                    force_download=True
                )
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_hf,
                    force_download=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                # Save to local models directory.
                os.makedirs(local_model_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_dir)
                self.base_model.save_pretrained(local_model_dir)
                print(f"{self.model_size} model downloaded and saved to: {local_model_dir}")
                
            except Exception as e2:
                print(f"Downloading from Hugging Face also failed: {e2}")
                raise RuntimeError(f"Unable to load {self.model_size} model, all methods failed")
    
    def _create_instruction_data(self, data: List[Tuple[str, int]]) -> Dataset:
        """Create instruction-formatted training data."""
        instructions = []
        
        for text, label in data:
            sentiment = "positive" if label == 1 else "negative"
            
            # Build instruction format.
            instruction = f"Please analyze the sentiment of the following Weibo text and answer 'positive' or 'negative'.\n\nText: {text}\n\nSentiment: "
            response = sentiment
            
            
            # Combine into full training text.
            full_text = f"{instruction}{response}{self.tokenizer.eos_token}"
            
            instructions.append({
                "instruction": instruction,
                "response": response,
                "text": full_text
            })
        
        return Dataset.from_list(instructions)
    
    def _tokenize_function(self, examples):
        """Tokenization function."""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def _setup_lora(self, **kwargs):
        """Set up LoRA configuration."""
        lora_r = kwargs.get('lora_r', self.config['lora_r'])
        lora_alpha = kwargs.get('lora_alpha', self.config['lora_alpha'])
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=kwargs.get('lora_dropout', 0.1),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        self.lora_model = get_peft_model(self.base_model, lora_config)
        
        # Parameter statistics.
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        
        print(f"LoRA setup complete (r={lora_r}, alpha={lora_alpha})")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable parameter ratio: {trainable_params / total_params * 100:.2f}%")
        self.lora_model.print_trainable_parameters()  # Built-in PEFT parameter stats.
        
        return lora_config
    
    def train(self, train_data: List[Tuple[str, int]], **kwargs) -> None:
        """Train the model."""
        print(f"Starting training for Qwen3-{self.model_size}-LoRA model...")
        
        # Load base model.
        self._load_base_model()
        
        # Set up LoRA.
        self._setup_lora(**kwargs)
        
        # Hyperparameters (recommended values or user-provided overrides).
        num_epochs = kwargs.get('num_epochs', 3)
        batch_size = kwargs.get('batch_size', self.config['recommended_batch_size'] // 2)  # LoRA typically uses smaller batches.
        learning_rate = kwargs.get('learning_rate', self.config['recommended_lr'] / 2)  # LoRA usually uses a smaller learning rate.
        output_dir = kwargs.get('output_dir', f'./models/qwen3_lora_{self.model_size.lower()}_checkpoints')
        
        print(f"Hyperparameters: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        # Create instruction-formatted data.
        train_dataset = self._create_instruction_data(train_data)
        
        # Tokenize data.
        tokenized_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # Training arguments.
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_drop_last=False,
            report_to=None,
        )
        
        # Data collator.
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer.
        trainer = Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training.
        print(f"Starting LoRA fine-tuning...")
        trainer.train()
        
        # Save model.
        self.lora_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"Qwen3-{self.model_size}-LoRA model training completed!")
    
    def _extract_sentiment(self, generated_text: str, instruction: str) -> int:
        """Extract sentiment label from generated text."""
        response = generated_text[len(instruction):].strip()
        
        if "positive" in response.lower():
            return 1
        elif "negative" in response.lower():
            return 0
        else:
            return 0
    
    def predict(self, texts: List[str]) -> List[int]:
        """Predict sentiment for multiple texts."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        predictions = []
        
        self.lora_model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc=f"Predicting with Qwen3-{self.model_size}"):
                pred, _ = self.predict_single(text)
                predictions.append(pred)
        
        return predictions
    
    def predict_single(self, text: str) -> Tuple[int, float]:
        """Predict sentiment for a single text."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        # Build instruction.
        instruction = f"Please analyze the sentiment of the following Weibo text and answer 'positive' or 'negative'.\n\nText: {text}\n\nSentiment: "
        
        # Tokenize.
        inputs = self.tokenizer(instruction, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response.
        self.lora_model.eval()
        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode generated text.
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract sentiment label.
        prediction = self._extract_sentiment(generated_text, instruction)
        confidence = 0.8  # Confidence estimation for generative output is simplified.
        
        return prediction, confidence
    
    def save_model(self, model_path: str = None) -> None:
        """Save model."""
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} has not been trained")
        
        if model_path is None:
            model_path = MODEL_PATHS["lora"][self.model_size]
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save LoRA weights.
        self.lora_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"LoRA model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Load base model.
        self._load_base_model()
        
        # Load LoRA weights.
        self.lora_model = PeftModel.from_pretrained(self.base_model, model_path)
        
        self.model = self.lora_model
        self.is_trained = True
        print(f"Loaded Qwen3-{self.model_size}-LoRA model: {model_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Universal Qwen3-LoRA training script')
    parser.add_argument('--model_size', type=str, choices=['0.6B', '4B', '8B'], 
                        help='Model size')
    parser.add_argument('--train_path', type=str, default='./dataset/train.txt',
                        help='Training data path')
    parser.add_argument('--test_path', type=str, default='./dataset/test.txt',
                        help='Test data path')
    parser.add_argument('--model_path', type=str, help='Model save path (optional)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size (optional, uses recommended value)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate (optional, uses recommended value)')
    parser.add_argument('--lora_r', type=int, help='LoRA rank (optional, uses recommended value)')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum training samples (0 means use all data)')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation-only mode')
    
    args = parser.parse_args()
    
    # If model size is not specified, ask user interactively.
    if not args.model_size:
        print("Qwen3-LoRA model training")
        print("="*40)
        print("Available model sizes:")
        print("  1. 0.6B - Lightweight, fast training, about 8GB VRAM")
        print("  2. 4B  - Medium scale, balanced performance, about 32GB VRAM") 
        print("  3. 8B  - Large scale, best performance, about 64GB VRAM")
        print("\nNote: LoRA fine-tuning generally needs more VRAM than embedding methods")
        
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
        
        print(f"Selected: Qwen3-{args.model_size} + LoRA")
        print()
    
    # Ensure models directory exists.
    os.makedirs('./models', exist_ok=True)
    
    # Create model.
    model = Qwen3LoRAUniversal(args.model_size)
    
    # Determine model save path.
    model_path = args.model_path or MODEL_PATHS["lora"][args.model_size]
    
    if args.eval_only:
        # Evaluation-only mode.
        print(f"Evaluation mode: loading Qwen3-{args.model_size}-LoRA model")
        model.load_model(model_path)
        
        _, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        # Use a small subset for LoRA evaluation.
        test_subset = test_data[:50]
        model.evaluate(test_subset)
    else:
        # Training mode.
        train_data, test_data = BaseQwenModel.load_data(args.train_path, args.test_path)
        
        # Training data handling.
        if args.max_samples > 0:
            train_subset = train_data[:args.max_samples]
            print(f"Using {len(train_subset)} samples for LoRA training")
        else:
            train_subset = train_data
            print(f"Using all {len(train_subset)} samples for LoRA training")
        
        # Prepare training kwargs.
        train_kwargs = {'num_epochs': args.epochs}
        if args.batch_size:
            train_kwargs['batch_size'] = args.batch_size
        if args.learning_rate:
            train_kwargs['learning_rate'] = args.learning_rate
        if args.lora_r:
            train_kwargs['lora_r'] = args.lora_r
        
        # Train model.
        model.train(train_subset, **train_kwargs)
        
        # Evaluate model (small test subset).
        test_subset = test_data[:50]
        model.evaluate(test_subset)
        
        # Save model.
        model.save_model(model_path)
        
        # Example predictions.
        print(f"\nQwen3-{args.model_size}-LoRA example predictions:")
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