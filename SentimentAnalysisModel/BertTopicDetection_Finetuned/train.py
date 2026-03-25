import os
import sys
import json
import re
import argparse
import math
import inspect
from typing import Dict, List, Optional, Tuple

# ========== Single GPU Lock (executed before importing torch/transformers) ==========
def _extract_gpu_arg(argv: List[str], default: str = "0") -> str:
    for i, arg in enumerate(argv):
        if arg.startswith("--gpu="):
            return arg.split("=", 1)[1]
        if arg == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
    return default

env_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
try:
    gpu_to_use = _extract_gpu_arg(sys.argv, default="0")
except Exception:
    gpu_to_use = "0"
# If not set or multiple GPUs are exposed, force single GPU (default 0) for stable direct execution
if (not env_vis) or ("," in env_vis):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_to_use
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Clean up distributed env vars that may be injected by external launchers to avoid accidental multi-GPU/distributed usage
for _k in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
    os.environ.pop(_k, None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
try:
    from transformers import EarlyStoppingCallback  # type: ignore
except Exception:  # pragma: no cover
    EarlyStoppingCallback = None  # type: ignore

# Preset optional Chinese base models (extendable)
BACKBONE_CANDIDATES: List[Tuple[str, str]] = [
    ("1) google-bert/bert-base-chinese", "google-bert/bert-base-chinese"),
    ("2) hfl/chinese-roberta-wwm-ext-large", "hfl/chinese-roberta-wwm-ext-large"),
    ("3) hfl/chinese-macbert-large", "hfl/chinese-macbert-large"),
    ("4) IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese", "IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese"),
    ("5) IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese", "IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese"),
    ("6) Langboat/mengzi-bert-base", "Langboat/mengzi-bert-base"),
    ("7) BAAI/bge-base-zh", "BAAI/bge-base-zh"),
    ("8) nghuyong/ernie-3.0-base-zh", "nghuyong/ernie-3.0-base-zh"),
]


def prompt_backbone_interactive(current_id: str) -> str:
    """Interactively select a base model.

    - In non-interactive environments (stdin is not a TTY) or when NON_INTERACTIVE=1 is set, returns current_id directly.
    - User can enter a number to select a preset, or paste any Hugging Face model ID.
    - Empty Enter uses the current default.
    """
    if os.environ.get("NON_INTERACTIVE", "0") == "1":
        return current_id
    try:
        if not sys.stdin.isatty():
            return current_id
    except Exception:
        return current_id

    print("\nOptional Chinese base models (press Enter to use default):")
    for label, hf_id in BACKBONE_CANDIDATES:
        print(f"  {label}")
    print(f"Current default: {current_id}")
    choice = input("Enter number or paste model ID (Enter to keep default): ").strip()
    if not choice:
        return current_id
    # Numeric selection
    if choice.isdigit():
        idx = int(choice)
        for label, hf_id in BACKBONE_CANDIDATES:
            if label.startswith(f"{idx})"):
                return hf_id
        print("Number not found, keeping default.")
        return current_id
    # Custom HF model ID
    return choice


def preprocess_text(text: str) -> str:
    return from django.utils.translation import ungettext


def ensure_base_model_local(model_name_or_path: str, local_model_root: str) -> Tuple[str, AutoTokenizer]:
    os.makedirs(local_model_root, exist_ok=True)
    base_dir = os.path.join(local_model_root, "bert-base-chinese")

    def is_ready(path: str) -> bool:
        return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))

    # 1) Local ready
    if is_ready(base_dir):
        tokenizer = AutoTokenizer.from_pretrained(base_dir)
        return base_dir, tokenizer

    # 2) Local cache
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        base = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
        os.makedirs(base_dir, exist_ok=True)
        tokenizer.save_pretrained(base_dir)
        base.save_pretrained(base_dir)
        return base_dir, tokenizer
    except Exception:
        pass

    # 3) Remote download
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    base = AutoModel.from_pretrained(model_name_or_path)
    os.makedirs(base_dir, exist_ok=True)
    tokenizer.save_pretrained(base_dir)
    base.save_pretrained(base_dir)
    return base_dir, tokenizer


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: AutoTokenizer,
        text_column: str,
        label_column: str,
        label2id: Dict[str, int],
        max_length: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        text = preprocess_text(row[self.text_column])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.label_column in row and pd.notna(row[self.label_column]):
            label_str = str(row[self.label_column])
            item["labels"] = torch.tensor(self.label2id[label_str], dtype=torch.long)
        return item


def build_label_mappings(train_df: pd.DataFrame, label_column: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels: List[str] = [str(x) for x in train_df[label_column].dropna().astype(str).tolist()]
    unique_sorted = sorted(set(labels))
    label2id = {label: i for i, label in enumerate(unique_sorted)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics_fn(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def autodetect_columns(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[str, str]:
    if text_col != "auto" and label_col != "auto":
        return text_col, label_col
    candidates_text = ["text", "content", "sentence", "title", "desc", "question"]
    candidates_label = ["label", "labels", "category", "topic", "class"]
    t = text_col
    l = label_col
    if text_col == "auto":
        for name in candidates_text:
            if name in df.columns:
                t = name
                break
    if label_col == "auto":
        for name in candidates_label:
            if name in df.columns:
                l = name
                break
    if t == "auto" or l == "auto":
        raise ValueError(
            f"Cannot auto-detect column names, please explicitly pass --text_col and --label_col. Available columns: {list(df.columns)}"
        )
    return t, l


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune text classification using google-bert/bert-base-chinese on the local dataset")
    parser.add_argument("--train_file", type=str, default="./dataset/web_text_zh_train.csv")
    parser.add_argument("--valid_file", type=str, default="./dataset/web_text_zh_valid.csv")
    parser.add_argument("--text_col", type=str, default="auto", help="Text column name, auto-detected by default")
    parser.add_argument("--label_col", type=str, default="auto", help="Label column name, auto-detected by default")
    parser.add_argument("--model_root", type=str, default="./model", help="Local model root directory")
    parser.add_argument("--pretrained_name", type=str, default="google-bert/bert-base-chinese", help="Hugging Face model ID; leave empty to enter interactive selection")
    parser.add_argument("--save_subdir", type=str, default="bert-chinese-classifier")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"), help="Specify single GPU, e.g. 0 or 1")
    parser.add_argument("--eval_fraction", type=float, default=0.25, help="How often to evaluate and save per epoch, e.g. 0.25 means every quarter epoch")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience (in evaluation rounds)")
    parser.add_argument("--early_stop_threshold", type=float, default=0.0, help="Early stopping minimum improvement threshold (same unit as metric_for_best_model)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = args.model_root if os.path.isabs(args.model_root) else os.path.join(script_dir, args.model_root)
    os.makedirs(model_root, exist_ok=True)

    # Interactively select base model (if interaction allowed and not disabled by env)
    selected_model_id = prompt_backbone_interactive(args.pretrained_name)
    # Ensure base model is ready
    base_dir, tokenizer = ensure_base_model_local(selected_model_id, model_root)
    print(f"[Info] Using base model directory: {base_dir}")

    # Load data
    train_path = args.train_file if os.path.isabs(args.train_file) else os.path.join(script_dir, args.train_file)
    valid_path = args.valid_file if os.path.isabs(args.valid_file) else os.path.join(script_dir, args.valid_file)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(f"Training set not found: {train_path}")
    train_df = pd.read_csv(train_path)
    if not os.path.isfile(valid_path):
        # Auto-split if validation set is not provided or does not exist
        shuffled = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.9)
        valid_df = shuffled.iloc[split_idx:].reset_index(drop=True)
        train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
    else:
        valid_df = pd.read_csv(valid_path)
    print(f"[Info] Training set: {train_path} | Samples: {len(train_df)}")
    print(f"[Info] Validation set: {valid_path if os.path.isfile(valid_path) else '(split from training set)'} | Samples: {len(valid_df)}")

    # Auto-detect column names
    text_col, label_col = autodetect_columns(train_df, args.text_col, args.label_col)
    print(f"[Info] Text column: {text_col} | Label column: {label_col}")

    # Label mapping (using union of training+validation sets to avoid errors from unseen labels in validation)
    combined_labels_df = pd.concat([train_df[[label_col]], valid_df[[label_col]]], ignore_index=True)
    label2id, id2label = build_label_mappings(combined_labels_df, label_col)
    if len(label2id) < 2:
        raise ValueError("Number of label classes is less than 2, cannot train classification model.")
    print(f"[Info] Number of label classes: {len(label2id)}")
    # Warn about labels in validation set not seen in training
    try:
        train_label_set = set(str(x) for x in train_df[label_col].dropna().astype(str).tolist())
        valid_label_set = set(str(x) for x in valid_df[label_col].dropna().astype(str).tolist())
        unseen_in_train = sorted(valid_label_set - train_label_set)
        if unseen_in_train:
            preview = ", ".join(unseen_in_train[:10])
            print(f"[Warn] Validation set contains {len(unseen_in_train)} labels not seen in training (included in mapping to avoid errors). Examples: {preview} ...")
    except Exception:
        pass

    # Dataset
    train_dataset = TextClassificationDataset(train_df, tokenizer, text_col, label_col, label2id, args.max_length)
    eval_dataset = TextClassificationDataset(valid_df, tokenizer, text_col, label_col, label2id, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    config = AutoConfig.from_pretrained(
        base_dir,
        num_labels=len(label2id),
        id2label={int(i): str(l) for i, l in id2label.items()},
        label2id={str(l): int(i) for l, i in label2id.items()},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_dir,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Training arguments
    output_dir = os.path.join(model_root, args.save_subdir)
    os.makedirs(output_dir, exist_ok=True)
    # Training arguments (compatible with different transformers versions)
    args_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_epochs,
        "logging_steps": 100,
        "fp16": args.fp16,
        "seed": args.seed,
    }

    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    # Optional parameters (only add if supported, simplified for compatibility)
    if "warmup_ratio" in allowed:
        args_dict["warmup_ratio"] = args.warmup_ratio
    if "report_to" in allowed:
        args_dict["report_to"] = []
    # Eval/save step interval: convert eval_fraction to steps per epoch
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(1, args.batch_size)))
    eval_every_steps = max(1, math.ceil(steps_per_epoch * max(0.01, min(1.0, args.eval_fraction))))
    # Strategy mode (compatible with new/old version field names)
    key_eval = "evaluation_strategy" if "evaluation_strategy" in allowed else ("eval_strategy" if "eval_strategy" in allowed else None)
    if key_eval:
        args_dict[key_eval] = "steps"
    if "save_strategy" in allowed:
        args_dict["save_strategy"] = "steps"
    if "eval_steps" in allowed:
        args_dict["eval_steps"] = eval_every_steps
    if "save_steps" in allowed:
        args_dict["save_steps"] = eval_every_steps
    if "save_total_limit" in allowed:
        args_dict["save_total_limit"] = 5
    # Align logging step with eval/save step to reduce output noise
    if "logging_steps" in allowed:
        args_dict["logging_steps"] = eval_every_steps
    # Best model rollback (only enabled when evaluation and save strategy are consistent)
    if "metric_for_best_model" in allowed:
        args_dict["metric_for_best_model"] = "f1"
    if "greater_is_better" in allowed:
        args_dict["greater_is_better"] = True
    if "load_best_model_at_end" in allowed:
        eval_strat = args_dict.get("evaluation_strategy", args_dict.get("eval_strategy"))
        save_strat = args_dict.get("save_strategy")
        if eval_strat == save_strat and eval_strat in ("steps", "epoch"):
            args_dict["load_best_model_at_end"] = True

    # Compatibility for versions without warmup_ratio: use warmup_steps=0 if supported
    if "warmup_ratio" not in allowed and "warmup_steps" in allowed:
        # Do not calculate total steps, default to 0
        args_dict["warmup_steps"] = 0

    # If strategy parameters are not supported: fall back to saving/evaluating every eval_every_steps
    if "save_strategy" not in allowed and "save_steps" in allowed:
        args_dict["save_steps"] = eval_every_steps
    if ("evaluation_strategy" not in allowed and "eval_strategy" not in allowed) and "eval_steps" in allowed:
        args_dict["eval_steps"] = eval_every_steps

    # If load_best_model_at_end is supported but eval/save strategy cannot be set consistently, disable it to avoid errors
    if "load_best_model_at_end" in allowed:
        want_load_best = args_dict.get("load_best_model_at_end", False)
        eval_set = args_dict.get("evaluation_strategy", None)
        save_set = args_dict.get("save_strategy", None)
        if want_load_best and (eval_set is None or save_set is None or eval_set != save_set):
            args_dict["load_best_model_at_end"] = False

    training_args = TrainingArguments(**args_dict)
    print("[Info] Training argument summary:")
    print(f"       epochs={args.num_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}, weight_decay={args.weight_decay}")
    print(f"       max_length={args.max_length}, seed={args.seed}, fp16={args.fp16}")
    if "warmup_ratio" in allowed and "warmup_ratio" in args_dict:
        print(f"       warmup_ratio={args_dict['warmup_ratio']}")
    elif "warmup_steps" in allowed and "warmup_steps" in args_dict:
        print(f"       warmup_steps={args_dict['warmup_steps']}")
    print(f"       steps_per_epoch={steps_per_epoch}, eval_every_steps={eval_every_steps}")
    print(f"       eval_strategy={args_dict.get('evaluation_strategy', args_dict.get('eval_strategy'))}, save_strategy={args_dict.get('save_strategy')}, logging_steps={args_dict.get('logging_steps')}")
    print(f"       save_total_limit={args_dict.get('save_total_limit', 'n/a')}, load_best_model_at_end={args_dict.get('load_best_model_at_end', False)}")

    callbacks = []
    if EarlyStoppingCallback is not None and (args_dict.get("evaluation_strategy") in ("steps", "epoch") or "eval_steps" in allowed):
        try:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stop_patience,
                    early_stopping_threshold=args.early_stop_threshold,
                )
            )
        except Exception:
            pass

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    # Device and GPU info
    try:
        device_cnt = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0) if device_cnt > 0 else "cpu"
        print(f"[Info] CUDA visible device count: {device_cnt}, current device: {dev_name}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    except Exception:
        pass

    print("[Info] Starting training ...")

    trainer.train()

    # Save
    tokenizer.save_pretrained(output_dir)
    trainer.model.config.id2label = {int(i): str(l) for i, l in id2label.items()}
    trainer.model.config.label2id = {str(l): int(i) for l, i in label2id.items()}
    trainer.save_model(output_dir)
    try:
        best_metric = getattr(trainer.state, "best_metric", None)
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_metric is not None and best_ckpt is not None:
            print(f"[Info] Best model: metric={best_metric:.6f} | checkpoint={best_ckpt}")
    except Exception:
        pass

    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"label2id": trainer.model.config.label2id, "id2label": trainer.model.config.id2label},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Training curve: optionally save training and validation loss
    try:
        import matplotlib.pyplot as plt  # type: ignore
        logs = trainer.state.log_history
        t_steps, t_losses, e_steps, e_losses = [], [], [], []
        step_counter = 0
        for rec in logs:
            if "loss" in rec and "epoch" in rec:
                step_counter += 1
                t_steps.append(step_counter)
                t_losses.append(rec["loss"])
            if "eval_loss" in rec:
                e_steps.append(step_counter)
                e_losses.append(rec["eval_loss"])
        if t_losses or e_losses:
            plt.figure(figsize=(8,4))
            if t_losses:
                plt.plot(t_steps, t_losses, label="train_loss")
            if e_losses:
                plt.plot(e_steps, e_losses, label="eval_loss")
            plt.xlabel("training step (logged)")
            plt.ylabel("loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_curve.png"))
    except Exception:
        pass

    print(f"Fine-tuning complete, model saved to: {output_dir}")


if __name__ == "__main__":
    main()


