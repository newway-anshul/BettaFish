## Topic Classification (BERT Chinese Base Model)

This directory provides a Chinese topic classification implementation using `google-bert/bert-base-chinese`:
- Automatically handles local/cached/remote three-stage loading logic;
- `train.py` for fine-tuning training; `predict.py` for single-text or interactive prediction;
- All models and weights are uniformly saved to the `model/` directory here.

Reference Model Card: [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)

### Dataset Highlights

- Approximately **4.1 million** pre-filtered high-quality questions and replies;
- Each question corresponds to one "[Topic]", covering **approximately 28,000** diverse topics;
- Selected from **14 million** original Q&A, retaining only answers with **at least 3 or more likes** to ensure content quality and interestingness;
- In addition to questions, topics, and one or more replies, each reply includes the number of likes, reply ID, and responder tags;
- After data cleaning and deduplication, divided into three parts: approximately **4.12 million** for training set, and several for validation/test (adjustable as needed).

> When actually training, please use the CSV files under `dataset/`; the script will automatically identify common column names or allow explicit specification via command parameters.

### Directory Structure

```
BertTopicDetection_Finetuned/
  ├─ dataset/                   # Data files placed here
  ├─ model/                     # Training outputs; also caches base BERT
  ├─ train.py
  ├─ predict.py
  └─ README.md
```

### Environment

```
pip install torch transformers scikit-learn pandas
```

Or use your existing Conda environment.

### Data Format

CSV must contain at least a text column and a label column; the script will attempt to auto-identify:
- Text column candidates: `text`/`content`/`sentence`/`title`/`desc`/`question`
- Label column candidates: `label`/`labels`/`category`/`topic`/`class`

If explicit specification is needed, use `--text_col` and `--label_col`.

### Training

```
python train.py \
  --train_file ./dataset/web_text_zh_train.csv \
  --valid_file ./dataset/web_text_zh_valid.csv \
  --text_col auto \
  --label_col auto \
  --model_root ./model \
  --save_subdir bert-chinese-classifier \
  --num_epochs 10 --batch_size 16 --learning_rate 2e-5 --fp16
```

Key Points:
- On first run, checks `model/bert-base-chinese`; if not found, tries local cache, then auto-downloads and saves if needed;
- Training process evaluates and saves at each step (default every 1/4 epoch), keeps at most 5 recent checkpoints (adjustable via `SAVE_TOTAL_LIMIT` environment variable);
- Supports early stopping (default patience 5 evaluations), and auto-rollbacks to the best model when evaluation/save strategies are consistent;
- Tokenizer, weights, and `label_map.json` are saved to `model/bert-chinese-classifier/`.

### Optional Chinese Base Models (Interactive Selection Before Training)

Default base model: `google-bert/bert-base-chinese`. When starting training, if the terminal is interactive, the program will prompt you to select from the options below (or enter any Hugging Face model ID):

1) `google-bert/bert-base-chinese`
2) `hfl/chinese-roberta-wwm-ext-large`
3) `hfl/chinese-macbert-large`
4) `IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese`
5) `IDEA-CCNL/Erlangshen-DeBERTa-v3-Base-Chinese`
6) `Langboat/mengzi-bert-base`
7) `BAAI/bge-base-zh` (better suited for retrieval/contrastive learning paradigm)
8) `nghuyong/ernie-3.0-base-zh`

Notes:
- In non-interactive environments (e.g., scheduling systems) or when `NON_INTERACTIVE=1` is set, the model specified by command-line parameter `--pretrained_name` will be used directly (default is `google-bert/bert-base-chinese`).
- After selection, the base model will be downloaded/cached to the `model/` directory for unified management.

### Prediction

Single text:
```
python predict.py --text "Which topic is this Weibo post discussing?" --model_root ./model --finetuned_subdir bert-chinese-classifier
```

Interactive:
```
python predict.py --interactive --model_root ./model --finetuned_subdir bert-chinese-classifier
```

Example Output:
```
Prediction Result: Sports-Football (Confidence: 0.9412)
```

### Explanation

- Both training and prediction have built-in basic Chinese text cleaning.
- The label set is based on the training set; the script automatically generates and saves `label_map.json`.

### Training Strategy (Brief Description)

- Base model: `google-bert/bert-base-chinese`; classification head dimension = number of unique labels in training set.
- Learning rate and regularization: `lr=2e-5`, `weight_decay=0.01`, can be fine-tuned to `1e-5~3e-5` on large datasets.
- Sequence length and batch size: `max_length=128`, `batch_size=16`; if truncation is severe, can increase to 256 (higher cost).
- Warmup: If supported by environment, use `warmup_ratio=0.1`; otherwise fall back to `warmup_steps=0`.
- Evaluation/Saving: Steps calculated by `--eval_fraction` (default 0.25), `save_total_limit=5` limits disk usage.
- Early stopping: Monitors weighted F1 (higher is better), default patience 5, improvement threshold 0.0.
- Single GPU stable operation: By default only one GPU is used, can be specified via `--gpu`; script cleans up distributed environment variables.


### Author Notes (On Ultra-Large Scale Multi-Classification)

- When topic categories reach tens of thousands, directly adding a single linear classification head (large softmax) after the encoder is often limited: long-tail categories are hard to learn, semantic sparsity, new topics cannot be adaptively integrated, and frequent retraining is needed after deployment.
- Improvement ideas (by recommended priority):
  - Retrieval/dual-tower paradigm (text vs. topic name/description contrastive learning) + nearest neighbor search + lightweight reranking, naturally supports incremental category expansion and fast updates;
  - Hierarchical classification (coarse-grained then fine-grained), significantly reduces single-head difficulty and computation;
  - Text-label joint modeling (using label descriptions), improves transferability of synonymous topics;
  - Training details: class-balanced/focal/label smoothing, sampled softmax, contrastive pre-training, etc.
- Important Declaration: The "static classification head fine-tuning" used in this directory is only a backup and learning reference. For English/multilingual short-text scenarios, topic changes rapidly, and traditional static classifiers struggle to keep pace; our work focuses on generative/self-supervised topic discovery and dynamic system construction directions like `TopicGPT`; this implementation aims to provide a runnable baseline and engineering example.


