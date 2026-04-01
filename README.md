# Fine-Tuning DistilBERT for Emotion Recognition (HuggingFace)

## Project Overview

This project fine-tunes DistilBERT ("distilbert-base-uncased") on the `dair-ai/emotion` dataset to perform multi-class emotion classification (e.g., joy, sadness, anger, fear, love, surprise). The main implementation is in the provided Jupyter notebook.

## Architecture (Data flow)

- Dataset: `dair-ai/emotion` loaded via the `datasets` library.
- Preprocessing: convert dataset to pandas for EDA; map numeric labels to human-readable `label_name`.
- Tokenization: `AutoTokenizer.from_pretrained('distilbert-base-uncased')` with padding/truncation (max_length=512).
- Model: `AutoModelForSequenceClassification` based on DistilBERT with `num_labels=num_labels`.
- Training: HuggingFace `Trainer` using `TrainingArguments` to handle training loop, evaluation and checkpoints.
- Evaluation: compute accuracy and weighted F1, print classification report on test set.

Simple flow diagram:

Text -> Tokenizer -> Input IDs -> DistilBERT -> Classification head -> Softmax -> Predicted label

## Implementation Details

- Tokenization is applied in a batched `dataset.map(tokenize, batched=True)` call with `padding=True` and `truncation=True`.
- The notebook initializes the model with `AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)` and moves it to GPU if available.
- TrainingArguments used in the notebook (as-is):

  - `output_dir='distilbert-finetuned-emotion'`
  - `num_train_epochs=2`
  - `per_device_train_batch_size=64` (reduce if you hit OOM)
  - `per_device_eval_batch_size=64`
  - `eval_strategy='epoch'`
  - `weight_decay=0.01`
  - `learning_rate=2e-5`

- `compute_metrics` calculates `accuracy` and weighted `f1_score` (scikit-learn).

## Files

- Notebook: Fine_Tuning_DistilBERT_for_Emotion_Recognition_using_HuggingFace.ipynb — main end-to-end pipeline (data load, EDA, tokenization, training, evaluation).
- README.md — this file.

## How to run (quick)

1. Create a Python environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main packages:

```powershell
pip install transformers datasets torch scikit-learn pandas matplotlib plotly
```

2. Start Jupyter and open the notebook:

```powershell
jupyter notebook Fine_Tuning_DistilBERT_for_Emotion_Recognition_using_HuggingFace.ipynb
```

3. Run cells top-to-bottom. Use a GPU runtime for reasonable training time.

## Notes & Tips

- The notebook uses `max_length=512` for tokenization; if you need faster runs or less memory use, lower it (e.g., 128 or 256).
- `per_device_train_batch_size=64` may be too large for most consumer GPUs — reduce to 8/16/32 as needed.
- Consider adding checkpointing, early stopping and a script-based training runner for reproducibility.

## Next steps (suggested)

- Add `requirements.txt` and an executable training script (`train.py`).
- Save and version best model checkpoints and evaluation artifacts.
- Add a small README section with expected hardware and run-time estimates.

---

If you want, I can also generate a `requirements.txt` and a small `train.py` wrapper next.
