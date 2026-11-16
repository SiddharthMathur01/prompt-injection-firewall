"""
Fine-tune deBERTa-v3-large as a binary prompt-injection classifier on the
dataset produced by step1-3.

Usage:
    pip install transformers datasets accelerate scikit-learn torch
    python train_deberta.py

Expects train.csv / validation.csv / test.csv (columns: text,label) in the
same directory as this script (or edit DATA_DIR below).
"""
import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 256

ds = load_dataset(
    "csv",
    data_files={
        "train": f"{DATA_DIR}/train.csv",
        "validation": f"{DATA_DIR}/validation.csv",
        "test": f"{DATA_DIR}/test.csv",
    },
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

ds = ds.map(tokenize, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

args = TrainingArguments(
    output_dir="./deberta-prompt-injection",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=20,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    print(trainer.evaluate(ds["test"]))
    trainer.save_model("./deberta-prompt-injection/final")
    tokenizer.save_pretrained("./deberta-prompt-injection/final")