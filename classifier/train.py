"""
classifier/train.py

Fine-tune DeBERTa-v3 as a binary prompt-injection classifier.

Usage:
    python -m classifier.train --data-dir ml/ --output-dir ml/deberta-prompt-injection
    
Expects train.csv / validation.csv / test.csv (columns: text, label) in --data-dir.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, and F1 for binary classification."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 for prompt injection detection")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-large", help="Base model")
    parser.add_argument("--data-dir", default="ml", help="Directory with train/validation/test CSVs")
    parser.add_argument("--output-dir", default="ml/deberta-prompt-injection", help="Output directory")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use fp16")
    args = parser.parse_args()

    ds = load_dataset(
        "csv",
        data_files={
            "train": os.path.join(args.data_dir, "train.csv"),
            "validation": os.path.join(args.data_dir, "validation.csv"),
            "test": os.path.join(args.data_dir, "test.csv"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds = ds.map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    
    print("\n=== Test set evaluation ===")
    results = trainer.evaluate(ds["test"])
    for key, value in sorted(results.items()):
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")


if __name__ == "__main__":
    main()
