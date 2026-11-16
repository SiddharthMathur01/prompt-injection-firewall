"""
classifier/inference.py

Tier 2 — ML intent classifier.

Loads a fine-tuned DeBERTa-v3 model (or ONNX-exported variant) and scores
a prompt for injection intent. Returns a confidence score (0.0–1.0) and
a predicted category.

Designed to run on CPU with acceptable latency (~10-50ms per prompt).
The ONNX path is preferred in production for ~3x speedup over PyTorch.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure OpenMP runtime does not abort on Windows conda environments
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


@dataclass
class ClassifierResult:
    """Output of the ML classifier."""
    confidence: float         # 0.0–1.0, probability of injection
    category: Optional[str]   # predicted attack category, if available
    label: int                # 0=benign, 1=injection
    raw_scores: dict          # raw logits/probabilities for both classes


class PromptClassifier:
    """
    Loads a fine-tuned transformer model and scores text for prompt
    injection intent.
    
    Supports two backends:
    - PyTorch (transformers): default, broadest compatibility
    - ONNX Runtime: ~3x faster inference, preferred for production
    
    Usage:
        classifier = PromptClassifier("ml/deberta-prompt-injection/final")
        result = classifier.classify("ignore all previous instructions")
        print(result.confidence, result.label)
    """
    
    def __init__(
        self,
        model_path: str = "ml/deberta-prompt-injection/final",
        device: str = "cpu",
        max_length: int = 256,
        category_map_path: Optional[str] = None,
        use_onnx: bool = False,
        onnx_path: Optional[str] = None,
    ):
        self.max_length = max_length
        self.device = device
        self.use_onnx = use_onnx
        self._model = None
        self._tokenizer = None
        self._ort_session = None
        self._category_map: dict[int, str] = {}
        
        # Load category map if available
        if category_map_path and os.path.exists(category_map_path):
            with open(category_map_path) as f:
                raw = json.load(f)
                # Map expects str keys -> str values; convert to int->str
                self._category_map = {int(k): v for k, v in raw.items()}
        
        if use_onnx and onnx_path:
            self._load_onnx(onnx_path, model_path)
        else:
            self._load_pytorch(model_path)
    
    def _load_pytorch(self, model_path: str) -> None:
        """Load the PyTorch model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(self.device)
        self._model.eval()
    
    def _load_onnx(self, onnx_path: str, tokenizer_path: str) -> None:
        """Load the ONNX model for faster inference."""
        import onnxruntime as ort
        from transformers import AutoTokenizer
        
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_session = ort.InferenceSession(onnx_path, sess_options)
    
    def classify(self, text: str) -> ClassifierResult:
        """
        Score a single prompt for injection intent.
        
        Args:
            text: The prompt text to classify.
            
        Returns:
            ClassifierResult with confidence, category, label, and raw scores.
        """
        if self.use_onnx and self._ort_session is not None:
            return self._classify_onnx(text)
        return self._classify_pytorch(text)
    
    def _classify_pytorch(self, text: str) -> ClassifierResult:
        """Classify using the PyTorch model."""
        import torch
        
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        label = int(np.argmax(probs))
        confidence = float(probs[1])  # probability of injection class
        
        return ClassifierResult(
            confidence=confidence,
            category=self._category_map.get(label),
            label=label,
            raw_scores={"benign": float(probs[0]), "injection": float(probs[1])},
        )
    
    def _classify_onnx(self, text: str) -> ClassifierResult:
        """Classify using the ONNX Runtime model."""
        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
            padding=True,
        )
        
        feed = {k: v for k, v in inputs.items() if k in [inp.name for inp in self._ort_session.get_inputs()]}
        outputs = self._ort_session.run(None, feed)
        
        logits = outputs[0][0]
        # softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        label = int(np.argmax(probs))
        confidence = float(probs[1])
        
        return ClassifierResult(
            confidence=confidence,
            category=self._category_map.get(label),
            label=label,
            raw_scores={"benign": float(probs[0]), "injection": float(probs[1])},
        )
    
    def classify_batch(self, texts: list[str]) -> list[ClassifierResult]:
        """
        Classify a batch of prompts.
        
        Args:
            texts: List of prompt texts to classify.
            
        Returns:
            List of ClassifierResult, one per input text.
        """
        if self.use_onnx and self._ort_session is not None:
            return [self._classify_onnx(t) for t in texts]
        return self._classify_pytorch_batch(texts)
    
    def _classify_pytorch_batch(self, texts: list[str]) -> list[ClassifierResult]:
        """Batch classify using PyTorch."""
        import torch
        
        inputs = self._tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        
        results = []
        for i in range(len(texts)):
            label = int(np.argmax(probs[i]))
            results.append(ClassifierResult(
                confidence=float(probs[i][1]),
                category=self._category_map.get(label),
                label=label,
                raw_scores={"benign": float(probs[i][0]), "injection": float(probs[i][1])},
            ))
        return results


def cli_scan():
    """CLI entry point for scanning a single prompt with the ML classifier."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Scan text with the ML prompt-injection classifier")
    parser.add_argument("prompt", nargs="*", help="Prompt text to scan")
    parser.add_argument("--model", default="ml/deberta-prompt-injection/final", help="Path to model")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    
    if args.prompt:
        text = " ".join(args.prompt)
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = input("Enter prompt: ")
    
    classifier = PromptClassifier(model_path=args.model, device=args.device)
    result = classifier.classify(text)
    
    if args.json:
        import json as json_mod
        print(json_mod.dumps({
            "confidence": result.confidence,
            "label": result.label,
            "category": result.category,
            "raw_scores": result.raw_scores,
        }, indent=2))
    else:
        verdict = "INJECTION" if result.label == 1 else "BENIGN"
        print(f"{verdict} (confidence: {result.confidence:.3f})")
    
    return 1 if result.label == 1 else 0


if __name__ == "__main__":
    raise SystemExit(cli_scan())
