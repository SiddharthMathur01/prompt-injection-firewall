"""
adjudicator/client.py

Tier 3 — LLM-as-judge adjudicator.

Implements the Adjudicator protocol defined in policy_engine.py. When the
policy cascade can't resolve a request (POL-007), this module calls a
heavier model to make the final block/allow decision.

Supports multiple backends:
  - Local model via transformers (Qwen3, Phi-4-mini, etc.)
  - Remote API call (OpenAI-compatible endpoint)
  - Mock implementation for testing

The CachingAdjudicator wrapper in policy_engine.py handles caching and
timeout fallback, so this module only needs to implement the raw judge()
call.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Import from the policy module
from policy.policy_engine import AdjudicatorResult, MLResult


def _load_prompt_template(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    prompts_dir = Path(__file__).parent / "prompts"
    path = prompts_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _format_rule_matches(matches: list) -> str:
    """Format rule matches for the adjudicator prompt."""
    if not matches:
        return "None"
    parts = []
    for m in matches:
        parts.append(f"  - {m.rule_id}: {m.name} (category={m.category}, severity={m.severity})")
    return "\n".join(parts)


class APIAdjudicator:
    """
    Adjudicator that calls an OpenAI-compatible API endpoint.
    
    This is the recommended production implementation — it can target
    any model exposed via an OpenAI-compatible API (vLLM, Ollama,
    Together, OpenRouter, etc.).
    
    Usage:
        adj = APIAdjudicator(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3-8B",
        )
        result = adj.judge(text, matches, ml_result)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen3:8b",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("ADJUDICATOR_API_KEY", "ollama")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = _load_prompt_template("judge_system.txt")
        self._user_template = _load_prompt_template("judge_user.txt")
    
    def judge(self, text: str, matches: list, ml: MLResult) -> AdjudicatorResult:
        """Call the adjudicator model and parse its verdict."""
        import httpx
        
        user_prompt = self._user_template.format(
            prompt_text=text,
            rule_matches=_format_rule_matches(matches),
            ml_confidence=f"{ml.confidence:.3f}",
            ml_category=ml.category or "none",
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=5.0,
        )
        response.raise_for_status()
        
        content = response.json()["choices"][0]["message"]["content"]
        return self._parse_verdict(content)
    
    def _parse_verdict(self, content: str) -> AdjudicatorResult:
        """Parse the model's JSON response into an AdjudicatorResult."""
        # Try to extract JSON from the response
        content = content.strip()
        
        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Failed to parse — treat as a low-confidence allow
                return AdjudicatorResult(
                    action="allow",
                    confidence=0.3,
                    reasoning=f"Failed to parse adjudicator response: {content[:200]}",
                )
        
        verdict = data.get("verdict", "allow").lower()
        if verdict not in ("block", "allow"):
            verdict = "allow"  # fail safe to allow on ambiguous verdict
        
        return AdjudicatorResult(
            action=verdict,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=str(data.get("reasoning", "No reasoning provided")),
        )


class TransformersAdjudicator:
    """
    Adjudicator that runs a local model via Hugging Face transformers.
    
    For when you want to run the adjudicator entirely locally without
    a separate serving infrastructure. Slower than API-based approaches
    but simpler to deploy.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        device: str = "cpu",
        max_new_tokens: int = 256,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype="auto",
        )
        self.max_new_tokens = max_new_tokens
        self._system_prompt = _load_prompt_template("judge_system.txt")
        self._user_template = _load_prompt_template("judge_user.txt")
    
    def judge(self, text: str, matches: list, ml: MLResult) -> AdjudicatorResult:
        """Run the local model and parse its verdict."""
        user_prompt = self._user_template.format(
            prompt_text=text,
            rule_matches=_format_rule_matches(matches),
            ml_confidence=f"{ml.confidence:.3f}",
            ml_category=ml.category or "none",
        )
        
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=True,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        
        return APIAdjudicator._parse_verdict(None, response)  # reuse parser


class MockAdjudicator:
    """
    Mock adjudicator for testing. Uses simple heuristics instead of a
    real model call.
    """
    
    def judge(self, text: str, matches: list, ml: MLResult) -> AdjudicatorResult:
        """Simple heuristic-based verdict for testing."""
        has_notable_rule = any(
            m.severity in ("medium", "high") for m in matches
        )
        
        if has_notable_rule:
            return AdjudicatorResult(
                action="block",
                confidence=0.72,
                reasoning="Rule pattern plausibly reflects genuine override intent on inspection.",
            )
        return AdjudicatorResult(
            action="allow",
            confidence=0.68,
            reasoning="Signal consistent with benign phrasing on closer reading.",
        )


def create_adjudicator(
    backend: str = "mock",
    **kwargs,
) -> "MockAdjudicator | APIAdjudicator | TransformersAdjudicator":
    """
    Factory function to create an adjudicator instance.
    
    Args:
        backend: "mock", "api", or "transformers"
        **kwargs: Backend-specific arguments
    """
    if backend == "mock":
        return MockAdjudicator()
    elif backend == "api":
        return APIAdjudicator(**kwargs)
    elif backend == "transformers":
        return TransformersAdjudicator(**kwargs)
    else:
        raise ValueError(f"Unknown adjudicator backend: {backend}")
