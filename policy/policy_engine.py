"""
policy/policy_engine.py

Evaluates policy_rules.yaml against a request's signals:
  - rule_matches: list of RuleMatch (from the regex rule engine)
  - ml_result:    MLResult (from the classifier)
  - source:       SourceContext (behavioral/rate signals for this identity)

Returns a Decision with the action AND the specific rule id that fired,
so every block/allow is traceable back to exactly one auditable reason.

Tier-3 adjudication: when POL-007 (ambiguous_adjudication) fires, the
engine calls an injected Adjudicator and folds its verdict into the final
Decision.
"""
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import yaml


@dataclass
class MLResult:
    """Output from the ML classifier, adapted for the policy engine."""
    confidence: float           # 0.0–1.0
    category: Optional[str]     # predicted attack category, if any


@dataclass
class SourceContext:
    """Behavioral/identity signals for this request source."""
    identity: str
    flag_count_last_5m: int = 0


@dataclass
class Decision:
    """The final decision for a request — the auditable output of the firewall."""
    action: str             # "block" | "flag_for_review" | "allow_and_log" | "allow"
    rule_id: str            # which policy rule produced this decision
    rule_name: str
    reason: str             # human-readable explanation
    adjudicator_verdict: Optional["AdjudicatorResult"] = None


@dataclass
class AdjudicatorResult:
    """Verdict from the Tier 3 adjudicator model."""
    action: str             # "block" | "allow"
    confidence: float
    reasoning: str          # short human-readable explanation
    from_cache: bool = False
    timed_out: bool = False


class Adjudicator(Protocol):
    """Protocol for Tier 3 adjudicator implementations."""
    def judge(self, text: str, matches: list, ml: MLResult) -> AdjudicatorResult:
        ...


class CachingAdjudicator:
    """
    Wraps any Adjudicator with caching + timeout-fallback behavior.
    """
    def __init__(self, inner: Adjudicator, timeout_ms: int = 2000, cache_ttl_s: int = 600):
        self.inner = inner
        self.timeout_ms = timeout_ms
        self.cache_ttl_s = cache_ttl_s
        self._cache: dict[str, tuple[float, AdjudicatorResult]] = {}

    def _cache_key(self, text: str) -> str:
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def judge(self, text: str, matches: list, ml: MLResult) -> AdjudicatorResult:
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached and (time.time() - cached[0]) < self.cache_ttl_s:
            result = cached[1]
            return AdjudicatorResult(
                result.action, result.confidence, result.reasoning, from_cache=True
            )

        try:
            result = self.inner.judge(text, matches, ml)
        except Exception as e:
            return AdjudicatorResult(
                action="allow",
                confidence=0.0,
                reasoning=f"adjudicator error, falling back: {e}",
                timed_out=True,
            )

        self._cache[key] = (time.time(), result)
        return result


# ---------------------------------------------------------------------------
# Per-rule condition checks
# ---------------------------------------------------------------------------

def _check_behavioral_escalation(matches, ml, source, params):
    return source.flag_count_last_5m >= 3


def _check_critical_rule_match(matches, ml, source, params):
    return any(m.severity == "critical" for m in matches)


def _check_high_ml_confidence(matches, ml, source, params):
    return ml.confidence > params["ml_threshold"]


def _check_rule_and_ml_agree(matches, ml, source, params):
    if not matches or ml.confidence <= params["ml_threshold"]:
        return False
    if params.get("require_same_category"):
        return ml.category in {m.category for m in matches}
    return True


def _check_multi_rule_corroboration(matches, ml, source, params):
    distinct_categories = {m.category for m in matches}
    return len(distinct_categories) >= params["min_distinct_categories"]


def _check_encoding_evasion_elevated(matches, ml, source, params):
    return any(m.category == "encoding_evasion" or m.category == "encoding_attack" for m in matches)


def _check_cheap_allow_negligible_signal(matches, ml, source, params):
    return (
        len(matches) == 1
        and matches[0].severity == "low"
        and ml.confidence < params["ml_threshold"]
    )


def _check_ambiguous_adjudication(matches, ml, source, params):
    return len(matches) > 0 or ml.confidence >= params["ml_threshold_low"]


def _check_default_allow(matches, ml, source, params):
    return True


RULE_CHECKS: dict[str, Callable] = {
    "POL-000": _check_behavioral_escalation,
    "POL-001": _check_critical_rule_match,
    "POL-002": _check_high_ml_confidence,
    "POL-003": _check_rule_and_ml_agree,
    "POL-004": _check_multi_rule_corroboration,
    "POL-005": _check_encoding_evasion_elevated,
    "POL-006": _check_cheap_allow_negligible_signal,
    "POL-007": _check_ambiguous_adjudication,
    "POL-999": _check_default_allow,
}


class PolicyEngine:
    """
    Evaluates the policy rule cascade against Tier 1 + Tier 2 outputs.
    First matching rule wins.
    """
    def __init__(
        self,
        rules_path: Optional[str] = None,
        adjudicator: Optional[Adjudicator] = None,
    ):
        if rules_path is None:
            rules_path = os.path.join(os.path.dirname(__file__), "policy_rules.yaml")
        
        with open(rules_path) as f:
            config = yaml.safe_load(f)
        
        self.rules = sorted(config["rules"], key=lambda r: r["priority"])
        adj_config = config.get("adjudicator", {})
        self.adjudicator_enabled = adj_config.get("enabled", False)
        self.fallback = adj_config.get("on_timeout_fallback", {})
        self.adjudicator = (
            CachingAdjudicator(
                adjudicator,
                timeout_ms=adj_config.get("timeout_ms", 2000),
                cache_ttl_s=adj_config.get("cache_ttl_seconds", 600),
            )
            if adjudicator
            else None
        )

    def evaluate(
        self,
        text: str,
        rule_matches: list,
        ml_result: MLResult,
        source: SourceContext,
    ) -> Decision:
        """Evaluate the policy cascade. First matching rule wins."""
        for rule in self.rules:
            check = RULE_CHECKS.get(rule["id"])
            if check is None:
                raise ValueError(f"No check function registered for {rule['id']}")
            params = rule.get("params", {})
            if not check(rule_matches, ml_result, source, params):
                continue

            if rule["action"] == "escalate_to_adjudicator":
                return self._resolve_via_adjudicator(
                    rule, text, rule_matches, ml_result
                )

            return Decision(
                action=rule["action"],
                rule_id=rule["id"],
                rule_name=rule["name"],
                reason=rule["description"].strip(),
            )
        raise RuntimeError("No policy rule matched, including default_allow")

    def _resolve_via_adjudicator(self, rule, text, rule_matches, ml_result) -> Decision:
        if not self.adjudicator_enabled or self.adjudicator is None:
            has_notable_rule = any(
                m.severity in ("medium", "high") for m in rule_matches
            )
            fallback_action = self.fallback.get(
                "has_high_or_medium_rule"
                if has_notable_rule
                else "rule_or_ml_only_weak",
                "flag_for_review",
            )
            return Decision(
                fallback_action,
                rule["id"],
                rule["name"] + "_no_adjudicator",
                rule["description"].strip(),
            )

        verdict = self.adjudicator.judge(text, rule_matches, ml_result)

        if verdict.timed_out:
            has_notable_rule = any(
                m.severity in ("medium", "high") for m in rule_matches
            )
            fallback_action = self.fallback.get(
                "has_high_or_medium_rule"
                if has_notable_rule
                else "rule_or_ml_only_weak",
                "flag_for_review",
            )
            return Decision(
                fallback_action,
                rule["id"],
                rule["name"] + "_timeout_fallback",
                verdict.reasoning,
                adjudicator_verdict=verdict,
            )

        return Decision(
            action=verdict.action,
            rule_id=rule["id"],
            rule_name=rule["name"],
            reason=f"Escalated to adjudicator: {verdict.reasoning}",
            adjudicator_verdict=verdict,
        )
