"""
tests/test_adjudicator.py

Unit tests for Tier 3 adjudicator module (client, caching, fallbacks).
"""
import pytest
from policy.policy_engine import (
    CachingAdjudicator,
    AdjudicatorResult,
    MLResult,
)
from policy.rule_engine import RuleMatch
from adjudicator.client import MockAdjudicator, create_adjudicator


def test_mock_adjudicator_high_severity():
    """Mock adjudicator blocks when high severity rule is present."""
    adj = MockAdjudicator()
    matches = [
        RuleMatch(
            rule_id="PI-001",
            name="Ignore Prev",
            category="prompt_injection",
            severity="high",
            severity_int=3,
            description="desc",
            source_file="injections.yaml",
            pattern="pattern",
        )
    ]
    res = adj.judge("test", matches, MLResult(confidence=0.3, category=None))
    assert res.action == "block"
    assert res.confidence > 0.5


def test_mock_adjudicator_low_severity():
    """Mock adjudicator allows when only low severity rules or no notable rules present."""
    adj = MockAdjudicator()
    res = adj.judge("benign question", [], MLResult(confidence=0.2, category=None))
    assert res.action == "allow"
    assert res.confidence > 0.5


def test_caching_adjudicator():
    """Caching wrapper returns cached result on identical prompt."""
    inner = MockAdjudicator()
    caching = CachingAdjudicator(inner, cache_ttl_s=10)
    
    res1 = caching.judge("identical prompt", [], MLResult(confidence=0.2, category=None))
    assert res1.from_cache is False
    
    res2 = caching.judge("identical prompt", [], MLResult(confidence=0.2, category=None))
    assert res2.from_cache is True
    assert res2.action == res1.action


def test_create_adjudicator_factory():
    """Factory returns correct instance."""
    adj = create_adjudicator(backend="mock")
    assert isinstance(adj, MockAdjudicator)
    
    with pytest.raises(ValueError):
        create_adjudicator(backend="nonexistent")
