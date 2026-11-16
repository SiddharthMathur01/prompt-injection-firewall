"""
policy/rule_engine.py

Tier 1 — Regex rule engine.

Loads YAML rule files, validates them against _schema.yaml, and matches
prompt text against all enabled rules. Designed for sub-millisecond
latency on the hot path.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# Severity integer -> string mapping (matches _schema.yaml)
SEVERITY_MAP = {1: "low", 2: "medium", 3: "high", 4: "critical"}


@dataclass
class RuleMatch:
    """A single rule that matched the input text."""
    rule_id: str
    name: str
    category: str
    severity: str           # "low" | "medium" | "high" | "critical"
    severity_int: int       # 1-4
    description: str
    source_file: str        # which YAML file this rule came from
    pattern: str            # the regex pattern that matched
    tags: list[str] = field(default_factory=list)


@dataclass
class CompiledRule:
    """A rule with its regex pre-compiled for fast matching."""
    id: str
    name: str
    category: str
    severity: str
    severity_int: int
    description: str
    pattern_str: str
    pattern: re.Pattern
    source_file: str
    enabled: bool
    tags: list[str] = field(default_factory=list)
    case_sensitive: bool = False


class SchemaValidationError(Exception):
    """Raised when a rule file fails schema validation."""
    pass


class RuleEngine:
    """
    Loads regex rules from YAML files and matches them against text.
    
    All rules are compiled once at startup. Matching is a simple linear
    scan — with ~100 rules this runs in well under a millisecond.
    
    Usage:
        engine = RuleEngine("rules")
        matches = engine.scan("ignore all previous instructions")
        for m in matches:
            print(f"{m.rule_id}: {m.name} ({m.severity})")
    """
    
    REQUIRED_FIELDS = {"id", "name", "description", "category", "severity", "pattern", "enabled"}
    
    def __init__(self, rules_dir: str = "rules", validate_schema: bool = True):
        self.rules_dir = Path(rules_dir)
        self.rules: list[CompiledRule] = []
        self._load_all_rules(validate_schema)
    
    def _load_all_rules(self, validate: bool) -> None:
        """Load and compile all enabled rules from every YAML file in the rules directory."""
        if not self.rules_dir.exists():
            raise FileNotFoundError(f"Rules directory not found: {self.rules_dir}")
        
        for rule_file in sorted(self.rules_dir.glob("*.yaml")):
            if rule_file.name.startswith("_"):
                continue  # skip _schema.yaml and other meta files
            
            with rule_file.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            
            for rule_def in data.get("rules", []):
                if validate:
                    self._validate_rule(rule_def, rule_file.name)
                
                if not rule_def.get("enabled", True):
                    continue
                
                pattern_str = rule_def["pattern"]
                case_sensitive = rule_def.get("case_sensitive", False)
                
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    compiled = re.compile(pattern_str, flags)
                except re.error as e:
                    raise SchemaValidationError(
                        f"Invalid regex in rule {rule_def['id']} from {rule_file.name}: {e}"
                    )
                
                severity_int = int(rule_def["severity"])
                severity_str = SEVERITY_MAP.get(severity_int, "unknown")
                
                self.rules.append(CompiledRule(
                    id=rule_def["id"],
                    name=rule_def["name"],
                    category=rule_def["category"],
                    severity=severity_str,
                    severity_int=severity_int,
                    description=rule_def.get("description", ""),
                    pattern_str=pattern_str,
                    pattern=compiled,
                    source_file=rule_file.name,
                    enabled=True,
                    tags=rule_def.get("tags", []),
                    case_sensitive=case_sensitive,
                ))
    
    def _validate_rule(self, rule: dict, source: str) -> None:
        """Validate a rule definition against the schema."""
        missing = self.REQUIRED_FIELDS - set(rule.keys())
        if missing:
            raise SchemaValidationError(
                f"Rule in {source} missing required fields: {missing}. "
                f"Rule data: {rule.get('id', 'UNKNOWN')}"
            )
        
        severity = rule.get("severity")
        if severity not in (1, 2, 3, 4):
            raise SchemaValidationError(
                f"Rule {rule['id']} in {source} has invalid severity: {severity}. "
                f"Must be 1-4."
            )
    
    def scan(self, text: str) -> list[RuleMatch]:
        """
        Scan text against all loaded rules. Returns all matches.
        
        Args:
            text: The prompt text to scan.
            
        Returns:
            List of RuleMatch objects for every rule that matched.
        """
        matches = []
        for rule in self.rules:
            if rule.pattern.search(text):
                matches.append(RuleMatch(
                    rule_id=rule.id,
                    name=rule.name,
                    category=rule.category,
                    severity=rule.severity,
                    severity_int=rule.severity_int,
                    description=rule.description,
                    source_file=rule.source_file,
                    pattern=rule.pattern_str,
                    tags=rule.tags,
                ))
        return matches
    
    @property
    def rule_count(self) -> int:
        """Number of loaded rules."""
        return len(self.rules)
    
    def get_rules_by_category(self, category: str) -> list[CompiledRule]:
        """Return all loaded rules for a given category."""
        return [r for r in self.rules if r.category == category]
    
    def get_categories(self) -> set[str]:
        """Return all unique categories across loaded rules."""
        return {r.category for r in self.rules}
