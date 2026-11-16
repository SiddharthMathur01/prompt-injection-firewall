"""
service/main.py

FastAPI service wrapper for the LLM Prompt Injection Firewall.

Exposes:
  - POST /scan          — scan a prompt and return the decision
  - POST /v1/chat/completions — OpenAI-compatible proxy endpoint
  - GET  /health        — health check
  - GET  /stats         — rule and request statistics
  - GET  /rules         — list loaded rules

All decisions are logged with the full audit trail (rule id, reasoning,
adjudicator verdict where applicable).
"""
import asyncio
import hmac
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Import project modules
from policy.rule_engine import RuleEngine
from policy.policy_engine import PolicyEngine, MLResult, SourceContext, Decision
from adjudicator.client import create_adjudicator

logger = logging.getLogger("pif")

# Firewall authentication is separate from the upstream provider credential.
# Callers send this secret in X-API-Key; Authorization remains available for
# the OpenAI-compatible proxy to forward to its upstream provider.
firewall_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_firewall_api_key(api_key: Optional[str] = Security(firewall_api_key_header)) -> None:
    expected = os.environ.get("PIF_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Firewall API key is not configured")
    if not api_key or not hmac.compare_digest(api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing firewall API key")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    """Direct scan request."""
    text: str = Field(..., description="Prompt text to scan")
    source_id: str = Field(default="anonymous", description="Source identifier for behavioral tracking")


class ScanResponse(BaseModel):
    """Scan result with full audit trail."""
    action: str
    rule_id: str
    rule_name: str
    reason: str
    tier1_matches: list[dict]
    tier2_confidence: float
    tier2_category: Optional[str]
    adjudicator_verdict: Optional[dict] = None
    scan_time_ms: float


class StatsResponse(BaseModel):
    """Service statistics."""
    total_requests: int
    total_blocked: int
    total_allowed: int
    total_flagged: int
    total_escalated: int
    rules_loaded: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

class FirewallState:
    """Holds all initialized components and counters."""
    def __init__(self):
        self.rule_engine: Optional[RuleEngine] = None
        self.policy_engine: Optional[PolicyEngine] = None
        self.classifier = None  # Optional: PromptClassifier
        self.config: dict = {}
        self.start_time: float = time.time()
        self.request_count: int = 0
        self.block_count: int = 0
        self.allow_count: int = 0
        self.flag_count: int = 0
        self.escalate_count: int = 0
        # Behavioral tracking: source_id -> list of flag timestamps
        self.source_flags: dict[str, list[float]] = {}
        self.audit_log_path: Optional[str] = None


state = FirewallState()


# ---------------------------------------------------------------------------
# Behavioral tracking
# ---------------------------------------------------------------------------

def get_flag_count(source_id: str, window_seconds: int = 300) -> int:
    """Count how many flags a source has triggered in the last N seconds."""
    now = time.time()
    cutoff = now - window_seconds
    flags = state.source_flags.get(source_id, [])
    # Prune old entries
    recent = [t for t in flags if t > cutoff]
    state.source_flags[source_id] = recent
    return len(recent)


def record_flag(source_id: str) -> None:
    """Record a flag event for behavioral tracking."""
    if source_id not in state.source_flags:
        state.source_flags[source_id] = []
    state.source_flags[source_id].append(time.time())


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def log_decision(text: str, decision: Decision, scan_time_ms: float, source_id: str, tier1_matches: list, ml_confidence: float) -> None:
    """Write a structured audit log entry."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "action": decision.action,
        "rule_id": decision.rule_id,
        "rule_name": decision.rule_name,
        "reason": decision.reason,
        "tier1_match_count": len(tier1_matches),
        "tier1_rule_ids": [m.rule_id for m in tier1_matches],
        "tier2_ml_confidence": ml_confidence,
        "scan_time_ms": round(scan_time_ms, 2),
        "prompt_length": len(text),
        "prompt_preview": text[:200] + "..." if len(text) > 200 else text,
    }
    
    if decision.adjudicator_verdict:
        entry["adjudicator"] = {
            "action": decision.adjudicator_verdict.action,
            "confidence": decision.adjudicator_verdict.confidence,
            "reasoning": decision.adjudicator_verdict.reasoning,
            "from_cache": decision.adjudicator_verdict.from_cache,
            "timed_out": decision.adjudicator_verdict.timed_out,
        }
    
    logger.info(json.dumps(entry))
    
    # Write to audit log file if configured
    if state.audit_log_path:
        os.makedirs(os.path.dirname(state.audit_log_path), exist_ok=True)
        with open(state.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Core scan logic
# ---------------------------------------------------------------------------

def scan_prompt(text: str, source_id: str = "anonymous") -> tuple[Decision, list, float, float]:
    """
    Run the full 3-tier pipeline on a prompt.
    
    Returns:
        (decision, tier1_matches, ml_confidence, scan_time_ms)
    """
    start = time.time()
    
    # Tier 1: Regex rules
    tier1_matches = state.rule_engine.scan(text)
    
    # Tier 2: ML classifier
    ml_confidence = 0.0
    ml_category = None
    if state.classifier is not None:
        try:
            ml_result = state.classifier.classify(text)
            ml_confidence = ml_result.confidence
            ml_category = ml_result.category
        except Exception as e:
            logger.warning(f"ML classifier error: {e}")
    
    # Behavioral tracking
    flag_count = get_flag_count(source_id)
    source = SourceContext(identity=source_id, flag_count_last_5m=flag_count)
    ml_input = MLResult(confidence=ml_confidence, category=ml_category)
    
    # Policy evaluation
    decision = state.policy_engine.evaluate(text, tier1_matches, ml_input, source)
    
    # Update stats
    state.request_count += 1
    if decision.action == "block":
        state.block_count += 1
        record_flag(source_id)
    elif decision.action == "allow":
        state.allow_count += 1
    elif decision.action == "flag_for_review":
        state.flag_count += 1
        record_flag(source_id)
    elif decision.action == "allow_and_log":
        state.allow_count += 1
    
    scan_time_ms = (time.time() - start) * 1000
    
    # Audit log
    log_decision(text, decision, scan_time_ms, source_id, tier1_matches, ml_confidence)
    
    return decision, tier1_matches, ml_confidence, scan_time_ms


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    config = load_config()
    state.config = config
    
    # Initialize rule engine
    rules_dir = config.get("rules", {}).get("directory", "rules")
    state.rule_engine = RuleEngine(rules_dir)
    logger.info(f"Loaded {state.rule_engine.rule_count} rules from {rules_dir}")
    
    # Initialize adjudicator
    adj_config = config.get("adjudicator", {})
    adj_backend = adj_config.get("backend", "mock")
    adjudicator = create_adjudicator(backend=adj_backend)
    
    # Initialize policy engine
    policy_rules_path = os.path.join("policy", "policy_rules.yaml")
    state.policy_engine = PolicyEngine(
        rules_path=policy_rules_path,
        adjudicator=adjudicator,
    )
    logger.info("Policy engine initialized")
    
    # Initialize ML classifier (optional — runs without it if model not available)
    classifier_config = config.get("classifier", {})
    model_path = classifier_config.get("model_path", "ml/deberta-prompt-injection/final")
    if os.path.exists(model_path):
        try:
            from classifier.inference import PromptClassifier
            state.classifier = PromptClassifier(
                model_path=model_path,
                device=classifier_config.get("device", "cpu"),
                max_length=classifier_config.get("max_length", 256),
            )
            logger.info(f"ML classifier loaded from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load ML classifier: {e}. Continuing without Tier 2.")
    else:
        logger.info(f"ML model not found at {model_path}. Running without Tier 2 classifier.")
    
    # Audit log path
    state.audit_log_path = config.get("logging", {}).get("audit_log_file")
    
    state.start_time = time.time()
    logger.info("Prompt Injection Firewall started")
    
    yield
    
    logger.info("Prompt Injection Firewall shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM Prompt Injection Firewall",
    description="Request-level security layer for LLM prompt injection detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rules_loaded": state.rule_engine.rule_count if state.rule_engine else 0,
        "classifier_loaded": state.classifier is not None,
        "uptime_seconds": round(time.time() - state.start_time, 1),
    }


@app.post("/scan", response_model=ScanResponse, dependencies=[Depends(require_firewall_api_key)])
async def scan_endpoint(req: ScanRequest):
    """Scan a prompt through the full firewall pipeline."""
    decision, tier1_matches, ml_confidence, scan_time_ms = scan_prompt(
        req.text, req.source_id
    )
    
    adj_dict = None
    if decision.adjudicator_verdict:
        adj_dict = {
            "action": decision.adjudicator_verdict.action,
            "confidence": decision.adjudicator_verdict.confidence,
            "reasoning": decision.adjudicator_verdict.reasoning,
            "from_cache": decision.adjudicator_verdict.from_cache,
        }
    
    return ScanResponse(
        action=decision.action,
        rule_id=decision.rule_id,
        rule_name=decision.rule_name,
        reason=decision.reason,
        tier1_matches=[
            {
                "rule_id": m.rule_id,
                "name": m.name,
                "category": m.category,
                "severity": m.severity,
            }
            for m in tier1_matches
        ],
        tier2_confidence=ml_confidence,
        tier2_category=None,
        adjudicator_verdict=adj_dict,
        scan_time_ms=round(scan_time_ms, 2),
    )


@app.post("/v1/chat/completions", dependencies=[Depends(require_firewall_api_key)])
async def openai_proxy(request: Request):
    """
    OpenAI-compatible proxy endpoint.
    
    Intercepts chat completion requests, scans all user messages through
    the firewall, and either forwards to the upstream LLM or blocks.
    """
    body = await request.json()
    messages = body.get("messages", [])
    
    # Extract source identity from headers or use default
    source_id = request.headers.get("X-Source-ID", "anonymous")
    
    # Scan all user messages
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                decision, _, _, scan_time_ms = scan_prompt(content, source_id)
                
                if decision.action == "block":
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": {
                                "message": "Request blocked by prompt injection firewall",
                                "type": "firewall_block",
                                "code": "prompt_injection_detected",
                                "rule_id": decision.rule_id,
                                "rule_name": decision.rule_name,
                                "reason": decision.reason,
                                "scan_time_ms": round(scan_time_ms, 2),
                            }
                        },
                    )
    
    # Forward to upstream LLM
    upstream_url = state.config.get("upstream", {}).get("openai", {}).get(
        "base_url", "https://api.openai.com/v1"
    )
    
    # Pass through the Authorization header
    headers = {
        "Content-Type": "application/json",
        "Authorization": request.headers.get("Authorization", ""),
    }
    
    async with httpx.AsyncClient() as client:
        upstream_response = await client.post(
            f"{upstream_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=60.0,
        )
    
    return JSONResponse(
        status_code=upstream_response.status_code,
        content=upstream_response.json(),
    )


@app.get("/stats", response_model=StatsResponse, dependencies=[Depends(require_firewall_api_key)])
async def stats_endpoint():
    """Return service statistics."""
    return StatsResponse(
        total_requests=state.request_count,
        total_blocked=state.block_count,
        total_allowed=state.allow_count,
        total_flagged=state.flag_count,
        total_escalated=state.escalate_count,
        rules_loaded=state.rule_engine.rule_count if state.rule_engine else 0,
        uptime_seconds=round(time.time() - state.start_time, 1),
    )


@app.get("/rules", dependencies=[Depends(require_firewall_api_key)])
async def rules_endpoint():
    """List all loaded rules."""
    if not state.rule_engine:
        return {"rules": []}
    
    return {
        "total": state.rule_engine.rule_count,
        "categories": sorted(state.rule_engine.get_categories()),
        "rules": [
            {
                "id": r.id,
                "name": r.name,
                "category": r.category,
                "severity": r.severity,
                "enabled": r.enabled,
                "source_file": r.source_file,
                "tags": r.tags,
            }
            for r in state.rule_engine.rules
        ],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_entry():
    """CLI entry point for the firewall service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Prompt Injection Firewall")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--workers", type=int, default=1, help="Worker count")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    
    uvicorn.run(
        "service.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    cli_entry()
