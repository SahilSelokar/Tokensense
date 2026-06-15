import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from tokensense.cost import estimate_cost, get_context_window

logger = logging.getLogger("tokensense.router")

def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Rough estimation of tokens in messages. Falls back to character length / 4.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    return total_chars // 4

@dataclass
class Rule:
    if_context_tokens_gt: Optional[int] = None
    if_task: Optional[str] = None
    if_error_code: Optional[int] = None
    if_estimated_cost_gt: Optional[float] = None
    
    deny_tiers: List[str] = field(default_factory=list)
    pin_tier: Optional[str] = None
    prefer_tier: Optional[str] = None
    escalate: bool = False
    
    def matches(self, context_tokens: int, task_hint: Optional[str], last_error_code: Optional[int], estimated_cost: Optional[float]) -> bool:
        # All conditions that are set must match (AND logic)
        if self.if_context_tokens_gt is not None and context_tokens <= self.if_context_tokens_gt:
            return False
        if self.if_task is not None and task_hint != self.if_task:
            return False
        if self.if_error_code is not None and last_error_code != self.if_error_code:
            return False
        if self.if_estimated_cost_gt is not None and (estimated_cost is None or estimated_cost <= self.if_estimated_cost_gt):
            return False
        
        # If no conditions are set, it doesn't match
        if self.if_context_tokens_gt is None and self.if_task is None and self.if_error_code is None and self.if_estimated_cost_gt is None:
            return False
            
        return True

@dataclass
class RoutingDecision:
    model: str
    tier: str
    reason: str
    estimated_cost_usd: float
    denied_tiers: List[str]

class Router:
    def __init__(
        self,
        tiers: Dict[str, List[str]],
        rules: Optional[List[Rule]] = None,
        on_failure: str = "escalate",
        default_tier: Optional[str] = None,
    ):
        self.tiers = tiers
        self.rules = rules or []
        self.on_failure = on_failure
        self.default_tier = default_tier

    def route(
        self,
        messages: List[Dict[str, Any]],
        task_hint: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        min_tier: Optional[str] = None,
        context_tokens: Optional[int] = None,
        last_error_code: Optional[int] = None,
    ) -> RoutingDecision:
        """
        Selects the best model based on rules, context budget, and cost constraints.
        Does not execute the API call itself.
        """
        if context_tokens is None:
            context_tokens = estimate_tokens(messages)
            
        denied_tiers: Set[str] = set()
        pinned_tier = None
        preferred_tier = None
        escalated = False

        # Pre-calculate a baseline estimated cost (using a medium fallback like gpt-4o-mini if no pinned tier)
        # For simplicity, we can do a quick check against the default/first tier's first model
        baseline_cost = 0.0
        first_tier_name = list(self.tiers.keys())[0] if self.tiers else None
        if first_tier_name and self.tiers[first_tier_name]:
            # Assuming output tokens roughly = 500 for baseline heuristic if we had to guess
            baseline_cost = estimate_cost(self.tiers[first_tier_name][0], context_tokens, 500)

        # 1. Evaluate rules
        for rule in self.rules:
            if rule.matches(context_tokens, task_hint, last_error_code, baseline_cost):
                if rule.deny_tiers:
                    denied_tiers.update(rule.deny_tiers)
                if rule.pin_tier:
                    pinned_tier = rule.pin_tier
                if rule.prefer_tier:
                    preferred_tier = rule.prefer_tier
                if rule.escalate:
                    escalated = True
        
        # Determine base ordered tiers (preserve definition order)
        ordered_tiers = list(self.tiers.keys())
        
        # Handle escalation manually in the logic if escalate rule matched
        if escalated:
            # We would drop the "preferred" or lowest tiers in a real scenario,
            # but to keep it simple, we just pretend the smallest tier is denied.
            if ordered_tiers and ordered_tiers[0] not in denied_tiers:
                denied_tiers.add(ordered_tiers[0])

        # Filter by min_tier if provided
        if min_tier and min_tier in ordered_tiers:
            min_tier_idx = ordered_tiers.index(min_tier)
            for t in ordered_tiers[:min_tier_idx]:
                denied_tiers.add(t)

        # Handle explicit pin
        if pinned_tier and pinned_tier in self.tiers and pinned_tier not in denied_tiers:
            eligible_tiers = [pinned_tier]
            reason = f"pinned to {pinned_tier} by rule"
        else:
            # Handle prefer tier
            eligible_tiers = []
            if preferred_tier and preferred_tier in self.tiers and preferred_tier not in denied_tiers:
                eligible_tiers.append(preferred_tier)
                reason = f"preferred {preferred_tier} by rule"
            else:
                reason = "default tier evaluation"
                
            for t in ordered_tiers:
                if t not in denied_tiers and t not in eligible_tiers:
                    eligible_tiers.append(t)
                    
        if not eligible_tiers:
            raise RuntimeError("No eligible tiers found after applying routing rules and constraints.")

        # 2. Iterate through eligible models and apply context budget and max_cost
        for tier in eligible_tiers:
            models = self.tiers[tier]
            for model in models:
                # Check context budget (10% safety margin)
                window = get_context_window(model)
                if window is not None:
                    if context_tokens > window * 0.9:
                        denied_tiers.add(tier)
                        continue # Skip to next model/tier

                # Check max cost
                estimated = estimate_cost(model, context_tokens, 500) # Assumes 500 output tokens for estimate
                if max_cost_usd is not None and estimated > max_cost_usd:
                    continue
                    
                # We found our winner
                if "pinned" not in reason and "preferred" not in reason:
                    reason = f"selected tier {tier} within constraints"
                    
                return RoutingDecision(
                    model=model,
                    tier=tier,
                    reason=reason,
                    estimated_cost_usd=estimated,
                    denied_tiers=list(denied_tiers)
                )

        raise RuntimeError("All routed models failed constraints (context budget or cost).")
