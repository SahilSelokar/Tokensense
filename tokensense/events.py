from dataclasses import dataclass
from typing import Optional, List

@dataclass
class CallEvent:
    ts: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[List[str]] = None
    routed_tier: Optional[str] = None
    error: Optional[str] = None
    
    # Internal fields for prompt/response if opt-in is enabled.
    # Note: the docs say these are never captured by default, but
    # the outputs might need to receive them if log_prompts=True.
    prompt: Optional[str] = None
    response: Optional[str] = None
    
    def to_dict(self) -> dict:
        data = {
            "ts": self.ts,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "tags": self.tags,
            "routed_tier": self.routed_tier,
            "error": self.error,
        }
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.response is not None:
            data["response"] = self.response
        return data
