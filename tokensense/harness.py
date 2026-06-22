import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

@dataclass
class ShadowTestFailure:
    index: int
    tier: str
    error: str
    current_output: Optional[str] = None
    candidate_output: Optional[str] = None

@dataclass
class TierStats:
    pass_count: int = 0
    fail_count: int = 0
    total_cost: float = 0.0
    total_latency_ms: int = 0

    @property
    def pass_rate(self) -> float:
        total = self.pass_count + self.fail_count
        return self.pass_count / total if total > 0 else 0.0

    @property
    def avg_cost(self) -> float:
        total = self.pass_count + self.fail_count
        return self.total_cost / total if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        total = self.pass_count + self.fail_count
        return self.total_latency_ms / total if total > 0 else 0.0

@dataclass
class ShadowTestReport:
    stats: Dict[str, TierStats] = field(default_factory=dict)
    failures: List[ShadowTestFailure] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if "candidate" in self.stats:
            return self.stats["candidate"].pass_rate
        return 0.0

    @property
    def cost_delta_pct(self) -> float:
        if "current" in self.stats and "candidate" in self.stats:
            curr_cost = self.stats["current"].avg_cost
            cand_cost = self.stats["candidate"].avg_cost
            if curr_cost > 0:
                return ((cand_cost - curr_cost) / curr_cost) * 100.0
        return 0.0

    @property
    def latency_delta_pct(self) -> float:
        if "current" in self.stats and "candidate" in self.stats:
            curr_lat = self.stats["current"].avg_latency_ms
            cand_lat = self.stats["candidate"].avg_latency_ms
            if curr_lat > 0:
                return ((cand_lat - curr_lat) / curr_lat) * 100.0
        return 0.0

    def summary(self) -> str:
        lines = []
        lines.append("ShadowTest Report")
        lines.append("──────────────────────────────────────")
        lines.append(f"{'Tier':<15} {'Pass Rate':<12} {'Avg Cost':<12} {'Avg Latency':<12}")
        
        for tier, stat in self.stats.items():
            lines.append(
                f"{tier:<15} "
                f"{stat.pass_rate * 100:.0f}%{'':<9} "
                f"${stat.avg_cost:.4f}{'':<5} "
                f"{int(stat.avg_latency_ms)}ms"
            )
            
        lines.append("")
        
        if self.failures:
            fail_indices = [str(f.index) for f in self.failures if f.tier == "candidate"]
            lines.append(f"⚠ {len(fail_indices)} candidate responses failed scoring.")
            if fail_indices:
                lines.append(f"Failed cases: #{', #'.join(fail_indices)} (see report.failures)")
        else:
            lines.append("✅ All candidate responses passed scoring.")
            
        return "\n".join(lines)

    def to_json(self) -> str:
        data = {
            "pass_rate": self.pass_rate,
            "cost_delta_pct": self.cost_delta_pct,
            "latency_delta_pct": self.latency_delta_pct,
            "stats": {
                k: {
                    "pass_rate": v.pass_rate,
                    "avg_cost": v.avg_cost,
                    "avg_latency": v.avg_latency_ms
                } for k, v in self.stats.items()
            },
            "failures": [
                {
                    "index": f.index,
                    "tier": f.tier,
                    "error": f.error,
                    "current_output": f.current_output,
                    "candidate_output": f.candidate_output
                } for f in self.failures
            ]
        }
        return json.dumps(data, indent=2)

    def to_markdown(self) -> str:
        lines = []
        lines.append("## ShadowTest Report")
        lines.append("")
        lines.append("| Tier | Pass Rate | Avg Cost | Avg Latency |")
        lines.append("|---|---|---|---|")
        
        for tier, stat in self.stats.items():
            lines.append(
                f"| {tier} | {stat.pass_rate * 100:.0f}% | ${stat.avg_cost:.4f} | {int(stat.avg_latency_ms)}ms |"
            )
            
        lines.append("")
        if self.failures:
            lines.append("### ⚠ Failures")
            for f in self.failures:
                lines.append(f"**Case #{f.index} ({f.tier})**")
                lines.append(f"- **Error:** {f.error}")
                if f.current_output:
                    lines.append("- **Current Output:**")
                    lines.append(f"```\n{f.current_output}\n```")
                if f.candidate_output:
                    lines.append("- **Candidate Output:**")
                    lines.append(f"```\n{f.candidate_output}\n```")
                lines.append("")
        return "\n".join(lines)

class ShadowTest:
    def __init__(
        self,
        clients: Dict[str, Any],
        prompts: List[Dict[str, Any]],
        scoring: str = "exact-match",
        judge: Optional[Any] = None,
        judge_rubric: Optional[str] = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.85
    ):
        self.clients = clients
        self.prompts = prompts
        self.scoring = scoring
        self.judge = judge
        self.judge_rubric = judge_rubric
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold

    def _extract_text(self, response: Any) -> str:
        """Pull text out of any SDK response shape."""
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        if hasattr(response, "content"):
            if isinstance(response.content, list):
                return "".join([getattr(c, "text", "") for c in response.content])
            return str(response.content)
        return str(response)

    def _score(self, index: int, tier: str, prompt: Dict[str, Any], text: str, reference_text: Optional[str] = None) -> Optional[str]:
        """Score a response. Returns an error string on failure, None on pass."""
        if self.scoring == "exact-match":
            expected = prompt.get("expected_output")
            if expected is not None and text.strip() != expected.strip():
                return "Output did not match expected_output exactly."
        elif self.scoring == "format-check":
            expected_format = prompt.get("expected_format")
            if expected_format == "json":
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    return "Output is not valid JSON."
        elif self.scoring == "similarity":
            if tier != "current" and reference_text:
                if not self.embedding_fn:
                    return "embedding_fn is required for similarity scoring."
                try:
                    emb1 = self.embedding_fn(text)
                    emb2 = self.embedding_fn(reference_text)
                    
                    # Cosine similarity
                    dot = sum(a * b for a, b in zip(emb1, emb2))
                    norm1 = sum(a * a for a in emb1) ** 0.5
                    norm2 = sum(b * b for b in emb2) ** 0.5
                    sim = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                    
                    if sim < self.similarity_threshold:
                        return f"Similarity score {sim:.2f} is below threshold {self.similarity_threshold}."
                except Exception as e:
                    return f"Similarity calculation failed: {str(e)}"
        elif self.scoring == "llm-judge":
            if tier != "current" and reference_text:
                if not self.judge or not self.judge_rubric:
                    return "judge and judge_rubric are required for llm-judge scoring."
                try:
                    judge_prompt = (
                        f"Rubric: {self.judge_rubric}\n"
                        f"Reference Output: {reference_text}\n"
                        f"Candidate Output: {text}\n"
                        "Evaluate if the Candidate Output passes the Rubric compared to the Reference Output. "
                        "Respond with exactly one word: PASS or FAIL."
                    )
                    
                    # Duck-type the judge client
                    if hasattr(self.judge, "messages") and hasattr(self.judge.messages, "create"):
                        resp = self.judge.messages.create(
                            model="claude-3-haiku-20240307",
                            messages=[{"role": "user", "content": judge_prompt}],
                            max_tokens=10
                        )
                    elif hasattr(self.judge, "chat") and hasattr(self.judge.chat.completions, "create"):
                        resp = self.judge.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": judge_prompt}],
                            max_tokens=10
                        )
                    elif callable(self.judge):
                        resp = self.judge(messages=[{"role": "user", "content": judge_prompt}])
                    else:
                        return "Judge client format not recognized."
                        
                    judge_text = self._extract_text(resp).strip().upper()
                    if "FAIL" in judge_text:
                        return "LLM Judge returned FAIL."
                    elif "PASS" not in judge_text:
                        return f"LLM Judge returned unexpected response: {judge_text}"
                except Exception as e:
                    return f"LLM Judge execution failed: {str(e)}"
                    
        return None

    def _run_single(self, index: int, tier: str, client: Any, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Run one prompt against one tier. Scoring happens later in run()."""
        start_time = time.time()
        
        # Use tier-specific model if defined (e.g. model_current, model_candidate)
        model = prompt.get(f"model_{tier}")
        
        # Strip out our metadata keys, pass everything else to the SDK
        payload = {}
        for k, v in prompt.items():
            if k not in ["expected_output", "expected_format"] and not k.startswith("model_"):
                payload[k] = v
                
        if model:
            payload["model"] = model
            
        try:
            # Duck-type the client to figure out which SDK method to call
            if hasattr(client, "messages") and hasattr(client.messages, "create"):
                response = client.messages.create(**payload)
            elif hasattr(client, "chat") and hasattr(client.chat.completions, "create"):
                response = client.chat.completions.create(**payload)
            elif callable(client):
                response = client(**payload)
            else:
                raise ValueError("Client format not recognized. Pass a callable or a standard SDK client.")
                
            latency_ms = int((time.time() - start_time) * 1000)
            text = self._extract_text(response)
            
            # Re-estimate cost from text length (we don't have access to
            # the observe() wrapper's event from here)
            from tokensense.cost import estimate_cost
            from tokensense.router import estimate_tokens
            
            in_tokens = estimate_tokens([{"content": str(payload)}])
            out_tokens = estimate_tokens([{"content": text}])
            cost = estimate_cost(model or tier, in_tokens, out_tokens)
            
            # Scoring happens in run() after all tiers finish (needed for comparative strategies)
            return {
                "index": index,
                "tier": tier,
                "success": True,  # Provisional — run() may flip this
                "error": None,
                "text": text,
                "latency_ms": latency_ms,
                "cost": cost
            }
        except Exception as e:
            return {
                "index": index,
                "tier": tier,
                "success": False,
                "error": str(e),
                "text": None,
                "latency_ms": int((time.time() - start_time) * 1000),
                "cost": 0.0
            }

    def run(self) -> ShadowTestReport:
        report = ShadowTestReport()
        for tier in self.clients.keys():
            report.stats[tier] = TierStats()

        futures = []
        with ThreadPoolExecutor() as executor:
            for i, prompt in enumerate(self.prompts):
                for tier, client in self.clients.items():
                    futures.append(executor.submit(self._run_single, i, tier, client, prompt))
                    
            results = []
            for future in as_completed(futures):
                results.append(future.result())

        # Group by prompt index so we can compare tiers side by side
        grouped = {}
        for r in results:
            idx = r["index"]
            if idx not in grouped:
                grouped[idx] = {}
            grouped[idx][r["tier"]] = r
            
        # Score and tally
        for idx, tiers in grouped.items():
            current_res = tiers.get("current")
            candidate_res = tiers.get("candidate")
            
            for tier, res in tiers.items():
                if res["text"] is not None and res["error"] is None:
                    # Second pass: score now that we have all tiers' outputs
                    ref_text = current_res["text"] if current_res else None
                    error = self._score(idx, tier, self.prompts[idx], res["text"], reference_text=ref_text)
                    if error:
                        res["success"] = False
                        res["error"] = error
            
            for tier, res in tiers.items():
                stat = report.stats[tier]
                if res["success"]:
                    stat.pass_count += 1
                else:
                    stat.fail_count += 1
                    report.failures.append(ShadowTestFailure(
                        index=idx,
                        tier=tier,
                        error=res["error"],
                        current_output=current_res["text"] if current_res else None,
                        candidate_output=candidate_res["text"] if candidate_res else None
                    ))
                    
                stat.total_cost += res["cost"]
                stat.total_latency_ms += res["latency_ms"]

        return report
