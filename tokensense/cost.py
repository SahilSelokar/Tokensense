from typing import Tuple, Optional

# (input_per_1M, output_per_1M)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-8": (15.00, 75.00),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Groq / Llama
    "llama3-8b-8192": (0.05, 0.10),
    "llama3-70b-8192": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
    # Google
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}

CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-opus-4-8": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    # Groq / Llama
    "llama3-8b-8192": 8_192,
    "llama3-70b-8192": 8_192,
    "llama-3.1-8b-instant": 131_072,
    "llama-3.3-70b-versatile": 131_072,
    "mixtral-8x7b-32768": 32_768,
    "gemma2-9b-it": 8_192,
    # Google
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
}

def _fuzzy_match_model(model_name: str, target_dict: dict) -> Optional[str]:
    # Exact match first
    if model_name in target_dict:
        return model_name
        
    # Strip provider prefix if present (e.g. groq/llama3-8b-8192)
    clean_model = model_name
    if "/" in clean_model:
        clean_model = clean_model.split("/", 1)[1]
        if clean_model in target_dict:
            return clean_model
            
    # Fuzzy substring match
    # Match longest base model name that is a prefix
    candidates = []
    for k in target_dict.keys():
        if clean_model.startswith(k):
            candidates.append(k)
    
    if candidates:
        return max(candidates, key=len)
        
    return None

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the estimated USD cost of an LLM invocation.
    """
    matched_key = _fuzzy_match_model(model_name, MODEL_COSTS)
    
    if matched_key:
        input_price, output_price = MODEL_COSTS[matched_key]
    else:
        # Fallback to a standard/average pricing
        input_price, output_price = 0.15, 0.60
        
    input_cost = (input_tokens / 1_000_000.0) * input_price
    output_cost = (output_tokens / 1_000_000.0) * output_price
    return input_cost + output_cost

# Aliasing calculate_cost to estimate_cost to prevent breaking internal modules
# if they still reference calculate_cost. We will update them though.
calculate_cost = estimate_cost

def get_context_window(model_name: str) -> Optional[int]:
    """
    Get the context window (in tokens) for a given model.
    Returns None if unknown.
    """
    matched_key = _fuzzy_match_model(model_name, CONTEXT_WINDOWS)
    if matched_key:
        return CONTEXT_WINDOWS[matched_key]
    return None
