
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    import tiktoken
except Exception:
    tiktoken = None

Text = str
Message = Dict[str, Any]  # {"role": "user"|"assistant"|"system"|"tool", "content": str}
Messages = List[Message]

# Pricing table: USD per 1K tokens
# Fill with your current prices. You can override at call time.
DEFAULT_PRICING = {
    # "model": (input_per_1k, output_per_1k)
    "gpt-4o-mini": (0.150, 0.600),
    "gpt-4o": (5.00, 15.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}

def _get_encoder(model: str):
    if not tiktoken:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to cl100k_base which covers most modern chat models
        return tiktoken.get_encoding("cl100k_base")

def _count_tokens_text(text: str, model: str) -> int:
    enc = _get_encoder(model)
    if enc:
        return len(enc.encode(text))
    # Fallback heuristic: ~4 chars/token
    return max(1, len(text) // 4)

def _render_messages(messages: Messages) -> str:
    # Simple serialization. Good enough for counting.
    # Example:
    # system: You are...
    # user: Hi
    # assistant: Hello
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # If content is a list (tool messages), join text parts
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def count_prompt_tokens(
    prompt: Union[Text, Messages],
    model: str,
) -> int:
    if isinstance(prompt, str):
        return _count_tokens_text(prompt, model)
    # messages list
    rendered = _render_messages(prompt)
    return _count_tokens_text(rendered, model)

def count_completion_tokens(
    completion: Text,
    model: str,
) -> int:
    return _count_tokens_text(completion, model)

def compute_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Optional[Dict[str, Tuple[float, float]]] = None,
) -> float:
    table = pricing or DEFAULT_PRICING
    if model not in table:
        raise ValueError(
            f"Missing pricing for model '{model}'. "
            f"Provide it via `pricing` or add to DEFAULT_PRICING."
        )
    in_per_1k, out_per_1k = table[model]
    return (prompt_tokens / 1000.0) * in_per_1k + (completion_tokens / 1000.0) * out_per_1k

def summarize_usage(
    model: str,
    prompt: Union[Text, Messages],
    completion: Text,
    pricing: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    ptok = count_prompt_tokens(prompt, model)
    ctok = count_completion_tokens(completion, model)
    cost = compute_cost_usd(model, ptok, ctok, pricing=pricing)
    return {
        "model": model,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "total_tokens": ptok + ctok,
        "cost_usd": round(cost, 6),
    }

if __name__ == "__main__":
    # Demo 1: raw strings
    prompt = "Summarize this: Deep learning transforms data into representations."
    completion = "Deep learning learns layered representations to extract patterns."
    stats = summarize_usage("gpt-4o-mini", prompt, completion)
    print(stats)

    # Demo 2: chat messages
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain LoRA in one paragraph."},
    ]
    completion2 = "LoRA trains small rank adapters while freezing the base model."
    stats2 = summarize_usage("gpt-4o-mini", messages, completion2)
    print(stats2)

    # Demo 3: custom pricing override
    custom = {"gpt-4o-mini": (0.2, 0.8)}
    stats3 = summarize_usage("gpt-4o-mini", prompt, completion, pricing=custom)
    print(stats3)
