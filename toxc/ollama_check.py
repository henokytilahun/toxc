"""
Pass 2 contextual analysis via Ollama (local LLM).

Detoxify scores sentences in isolation — it has no concept of sarcasm,
sports metaphors, hyperbole, or compliments. This module runs every
sentence that scored above threshold through a local LLM with its
surrounding context to determine genuine harm vs. false positive,
and generates ad-safe rewrites that preserve the creator's voice.

Requires Ollama running locally:
  brew install ollama          # or https://ollama.com
  ollama pull llama3.2         # recommended model
  ollama serve                 # if not already running as a service
"""

import json
import re


_SYSTEM_PROMPT = (
    "You are a content policy expert helping YouTube creators identify genuine "
    "toxicity vs. hyperbolic speech, sarcasm, compliments, and cultural expressions.\n\n"
    "You will receive a sentence with surrounding context and its raw Detoxify toxicity score. "
    "Determine whether the sentence represents genuinely harmful content (hate speech, real "
    "threats, actual insults) or a false positive (sports metaphors, hype language, sarcasm, "
    "self-directed frustration, compliments using hyperbolic words).\n\n"
    "If it is a false positive, provide two ad-safe rewrites that preserve the creator's "
    "exact voice, energy, and intent — do NOT sanitize into corporate language.\n\n"
    "Reply ONLY with valid JSON. No text outside the JSON object."
)

_USER_TEMPLATE = """\
Context before: "{before}"
Sentence to evaluate: "{sentence}"
Context after: "{after}"
Raw Detoxify toxicity score: {score:.2f}

Reply with this JSON and nothing else:
{{
  "genuine_harm": true or false,
  "intent": "one of: compliment | hyperbole | sarcasm | humor | criticism | neutral | harmful | threat | hate_speech",
  "adjusted_score": <float 0.0-1.0, your honest reassessment>,
  "reason": "<one sentence — why this is or is not genuinely harmful>",
  "safe_rewrite": "<ad-safe version preserving creator's voice — only if genuine_harm is false>",
  "alt_rewrite": "<second option with slightly different energy — only if genuine_harm is false>"
}}"""


def is_available(model: str = "llama3.2") -> tuple[bool, str]:
    """
    Check if Ollama is reachable and the requested model (or a fallback) is pulled.

    Returns:
        (True, resolved_model_name)  if ready
        (False, error_message)       if not available
    """
    try:
        import ollama
    except ImportError:
        return False, (
            "ollama package not installed. Fix with:\n"
            "  pip install ollama"
        )

    try:
        model_list = ollama.list()
        pulled = [m.model.split(":")[0] for m in model_list.models]
    except Exception as e:
        return False, (
            f"Ollama not running (connection refused).\n"
            f"Fix with:  ollama serve\n"
            f"Detail: {e}"
        )

    if not pulled:
        return False, (
            "Ollama is running but no models are pulled.\n"
            "Fix with:  ollama pull llama3.2"
        )

    # Exact match first
    target = model.split(":")[0]
    if target in pulled:
        return True, model

    # Fallback preference order
    for fallback in ["llama3.2", "llama3", "mistral", "gemma2", "phi3"]:
        if fallback in pulled:
            return True, fallback

    # Use whatever is available
    return True, pulled[0]


def contextual_check(
    sentence: str,
    context_before: str,
    context_after: str,
    raw_score: float,
    model: str = "llama3.2",
) -> dict:
    """
    Run a single sentence through Ollama for contextual false-positive detection.

    Returns a dict with keys:
        genuine_harm   bool
        intent         str
        adjusted_score float
        reason         str
        safe_rewrite   str  (only if not genuine_harm)
        alt_rewrite    str  (only if not genuine_harm)
        error          bool (only on parse/connection failure)
    """
    import ollama

    prompt = _USER_TEMPLATE.format(
        before=(context_before or "")[:200],
        sentence=sentence[:300],
        after=(context_after or "")[:200],
        score=raw_score,
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1},  # low temp for determinism
        )
        raw = response["message"]["content"].strip()

        # Strip markdown code fences if the model wraps its JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)

        result = json.loads(raw)

        # Sanitise / clamp fields
        result["adjusted_score"] = max(
            0.0, min(1.0, float(result.get("adjusted_score", raw_score)))
        )
        result["genuine_harm"] = bool(result.get("genuine_harm", True))

        # Remove rewrite fields when not applicable
        if result["genuine_harm"]:
            result.pop("safe_rewrite", None)
            result.pop("alt_rewrite",  None)

        return result

    except json.JSONDecodeError:
        return {
            "genuine_harm":    True,
            "intent":          "unknown",
            "adjusted_score":  raw_score,
            "reason":          "JSON parse error — treating as flagged to be safe.",
            "error":           True,
        }
    except Exception as e:
        return {
            "genuine_harm":    True,
            "intent":          "unknown",
            "adjusted_score":  raw_score,
            "reason":          str(e),
            "error":           True,
        }


def contextual_check_batch(
    sentences: list[dict],
    model: str = "llama3.2",
    threshold: float = 0.35,
    on_progress=None,
) -> list[dict]:
    """
    Run contextual checks on all sentences at or above `threshold`.
    Adds a ``context_check`` key to each qualifying sentence in-place.

    Args:
        sentences:    list of sentence dicts (must have "toxicity" and "text")
        model:        Ollama model name
        threshold:    minimum toxicity score to check (default 0.35)
        on_progress:  optional callable(checked: int, total: int) for progress updates

    Returns:
        The same ``sentences`` list, mutated in-place.
    """
    candidates = [
        (i, s) for i, s in enumerate(sentences)
        if s.get("toxicity", 0) >= threshold
    ]
    total = len(candidates)

    for done, (i, s) in enumerate(candidates):
        before = sentences[i - 1]["text"] if i > 0 else ""
        after  = sentences[i + 1]["text"] if i < len(sentences) - 1 else ""

        check = contextual_check(s["text"], before, after, s["toxicity"], model)
        sentences[i]["context_check"] = check

        if on_progress:
            on_progress(done + 1, total)

    return sentences
