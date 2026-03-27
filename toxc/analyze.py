import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rich.console import Console

_vader = SentimentIntensityAnalyzer()
_console = Console(stderr=True)


def _vader_fallback_tox(vader: dict) -> dict:
    neg = vader["neg"]
    return {
        "toxicity": min(neg * 1.5, 1.0),
        "severe_toxicity": min(neg * 0.5, 1.0),
        "obscene": min(neg * 0.8, 1.0),
        "threat": min(neg * 0.3, 1.0),
        "insult": min(neg * 1.0, 1.0),
        "identity_attack": min(neg * 0.2, 1.0),
    }


def _build_result(text: str, vader: dict, tox: dict, fast: bool = False) -> dict:
    sentiment_score = vader["compound"]
    toxicity = tox.get("toxicity", 0)

    if toxicity >= 0.7:
        verdict = "Toxic"
        if tox.get("threat", 0) > 0.5:
            detail = "threatening language detected"
        elif tox.get("insult", 0) > 0.5:
            detail = "personal attack detected"
        elif tox.get("identity_attack", 0) > 0.5:
            detail = "identity-based attack detected"
        else:
            detail = "harmful content detected"
    elif toxicity >= 0.4:
        verdict = "Borderline"
        detail = "potentially problematic content"
    else:
        verdict = "Clean"
        detail = "no significant toxicity detected"

    return {
        "text": text,
        "sentiment": sentiment_score,
        "toxicity": toxicity,
        "dimensions": {
            "severe_toxicity": tox.get("severe_toxicity", 0),
            "obscene": tox.get("obscene", 0),
            "threat": tox.get("threat", 0),
            "insult": tox.get("insult", 0),
            "identity_attack": tox.get("identity_attack", 0),
        },
        "verdict": verdict,
        "detail": detail,
        "fast": fast,
    }


def _load_detoxify():
    import transformers
    transformers.logging.set_verbosity_error()
    from detoxify import Detoxify
    with _console.status("[dim]Loading model…[/dim]", spinner="dots"):
        return Detoxify("original")


def analyze(text: str, use_detoxify: bool = True) -> dict:
    vader = _vader.polarity_scores(text)
    if use_detoxify:
        try:
            tox = {k: float(v) for k, v in _load_detoxify().predict(text).items()}
            return _build_result(text, vader, tox, fast=False)
        except Exception:
            pass
    tox = _vader_fallback_tox(vader)
    return _build_result(text, vader, tox, fast=True)


def analyze_batch(texts: list[str], use_detoxify: bool = True) -> list[dict]:
    texts = [t for t in texts if t.strip()]
    model = None
    if use_detoxify:
        try:
            model = _load_detoxify()
        except Exception:
            use_detoxify = False

    results = []
    for text in texts:
        vader = _vader.polarity_scores(text)
        if model is not None:
            tox = {k: float(v) for k, v in model.predict(text).items()}
            results.append(_build_result(text, vader, tox, fast=False))
        else:
            tox = _vader_fallback_tox(vader)
            results.append(_build_result(text, vader, tox, fast=True))
    return results
