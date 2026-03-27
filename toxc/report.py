import json
import datetime
from pathlib import Path

DIMS = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]


def _composite_toxicity(sentences: list[dict]) -> tuple[float, dict]:
    """
    Peak-aware composite score so sparse-but-severe toxicity isn't buried
    by a majority of clean filler sentences.

    Components (all in [0, 1]):
      density  (25%) — length-weighted mean across all sentences
      peak     (50%) — mean of the top 5 % most toxic sentences
      rate     (25%) — fraction of sentences classified as Toxic (≥ 0.7)

    Returns (composite_score, breakdown_dict).
    """
    n = len(sentences)
    lengths = [len(s["text"]) for s in sentences]
    total_len = sum(lengths) or 1
    toxicities = [s["toxicity"] for s in sentences]

    # 1. Density — same length-weighted mean as before
    density = sum(s["toxicity"] * len(s["text"]) for s in sentences) / total_len

    # 2. Peak severity — mean of top 5 % (minimum 1 sentence)
    top_n = max(1, round(n * 0.05))
    peak = sum(sorted(toxicities, reverse=True)[:top_n]) / top_n

    # 3. Rate — proportion of sentences that are Toxic
    toxic_count = sum(1 for t in toxicities if t >= 0.7)
    rate = toxic_count / n

    composite = 0.25 * density + 0.50 * peak + 0.25 * rate

    breakdown = {
        "density": round(density, 4),
        "peak": round(peak, 4),
        "rate": round(rate, 4),
    }
    return round(min(composite, 1.0), 4), breakdown


def aggregate(
    sentences: list[dict],
    audio_path: str,
    model_size: str,
    duration: float,
) -> dict:
    for i, s in enumerate(sentences):
        s["idx"] = i

    lengths = [len(s["text"]) for s in sentences]
    total_len = sum(lengths) or 1
    toxicities = [s["toxicity"] for s in sentences]

    composite_tox, score_breakdown = _composite_toxicity(sentences)
    weighted_sent = sum(s["sentiment"] * len(s["text"]) for s in sentences) / total_len

    if composite_tox >= 0.7:
        verdict = "Toxic"
    elif composite_tox >= 0.4:
        verdict = "Borderline"
    else:
        verdict = "Clean"

    fast = sentences[0].get("fast", False) if sentences else False

    # Weighted-average each sub-dimension across all sentences
    agg_dims = {}
    if not fast:
        for dim in DIMS:
            agg_dims[dim] = sum(s["dimensions"].get(dim, 0) * len(s["text"]) for s in sentences) / total_len

    top5 = sorted(sentences, key=lambda s: s["toxicity"], reverse=True)[:5]

    peaks = {}
    if not fast:
        for dim in DIMS:
            best = max(sentences, key=lambda s: s["dimensions"].get(dim, 0))
            peaks[dim] = {"score": best["dimensions"].get(dim, 0), "sentence": best}

    return {
        "audio": str(audio_path),
        "model": model_size,
        "duration": duration,
        "analyzed_at": datetime.datetime.now().strftime("%b %d %Y"),
        "overall": {
            "toxicity": composite_tox,
            "score_breakdown": score_breakdown,
            "sentiment": weighted_sent,
            "verdict": verdict,
            "toxic_count": sum(1 for t in toxicities if t >= 0.7),
            "sentence_count": len(sentences),
            "dimensions": agg_dims,
        },
        "sentences": sentences,
        "top_toxic": top5,
        "peaks_by_dim": peaks,
        "fast": fast,
    }


def write_html(data: dict, output_path: str):
    template_path = Path(__file__).parent / "template.html"
    template = template_path.read_text()
    report_js = f"const REPORT = {json.dumps(data, indent=2)};"
    html = template.replace("const REPORT = null;", report_js)
    Path(output_path).write_text(html)
