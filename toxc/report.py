import json
import datetime
from pathlib import Path

DIMS = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]

# Default thresholds (overridden per category by profile.get_thresholds)
_DEFAULT_THRESHOLDS = {
    "high_tox": 0.60, "medium_tox": 0.28,
    "identity": 0.25, "threat": 0.25, "obscene": 0.30, "flagged_rate": 0.15,
}


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
    profile: dict | None = None,
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

    data = {
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
    data["monetization"] = monetization_risk(data, profile=profile)

    if profile:
        from toxc.profile import channel_context
        risk_level = data["monetization"]["risk_level"]
        data["channel"] = channel_context(profile, risk_level)

    return data


def monetization_risk(data: dict, profile: dict | None = None) -> dict:
    """
    Map toxicity signals onto YouTube's advertiser-friendliness criteria.
    Uses category-aware thresholds when a channel profile is provided.
    Returns a risk assessment dict that gets injected into the HTML report.
    """
    from toxc.profile import get_thresholds
    T = get_thresholds(profile) if profile else _DEFAULT_THRESHOLDS

    O = data["overall"]
    sentences = data["sentences"]
    dims = O.get("dimensions", {})
    breakdown = O.get("score_breakdown", {})

    tox          = O["toxicity"]
    identity     = dims.get("identity_attack", 0)
    threat       = dims.get("threat", 0)
    obscene      = dims.get("obscene", 0)
    flagged_rate = breakdown.get("rate", 0)

    # First-7-seconds rule — historically YouTube's strictest window
    first7_hits = [s for s in sentences if s.get("start", 99) < 7 and s["toxicity"] >= 0.35]

    # Build individual risk signals
    signals = []
    if identity >= T["identity"]:
        signals.append({"signal": "hate_speech",  "severity": "high",   "label": "Hate speech / slurs detected",         "value": round(identity, 3)})
    if threat >= T["threat"]:
        signals.append({"signal": "threats",      "severity": "high",   "label": "Threats / violent language",            "value": round(threat, 3)})
    if obscene >= T["obscene"]:
        signals.append({"signal": "profanity",    "severity": "high",   "label": "Heavy profanity",                       "value": round(obscene, 3)})
    elif obscene >= T["obscene"] * 0.4:
        signals.append({"signal": "profanity",    "severity": "medium", "label": "Mild profanity",                        "value": round(obscene, 3)})
    if flagged_rate >= T["flagged_rate"]:
        signals.append({"signal": "density",      "severity": "high",   "label": f"{flagged_rate:.0%} of sentences flagged — focal-point risk", "value": round(flagged_rate, 3)})
    elif flagged_rate >= T["flagged_rate"] * 0.33:
        signals.append({"signal": "density",      "severity": "medium", "label": f"{flagged_rate:.0%} of sentences flagged", "value": round(flagged_rate, 3)})
    if first7_hits:
        signals.append({"signal": "first7",       "severity": "medium", "label": "Flagged content in first 7 seconds",   "value": round(first7_hits[0]["toxicity"], 3)})

    # Category context signal
    if profile:
        cat = profile.get("category", "other").lower()
        cat_label = cat.title()
        if cat == "kids":
            signals.insert(0, {"signal": "category", "severity": "high", "label": f"{cat_label} channel — strictest scrutiny applies", "value": 0})
        elif cat == "education":
            signals.insert(0, {"signal": "category", "severity": "medium", "label": f"{cat_label} channel — higher content standards", "value": 0})

    has_high   = any(s["severity"] == "high"   for s in signals)
    has_medium = any(s["severity"] == "medium" for s in signals)

    if tox >= T["high_tox"] or has_high:
        risk_level = "HIGH"
        ad_verdict = "No ads likely"
        ad_detail  = "Content will likely be demonetized. Edit flagged moments before uploading."
    elif tox >= T["medium_tox"] or has_medium:
        risk_level = "MEDIUM"
        ad_verdict = "Limited ads"
        ad_detail  = "Some ads may run. Review flagged moments for maximum CPM."
    else:
        risk_level = "LOW"
        ad_verdict = "Full monetization likely"
        ad_detail  = "Content appears advertiser-friendly."

    # Actionable recommendations
    recs = []
    if risk_level == "LOW":
        recs.append({"type": "pass", "text": "Safe to upload with full monetization"})
    if first7_hits:
        recs.append({"type": "warning", "text": "Move or remove flagged content from first 7 seconds"})
    top_edits = sorted([s for s in sentences if s["toxicity"] >= 0.35], key=lambda s: s["toxicity"], reverse=True)[:3]
    for s in top_edits:
        m, sec = int(s["start"] // 60), int(s["start"] % 60)
        recs.append({"type": "edit", "text": f"Consider editing {m}:{sec:02d} to protect monetization",
                     "snippet": s["text"][:70], "timestamp": s["start"]})
    if not recs:
        recs.append({"type": "pass", "text": "No specific edits needed"})

    return {
        "risk_level":       risk_level,
        "ad_verdict":       ad_verdict,
        "ad_detail":        ad_detail,
        "first7_flagged":   bool(first7_hits),
        "first7_sentences": first7_hits[:3],
        "signals":          signals,
        "recommendations":  recs,
    }


def write_html(data: dict, output_path: str):
    template_path = Path(__file__).parent / "template.html"
    template = template_path.read_text()
    report_js = f"const REPORT = {json.dumps(data, indent=2)};"
    html = template.replace("const REPORT = null;", report_js)
    Path(output_path).write_text(html)
