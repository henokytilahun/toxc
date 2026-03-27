import json
import datetime
from pathlib import Path

DIMS = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]


def aggregate(
    sentences: list[dict],
    audio_path: str,
    model_size: str,
    duration: float,
) -> dict:
    for i, s in enumerate(sentences):
        s["idx"] = i

    toxicities = [s["toxicity"] for s in sentences]

    lengths = [len(s["text"]) for s in sentences]
    total_len = sum(lengths) or 1
    weighted_tox  = sum(s["toxicity"]  * len(s["text"]) for s in sentences) / total_len
    weighted_sent = sum(s["sentiment"] * len(s["text"]) for s in sentences) / total_len

    if weighted_tox >= 0.7:
        verdict = "Toxic"
    elif weighted_tox >= 0.4:
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
            "toxicity": weighted_tox,
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
