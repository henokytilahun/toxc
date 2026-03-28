import json
import datetime
from pathlib import Path

DIMS = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"]

# Default thresholds (overridden per category by profile.get_thresholds)
#
# Signal hierarchy matches YouTube's actual advertiser-friendliness policy:
#
#   identity_attack  — hate speech / slurs. Brands will not appear next to this.
#                      Instant demonetization at very low scores. Lowest threshold.
#   threat           — incitement / violent language. Also instant demonetization.
#   severe_toxicity  — extreme degrading content. Instant demonetization.
#   obscene          — profanity. Usually → limited ads (yellow), NOT full demonetization
#                      unless extremely heavy. Higher threshold, lower severity ceiling.
#   flagged_rate     — density of flagged sentences. High density = pattern, not slip.
#
_DEFAULT_THRESHOLDS = {
    "high_tox":       0.60,
    "medium_tox":     0.28,
    "identity":       0.08,   # hate speech — fires HIGH at very low scores
    "threat":         0.12,   # violent language — also fires HIGH early
    "severe":         0.15,   # extreme content — fires HIGH
    "obscene_high":   0.55,   # heavy profanity → HIGH (full demonetization)
    "obscene_medium": 0.12,   # mild profanity → MEDIUM (limited ads only)
    "flagged_rate":   0.15,
}


def _composite_toxicity(sentences: list[dict], tox_key: str = "toxicity") -> tuple[float, dict]:
    """
    Peak-aware composite score so sparse-but-severe toxicity isn't buried
    by a majority of clean filler sentences.

    Components (all in [0, 1]):
      density  (25%) — length-weighted mean across all sentences
      peak     (50%) — mean of the top 5 % most toxic sentences
      rate     (25%) — fraction of sentences classified as Toxic (≥ 0.7)

    tox_key: which field to read from each sentence dict.
             Use "adjusted_toxicity" for context-check-adjusted scoring.

    Returns (composite_score, breakdown_dict).
    """
    n = len(sentences)
    lengths = [len(s["text"]) for s in sentences]
    total_len = sum(lengths) or 1
    toxicities = [s.get(tox_key, s["toxicity"]) for s in sentences]

    # 1. Density — length-weighted mean
    density = sum(toxicities[i] * lengths[i] for i in range(n)) / total_len

    # 2. Peak severity — mean of top 5 % (minimum 1 sentence)
    top_n = max(1, round(n * 0.05))
    peak = sum(sorted(toxicities, reverse=True)[:top_n]) / top_n

    # 3. Rate — proportion of sentences that are Toxic
    toxic_count = sum(1 for t in toxicities if t >= 0.7)
    rate = toxic_count / n

    composite = 0.25 * density + 0.50 * peak + 0.25 * rate

    breakdown = {
        "density": round(density, 4),
        "peak":    round(peak,    4),
        "rate":    round(rate,    4),
    }
    return round(min(composite, 1.0), 4), breakdown


def aggregate(
    sentences: list[dict],
    audio_path: str,
    model_size: str,
    duration: float,
    profile: dict | None = None,
    ollama_model: str | None = None,
    policy_review_model: str | None = None,
) -> dict:
    for i, s in enumerate(sentences):
        s["idx"] = i

    # ── Pass 2: contextual false-positive check via Ollama ────────────────────
    context_check_ran = False
    context_summary: dict = {"ran": False}

    if ollama_model:
        try:
            import os
            from rich.console import Console as _RC
            _rc = _RC(stderr=True)
            from toxc.ollama_check import contextual_check_batch
            candidates = [s for s in sentences if s.get("toxicity", 0) >= 0.35]
            n_check = len(candidates)
            if n_check:
                with _rc.status(
                    f"[dim]Context-checking {n_check} flagged sentences with {ollama_model}…[/dim]",
                    spinner="dots",
                ):
                    contextual_check_batch(sentences, model=ollama_model, threshold=0.35)
                context_check_ran = True
        except Exception:
            pass  # gracefully skip — Detoxify scores remain intact

    # Apply adjusted toxicity per sentence
    cleared = confirmed = 0
    for s in sentences:
        cc = s.get("context_check")
        if cc and not cc.get("error"):
            s["adjusted_toxicity"] = cc["adjusted_score"]
            if cc.get("genuine_harm"):
                confirmed += 1
            else:
                cleared += 1
        else:
            s["adjusted_toxicity"] = s["toxicity"]

    if context_check_ran:
        context_summary = {
            "ran":       True,
            "model":     ollama_model,
            "checked":   cleared + confirmed,
            "cleared":   cleared,
            "confirmed": confirmed,
        }

    # ── Pass 3: full LLM policy review ───────────────────────────────────────
    policy_review_result: dict = {"ran": False}

    if policy_review_model:
        try:
            from rich.console import Console as _RC2
            _rc2 = _RC2(stderr=True)
            from toxc.policy_review import policy_review as _policy_review
            title = str(audio_path)
            category = (profile or {}).get("category", "other")
            with _rc2.status(
                f"[dim]Running LLM policy review with {policy_review_model}…[/dim]",
                spinner="dots",
            ):
                policy_review_result = _policy_review(
                    sentences,
                    title=title,
                    channel_category=category,
                    model=policy_review_model,
                )
        except Exception:
            pass  # gracefully skip — existing scores remain intact

    # ── Composite scores ──────────────────────────────────────────────────────
    lengths   = [len(s["text"]) for s in sentences]
    total_len = sum(lengths) or 1

    raw_composite,  raw_breakdown  = _composite_toxicity(sentences, "toxicity")
    adj_composite,  adj_breakdown  = _composite_toxicity(sentences, "adjusted_toxicity")

    # Use adjusted composite for the public verdict when context check ran
    composite_tox   = adj_composite
    score_breakdown = adj_breakdown

    weighted_sent = sum(s["sentiment"] * len(s["text"]) for s in sentences) / total_len

    if composite_tox >= 0.7:
        verdict = "Toxic"
    elif composite_tox >= 0.4:
        verdict = "Borderline"
    else:
        verdict = "Clean"

    fast = sentences[0].get("fast", False) if sentences else False

    agg_dims = {}
    if not fast:
        for dim in DIMS:
            agg_dims[dim] = (
                sum(s["dimensions"].get(dim, 0) * len(s["text"]) for s in sentences) / total_len
            )

    # Top-5 by adjusted toxicity so cleared false-positives don't dominate
    top5 = sorted(sentences, key=lambda s: s["adjusted_toxicity"], reverse=True)[:5]

    peaks = {}
    if not fast:
        for dim in DIMS:
            best = max(sentences, key=lambda s: s["dimensions"].get(dim, 0))
            peaks[dim] = {"score": best["dimensions"].get(dim, 0), "sentence": best}

    data = {
        "audio":       str(audio_path),
        "model":       model_size,
        "duration":    duration,
        "analyzed_at": datetime.datetime.now().strftime("%b %d %Y"),
        "overall": {
            "toxicity":       composite_tox,
            "raw_toxicity":   raw_composite,
            "score_breakdown": score_breakdown,
            "sentiment":      weighted_sent,
            "verdict":        verdict,
            "toxic_count":    sum(1 for s in sentences if s["adjusted_toxicity"] >= 0.7),
            "sentence_count": len(sentences),
            "dimensions":     agg_dims,
        },
        "context_check":  context_summary,
        "policy_review":  policy_review_result,
        "sentences":      sentences,
        "top_toxic":     top5,
        "peaks_by_dim":  peaks,
        "fast":          fast,
    }
    # ── Per-speaker breakdown (only when diarization ran) ────────────────────
    speaker_map: dict[str, dict] = {}
    for s in sentences:
        spk = s.get("speaker")
        if not spk:
            continue
        if spk not in speaker_map:
            speaker_map[spk] = {
                "speaker":        spk,
                "sentence_count": 0,
                "tox_sum":        0.0,
                "flagged":        0,
                "peak_tox":       0.0,
                "peak_sentence":  None,
            }
        tox = s.get("adjusted_toxicity", s["toxicity"])
        speaker_map[spk]["sentence_count"] += 1
        speaker_map[spk]["tox_sum"]        += tox
        if tox >= 0.4:
            speaker_map[spk]["flagged"] += 1
        if tox > speaker_map[spk]["peak_tox"]:
            speaker_map[spk]["peak_tox"]      = tox
            speaker_map[spk]["peak_sentence"] = s["text"][:80]

    speakers_list = []
    for spk_data in speaker_map.values():
        n = spk_data["sentence_count"]
        avg = round(spk_data["tox_sum"] / n, 4) if n else 0.0
        spk_data["avg_toxicity"] = avg
        spk_data["peak_tox"]     = round(spk_data["peak_tox"], 4)
        del spk_data["tox_sum"]
        if avg >= 0.4:
            spk_data["verdict"] = "Review"
        elif avg >= 0.15:
            spk_data["verdict"] = "Borderline"
        else:
            spk_data["verdict"] = "Clean"
        speakers_list.append(spk_data)

    # Sort by avg toxicity descending so riskiest speaker is first
    speakers_list.sort(key=lambda x: x["avg_toxicity"], reverse=True)
    data["speakers"] = speakers_list

    data["monetization"] = monetization_risk(data, profile=profile)

    if profile:
        from toxc.profile import channel_context
        risk_level = data["monetization"]["risk_level"]
        data["channel"] = channel_context(profile, risk_level)

        revenue_at_risk = data["channel"].get("financial", {}).get("revenue_at_risk", 0)
        if revenue_at_risk > 0:
            edit_recs = [r for r in data["monetization"]["recommendations"] if r["type"] == "edit"]
            total_tox = sum(r.get("toxicity", 0) for r in edit_recs) or 1
            for rec in edit_recs:
                share = rec.get("toxicity", 0) / total_tox
                rec["saves"] = round(revenue_at_risk * share, 2)

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
    severe       = dims.get("severe_toxicity", 0)
    flagged_rate = breakdown.get("rate", 0)

    # First-7-seconds rule — use adjusted toxicity so cleared false-positives don't fire
    first7_hits = [s for s in sentences if s.get("start", 99) < 7 and s.get("adjusted_toxicity", s["toxicity"]) >= 0.35]

    # Build signals in priority order, reflecting YouTube's actual demonetization hierarchy:
    # identity_attack > threat > severe_toxicity > obscene (high) > obscene (mild) > density
    signals = []

    # Tier 1 — instant demonetization signals (brands will not appear next to these)
    if identity >= T["identity"]:
        signals.append({"signal": "hate_speech", "severity": "high",
                        "label": "Hate speech / identity attacks — instant demonetization risk",
                        "value": round(identity, 3)})

    if threat >= T["threat"]:
        signals.append({"signal": "threats", "severity": "high",
                        "label": "Threats or incitement to violence",
                        "value": round(threat, 3)})

    if severe >= T["severe"]:
        signals.append({"signal": "severe", "severity": "high",
                        "label": "Severely toxic content",
                        "value": round(severe, 3)})

    # Tier 2 — profanity (usually limited ads, not full demonetization)
    if obscene >= T["obscene_high"]:
        signals.append({"signal": "profanity", "severity": "high",
                        "label": "Heavy profanity throughout — full demonetization likely",
                        "value": round(obscene, 3)})
    elif obscene >= T["obscene_medium"]:
        signals.append({"signal": "profanity", "severity": "medium",
                        "label": "Profanity present — limited ads (yellow icon likely)",
                        "value": round(obscene, 3)})

    # Tier 3 — density / pattern signals
    if flagged_rate >= T["flagged_rate"]:
        signals.append({"signal": "density", "severity": "high",
                        "label": f"{flagged_rate:.0%} of sentences flagged — pattern risk",
                        "value": round(flagged_rate, 3)})
    elif flagged_rate >= T["flagged_rate"] * 0.33:
        signals.append({"signal": "density", "severity": "medium",
                        "label": f"{flagged_rate:.0%} of sentences flagged",
                        "value": round(flagged_rate, 3)})

    if first7_hits:
        signals.append({"signal": "first7", "severity": "medium",
                        "label": "Flagged content in first 7 seconds",
                        "value": round(first7_hits[0]["toxicity"], 3)})

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

    # Consequence breakdown

    # Monetization status label
    mono_status = {"HIGH": "NO ADS", "MEDIUM": "LIMITED ADS", "LOW": "FULL MONETIZATION"}[risk_level]
    mono_severity = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}[risk_level]

    # Strike risk — separate from overall content risk
    strikes = (profile or {}).get("past_strikes", 0)
    if has_high or (risk_level == "HIGH"):
        strike_risk, strike_sev = "ELEVATED", "medium"
    elif risk_level == "MEDIUM" and strikes >= 1:
        strike_risk, strike_sev = "ELEVATED", "medium"
    else:
        strike_risk, strike_sev = "LOW", "low"

    # Age restriction risk
    if threat >= 0.5 or severe >= 0.4 or identity >= 0.5:
        age_restriction, age_sev = "LIKELY", "high"
    elif threat >= 0.2 or severe >= 0.2:
        age_restriction, age_sev = "POSSIBLE", "medium"
    else:
        age_restriction, age_sev = "NONE", "low"

    # Advertiser opt-out risk
    if risk_level == "HIGH":
        optout_risk, optout_sev = "HIGH", "high"
    elif risk_level == "MEDIUM":
        optout_risk, optout_sev = "MODERATE", "medium"
    else:
        optout_risk, optout_sev = "LOW", "low"

    consequence_breakdown = [
        {"label": "Monetization",          "value": mono_status,   "severity": mono_severity},
        {"label": "Strike risk",           "value": strike_risk,   "severity": strike_sev},
        {"label": "Age restriction",       "value": age_restriction, "severity": age_sev},
        {"label": "Advertiser opt-out risk", "value": optout_risk, "severity": optout_sev},
    ]

    # Actionable recommendations
    recs = []
    if risk_level == "LOW":
        recs.append({"type": "pass", "text": "Safe to upload with full monetization"})
    if first7_hits:
        recs.append({"type": "warning", "text": "Move or remove flagged content from first 7 seconds"})
    # Only surface as "edit" recommendations those that are confirmed genuine harm
    # (context_check.genuine_harm = True) or have no context check at all.
    def _is_genuine(s):
        cc = s.get("context_check")
        if cc:
            # Parse errors mean we genuinely don't know — don't surface as a
            # confirmed flag. The raw score is still visible in the transcript.
            if cc.get("error"):
                return False
            return cc.get("genuine_harm", True)
        return True  # no context check ran → trust Detoxify score

    top_edits = sorted(
        [s for s in sentences if s.get("adjusted_toxicity", s["toxicity"]) >= 0.35 and _is_genuine(s)],
        key=lambda s: s.get("adjusted_toxicity", s["toxicity"]),
        reverse=True,
    )[:3]
    for s in top_edits:
        m, sec = int(s["start"] // 60), int(s["start"] % 60)
        cc = s.get("context_check") or {}
        recs.append({
            "type":      "edit",
            "text":      f"Consider editing {m}:{sec:02d} to protect monetization",
            "snippet":   s["text"][:70],
            "timestamp": s["start"],
            "toxicity":  s.get("adjusted_toxicity", s["toxicity"]),
            "intent":    cc.get("intent"),
            "reason":    cc.get("reason"),
            "safe_rewrite": cc.get("safe_rewrite"),
            "alt_rewrite":  cc.get("alt_rewrite"),
        })
    if not recs:
        recs.append({"type": "pass", "text": "No specific edits needed"})

    return {
        "risk_level":            risk_level,
        "ad_verdict":            ad_verdict,
        "ad_detail":             ad_detail,
        "first7_flagged":        bool(first7_hits),
        "first7_sentences":      first7_hits[:3],
        "signals":               signals,
        "consequence_breakdown": consequence_breakdown,
        "recommendations":       recs,
    }


def write_html(data: dict, output_path: str):
    template_path = Path(__file__).parent / "template.html"
    template = template_path.read_text()
    report_js = f"const REPORT = {json.dumps(data, indent=2)};"
    html = template.replace("const REPORT = null;", report_js)
    Path(output_path).write_text(html)
