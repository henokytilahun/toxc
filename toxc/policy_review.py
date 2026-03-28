"""
Pass 3 — LLM-powered YouTube policy review.

Reads the full transcript + YouTube's Advertiser-Friendly Content Guidelines
and renders a holistic verdict that rule-based models fundamentally cannot:

- Policy-aware judgment (the LLM reads the actual guidelines)
- Intent understanding across the full narrative arc
- Title and thumbnail risk assessment
- Voice-preserving rewrites for confirmed flags

This runs AFTER Pass 1 (Detoxify/VADER) and optionally after Pass 2 (context
check), so the flagged sentences passed in already have adjusted scores when
context check has run.

Requires Ollama to be running locally (same dependency as Pass 2):
  ollama serve
  ollama pull llama3.2
"""

import json
import re


_SYSTEM_PROMPT = (
    "You are a YouTube advertiser-friendliness reviewer. "
    "You know YouTube's Advertiser-Friendly Content Guidelines thoroughly.\n\n"

    "WILL CAUSE DEMONETIZATION (red):\n"
    "- Excessive profanity as the focal point of the content\n"
    "- Racial, homophobic, or identity-based slurs\n"
    "- Graphic violence or real threats of harm\n"
    "- Sexual content or explicit language\n"
    "- Promotion of dangerous acts or harmful substances\n"
    "- Sensitive events exploited for shock value or profit\n\n"

    "WILL CAUSE LIMITED ADS / YELLOW ICON (yellow):\n"
    "- Moderate profanity (not the focal point)\n"
    "- Controversial political or social topics without nuance\n"
    "- Tragedy or conflict covered (even respectfully)\n"
    "- Drug/alcohol references (not promotional)\n"
    "- Profanity in the first 7 seconds\n\n"

    "SAFE FOR FULL MONETIZATION (green):\n"
    "- Clean language throughout\n"
    "- Controversial topics discussed with genuine context and balance\n"
    "- Competitive or hyperbolic language that is clearly not literal\n"
    "- Criticism, satire, or commentary — when not hateful\n\n"

    "CRITICAL NUANCES YOU MUST APPLY:\n"
    "- 'You're killing it', 'insane performance', 'destroy this argument' are NOT violations\n"
    "- Sarcasm and irony must be read in full context, not in isolation\n"
    "- Titles and thumbnails are held to STRICTER standards than video content\n"
    "- Profanity DENSITY matters more than isolated individual instances\n"
    "- Gaming and commentary channels have more latitude than education or kids content\n"
    "- A video can have zero toxic sentences and still get demonetized for overall topic sensitivity\n\n"

    "Reply ONLY with valid JSON. No text outside the JSON object."
)

_USER_TEMPLATE = """\
VIDEO TITLE: "{title}"
CHANNEL CATEGORY: {category}

FULL TRANSCRIPT:
{transcript}

SENTENCES FLAGGED BY TOXICITY MODEL (raw scores, may include false positives):
{flagged_json}

Respond with this exact JSON structure and nothing else:
{{
  "monetization_verdict": "green" or "yellow" or "red",
  "confidence": <float 0.0-1.0>,
  "primary_risk": "<main issue if any, or 'none'>",
  "policy_violations": [
    {{
      "policy": "<name of YouTube guideline>",
      "severity": "low" or "medium" or "high",
      "timestamp": "<M:SS or 'title'>",
      "quote": "<exact sentence or title text>",
      "reason": "<why this is a risk>"
    }}
  ],
  "false_positives": ["<sentences the toxicity model flagged that are clearly fine in context>"],
  "title_risk": "green" or "yellow" or "red",
  "title_reason": "<one sentence explaining the title risk rating>",
  "overall_summary": "<2-3 sentence plain-English verdict the creator can act on>",
  "rewrites": [
    {{
      "timestamp": "<M:SS or 'title'>",
      "original": "<original text>",
      "safe_rewrite": "<same energy and voice, ad-safe>",
      "conservative_rewrite": "<safer option — may shift tone slightly>"
    }}
  ],
  "publish_recommendation": "safe" or "edit_first" or "reconsider"
}}"""


def _repair_json(raw: str) -> str:
    """Apply common fixes for LLM-produced JSON."""
    s = raw
    s = re.sub(r",\s*([}\]])", r"\1", s)          # trailing commas
    s = re.sub(r'\bTrue\b', 'true', s)             # Python booleans
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b', 'null', s)
    # unescaped newlines inside string values
    s = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace('\n', ' '), s, flags=re.DOTALL)
    return s


def _extract_json_object(raw: str) -> str | None:
    """Find the outermost balanced {...} in raw text."""
    start = raw.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == '{':
            depth += 1
        elif raw[i] == '}':
            depth -= 1
        if depth == 0:
            return raw[start:i + 1]
    # Unbalanced — try closing it
    return raw[start:] + '}'* depth if depth > 0 else None


def _parse_json_lenient(raw: str) -> dict:
    """Try increasingly aggressive strategies to parse LLM JSON output."""
    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract {...} then parse
    extracted = _extract_json_object(raw)
    if extracted:
        try:
            return json.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 3: repair + parse
        repaired = _repair_json(extracted)
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: repair full text, then extract
    repaired_full = _repair_json(raw)
    extracted2 = _extract_json_object(repaired_full)
    if extracted2:
        try:
            return json.loads(extracted2)
        except (json.JSONDecodeError, ValueError):
            pass

    raise json.JSONDecodeError(
        "Could not parse LLM JSON", raw[:200], 0
    )


def policy_review(
    sentences: list[dict],
    title: str | None = None,
    channel_category: str = "other",
    model: str = "llama3.2",
) -> dict:
    """
    Run a full LLM policy review on the transcript.

    Uses the complete transcript for holistic narrative judgment — not just
    individual flagged sentences. This catches videos that have zero toxic
    sentences but still risk demonetization due to overall topic sensitivity.

    Args:
        sentences:         list of sentence dicts (must have "text", "start",
                           "toxicity", optionally "adjusted_toxicity")
        title:             video title (used for title risk assessment)
        channel_category:  channel category string (gaming/education/etc.)
        model:             Ollama model name

    Returns:
        Structured verdict dict. Always has "ran": True.
        Has "error": True on failure.
    """
    import ollama

    # Build full transcript (truncate to ~8000 chars to stay within context)
    # Include speaker labels when diarization ran — gives the LLM crucial context
    # for intent disambiguation ("SPEAKER_00 calling SPEAKER_01's argument brutal
    # is a compliment, not a threat").
    transcript_lines = []
    for s in sentences:
        start = s.get("start", 0)
        m, sec = int(start // 60), int(start % 60)
        speaker = s.get("speaker")
        prefix = f"[{m}:{sec:02d}] [{speaker}] " if speaker else f"[{m}:{sec:02d}] "
        transcript_lines.append(f"{prefix}{s['text']}")
    transcript = "\n".join(transcript_lines)
    if len(transcript) > 8000:
        transcript = transcript[:8000] + "\n[... transcript truncated ...]"

    # Build flagged sentences list — prefer adjusted_toxicity if Pass 2 ran
    flagged = []
    for s in sentences:
        score = s.get("adjusted_toxicity", s.get("toxicity", 0))
        if score >= 0.35:
            start = s.get("start", 0)
            entry = {
                "timestamp": f"{int(start // 60)}:{int(start % 60):02d}",
                "text": s["text"],
                "raw_score": round(s.get("toxicity", 0), 3),
                "adjusted_score": round(score, 3),
            }
            # Include Pass 2 intent if available — helps the LLM
            cc = s.get("context_check")
            if cc and not cc.get("error"):
                entry["pass2_intent"] = cc.get("intent", "unknown")
                entry["pass2_genuine_harm"] = cc.get("genuine_harm", True)
            flagged.append(entry)

    prompt = _USER_TEMPLATE.format(
        title=title or "(no title)",
        category=channel_category,
        transcript=transcript,
        flagged_json=json.dumps(flagged, indent=2)[:3000],
    )

    def _call_and_parse(messages, attempt_label=""):
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 2048},
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
        return _parse_json_lenient(raw), raw

    # Attempt 1: full prompt
    last_raw = ""
    try:
        result, last_raw = _call_and_parse([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ], "full")
    except (json.JSONDecodeError, ValueError):
        # Attempt 2: retry with a much simpler schema
        simple_prompt = (
            f'VIDEO TITLE: "{title or "(no title")}"\n\n'
            f"TRANSCRIPT (first 3000 chars):\n{transcript[:3000]}\n\n"
            "Reply with ONLY this JSON and nothing else:\n"
            '{"monetization_verdict":"green or yellow or red",'
            '"confidence":0.7,'
            '"primary_risk":"main risk or none",'
            '"overall_summary":"2 sentence verdict",'
            '"publish_recommendation":"safe or edit_first or reconsider",'
            '"title_risk":"green or yellow or red",'
            '"title_reason":"one sentence",'
            '"policy_violations":[],"false_positives":[],"rewrites":[]}'
        )
        try:
            result, last_raw = _call_and_parse([
                {"role": "system", "content": "You are a YouTube ad-safety reviewer. Reply ONLY with valid JSON."},
                {"role": "user",   "content": simple_prompt},
            ], "simple")
        except (json.JSONDecodeError, ValueError) as e:
            snippet = last_raw[:300] if last_raw else "(empty response)"
            return {
                "ran": True, "error": True, "model": model,
                "error_detail": f"JSON parse error after 2 attempts: {e}",
                "raw_snippet": snippet,
                "monetization_verdict": "yellow", "confidence": 0.5,
                "overall_summary": "Policy review ran but could not parse LLM response.",
                "publish_recommendation": "edit_first",
                "policy_violations": [], "false_positives": [], "rewrites": [],
                "primary_risk": "none", "title_risk": "green", "title_reason": "",
            }
    except Exception as e:
        return {
            "ran": True, "error": True, "model": model,
            "error_detail": str(e),
            "monetization_verdict": "yellow", "confidence": 0.5,
            "overall_summary": "Policy review failed — Ollama connection error.",
            "publish_recommendation": "edit_first",
            "policy_violations": [], "false_positives": [], "rewrites": [],
            "primary_risk": "none", "title_risk": "green", "title_reason": "",
        }

    # Normalise and clamp fields
    result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.7))))
    result["monetization_verdict"] = result.get("monetization_verdict", "yellow").lower()
    result["title_risk"] = result.get("title_risk", "green").lower()
    result["publish_recommendation"] = result.get("publish_recommendation", "edit_first")
    result.setdefault("policy_violations", [])
    result.setdefault("rewrites", [])

    raw_fps = result.get("false_positives") or []
    result["false_positives"] = [
        fp if isinstance(fp, str) else fp.get("text") or fp.get("quote") or str(fp)
        for fp in raw_fps
    ]

    result["rewrites"] = [
        rw for rw in result["rewrites"]
        if rw.get("safe_rewrite") or rw.get("conservative_rewrite")
    ]
    result.setdefault("overall_summary", "")
    result.setdefault("primary_risk", "none")
    result.setdefault("title_reason", "")
    result["ran"] = True
    result["model"] = model
    return result
