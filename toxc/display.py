from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

console = Console()

SENTIMENT_LABELS = {
    (0.05, 1.0): ("Positive", "green"),
    (-0.05, 0.05): ("Neutral", "yellow"),
    (-1.0, -0.05): ("Negative", "red"),
}

VERDICT_COLORS = {
    "Toxic": "bold red",
    "Borderline": "bold yellow",
    "Clean": "bold green",
}

def _bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)

def _sentiment_label(score: float):
    for (lo, hi), (label, color) in SENTIMENT_LABELS.items():
        if lo <= score <= hi:
            return label, color
    return "Neutral", "yellow"

def render_single(result: dict):
    toxicity = result["toxicity"]
    sentiment = result["sentiment"]
    verdict = result["verdict"]
    detail = result["detail"]

    verdict_color = VERDICT_COLORS.get(verdict, "white")
    sent_label, sent_color = _sentiment_label(sentiment)

    lines = Text()

    # Main scores
    lines.append(f"\n  {_bar(toxicity)}  ", style="dim")
    tox_color = "red" if toxicity >= 0.7 else "yellow" if toxicity >= 0.4 else "green"
    lines.append(f"Toxicity   ", style="bold")
    lines.append(f"{toxicity:.2f}  ", style=tox_color + " bold")
    lines.append(verdict.upper() + "\n", style=verdict_color)

    lines.append(f"\n  {_bar(abs(sentiment))}  ", style="dim")
    lines.append(f"Sentiment  ", style="bold")
    lines.append(f"{sentiment:+.2f}  ", style=sent_color + " bold")
    lines.append(sent_label + "\n", style=sent_color)

    # Dimensions (Detoxify only)
    if not result.get("fast"):
        dims = result["dimensions"]
        dim_labels = {
            "severe_toxicity": "Severe     ",
            "obscene":         "Obscene    ",
            "threat":          "Threat     ",
            "insult":          "Insult     ",
            "identity_attack": "Identity   ",
        }

        lines.append("\n")
        for key, label in dim_labels.items():
            score = dims.get(key, 0)
            if score > 0.05:
                color = "red" if score > 0.5 else "yellow" if score > 0.2 else "dim"
                lines.append(f"  {_bar(score, 8)}  ", style="dim")
                lines.append(label, style="bold")
                lines.append(f"{score:.2f}\n", style=color)

    lines.append(f"\n  Verdict: ", style="bold")
    lines.append(f"{verdict}", style=verdict_color)
    lines.append(f" — {detail}\n", style="dim")

    model_label = "vader" if result.get("fast") else "toxic-bert"
    panel = Panel(
        lines,
        title="[bold]toxc[/bold]",
        subtitle=f"[dim]{model_label}[/dim]",
        border_style=tox_color,
        padding=(0, 1),
    )
    console.print(panel)


def render_batch(results: list[dict]):
    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Text", style="dim", max_width=40)
    table.add_column("Toxicity", justify="center")
    table.add_column("Sentiment", justify="center")
    table.add_column("Verdict", justify="center")

    for r in results:
        tox = r["toxicity"]
        sent = r["sentiment"]
        verdict = r["verdict"]

        tox_color = "red" if tox >= 0.7 else "yellow" if tox >= 0.4 else "green"
        sent_label, sent_color = _sentiment_label(sent)
        verdict_color = VERDICT_COLORS.get(verdict, "white")

        text_preview = r["text"][:38] + "…" if len(r["text"]) > 40 else r["text"]

        table.add_row(
            text_preview,
            Text(f"{tox:.2f}", style=tox_color + " bold"),
            Text(f"{sent:+.2f} {sent_label}", style=sent_color),
            Text(verdict, style=verdict_color),
        )

    console.print(table)


def render_voice_summary(data: dict):
    O = data["overall"]
    M = data.get("monetization", {})
    toxicity = O["toxicity"]
    sentiment = O["sentiment"]

    tox_color = "red" if toxicity >= 0.7 else "yellow" if toxicity >= 0.4 else "green"
    sent_label, sent_color = _sentiment_label(sentiment)

    risk_level = M.get("risk_level", "")
    risk_color = {"HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "bold green"}.get(risk_level, "white")
    ad_verdict = M.get("ad_verdict", "")

    lines = Text()

    # Monetization verdict (top of panel)
    if risk_level:
        lines.append(f"\n  Ad Safety   ", style="bold")
        lines.append(f"{risk_level}  ", style=risk_color)
        lines.append(ad_verdict + "\n", style="dim")

    lines.append(f"\n  {_bar(toxicity)}  ", style="dim")
    lines.append("Toxicity   ", style="bold")
    lines.append(f"{toxicity:.2f}\n", style=tox_color + " bold")

    lines.append(f"\n  {_bar(abs(sentiment))}  ", style="dim")
    lines.append("Sentiment  ", style="bold")
    lines.append(f"{sentiment:+.2f}  ", style=sent_color + " bold")
    lines.append(sent_label + "\n", style=sent_color)

    lines.append(f"\n  Sentences  ", style="bold")
    lines.append(f"{O['toxic_count']} flagged", style="red" if O['toxic_count'] else "dim")
    lines.append(" / ", style="dim")
    lines.append(f"{O['sentence_count']} total\n", style="dim")

    # Per-speaker breakdown
    speakers = data.get("speakers") or []
    if speakers:
        lines.append("\n  [bold]Speakers[/bold]\n")
        for spk in speakers:
            sc_map = {"Review": "red", "Borderline": "yellow", "Clean": "green"}
            sc = sc_map.get(spk["verdict"], "dim")
            lines.append(f"  {spk['speaker']:<16}", style="dim")
            lines.append(f"{spk['avg_toxicity']:.2f}  ", style=sc + " bold")
            lines.append(f"{spk['sentence_count']} sent", style="dim")
            if spk["flagged"]:
                lines.append(f"  {spk['flagged']} flagged", style="yellow")
            lines.append(f"  {spk['verdict']}\n", style=sc)

    # Risk signals
    signals = M.get("signals", [])
    if signals:
        lines.append("\n  [bold]Risk signals[/bold]\n")
        for sig in signals:
            sc = "red" if sig["severity"] == "high" else "yellow"
            lines.append(f"  {'⚠' if sig['severity'] == 'high' else '·'}  ", style=sc)
            lines.append(sig["label"] + "\n", style="dim")
    elif risk_level == "LOW":
        lines.append("\n  ✓  ", style="bold green")
        lines.append("No significant risk signals\n", style="dim")

    # Top risky moments
    top = [s for s in data["top_toxic"] if s["toxicity"] >= 0.35][:3]
    if top:
        lines.append("\n  [bold]Flagged moments[/bold]\n")
        for s in top:
            m, sec = int(s["start"] // 60), int(s["start"] % 60)
            tc = "red" if s["toxicity"] >= 0.7 else "yellow"
            lines.append(f"  {m}:{sec:02d}  ", style="dim")
            lines.append(f"{s['toxicity']:.2f}  ", style=tc + " bold")
            lines.append(s["text"][:58] + ("…" if len(s["text"]) > 58 else "") + "\n", style="dim")

    # Recommendations
    recs = [r for r in M.get("recommendations", []) if r["type"] == "edit"][:2]
    if recs:
        lines.append("\n  [bold]Recommendations[/bold]\n")
        for r in recs:
            lines.append(f"  ·  ", style="yellow")
            lines.append(r["text"] + "\n", style="dim")

    # Policy review verdict (Pass 3)
    PR = data.get("policy_review") or {}
    if PR.get("ran") and not PR.get("error"):
        v = PR.get("monetization_verdict", "yellow")
        vc_map = {"green": "green", "yellow": "yellow", "red": "red"}
        prc = vc_map.get(v, "yellow")
        label_map = {
            "green":  "Full monetization likely",
            "yellow": "Limited ads likely",
            "red":    "Demonetization likely",
        }
        rec_map = {
            "safe":        "Safe to publish",
            "edit_first":  "Edit before publishing",
            "reconsider":  "Reconsider content",
        }
        lines.append("\n  [bold]AI Policy Review[/bold]  ", style="bold")
        lines.append(f"[{prc}]{label_map.get(v, v)}[/{prc}]", style="")
        conf = PR.get("confidence")
        if conf is not None:
            lines.append(f"  {int(conf*100)}% confidence\n", style="dim")
        else:
            lines.append("\n")
        if PR.get("overall_summary"):
            summary = PR["overall_summary"]
            lines.append(f"  {summary}\n", style="dim")
        rec = PR.get("publish_recommendation", "edit_first")
        lines.append(f"  Recommendation  ", style="bold")
        lines.append(rec_map.get(rec, rec) + "\n", style="dim")
        viols = PR.get("policy_violations") or []
        if viols:
            lines.append(f"\n  {len(viols)} policy flag{'s' if len(viols) != 1 else ''}\n", style="yellow")
            for vi in viols[:3]:
                ts = vi.get("timestamp", "")
                policy = vi.get("policy", "")
                lines.append(f"  ⚠  {ts}  ", style="yellow dim")
                lines.append(f"{policy}\n", style="dim")

    lines.append("\n")

    model_label = "vader · whisper" if data.get("fast") else "toxic-bert · whisper"
    panel = Panel(
        lines,
        title="[bold]toxc voice[/bold]",
        subtitle=f"[dim]{model_label}[/dim]",
        border_style=tox_color,
        padding=(0, 1),
    )
    console.print(panel)


def render_check(result: dict):
    """Render a YouTube title/text safety check."""
    toxicity = result["toxicity"]
    tox_color = "red" if toxicity >= 0.7 else "yellow" if toxicity >= 0.4 else "green"

    # Simple YouTube risk mapping for a single text snippet
    dims = result.get("dimensions", {})
    identity = dims.get("identity_attack", 0)
    threat   = dims.get("threat", 0)
    obscene  = dims.get("obscene", 0)

    if toxicity >= 0.6 or identity >= 0.3 or threat >= 0.3:
        risk, risk_color, advice = "HIGH RISK", "bold red", "Not safe — avoid in title/thumbnail"
    elif toxicity >= 0.3 or obscene >= 0.15:
        risk, risk_color, advice = "CAUTION", "bold yellow", "May trigger limited ads — rephrase if possible"
    else:
        risk, risk_color, advice = "SAFE", "bold green", "Safe for title and thumbnail use"

    lines = Text()
    lines.append(f"\n  YouTube Risk  ", style="bold")
    lines.append(f"{risk}\n", style=risk_color)

    lines.append(f"\n  {_bar(toxicity)}  ", style="dim")
    lines.append("Toxicity   ", style="bold")
    lines.append(f"{toxicity:.2f}\n", style=tox_color + " bold")

    if not result.get("fast") and any(v > 0.05 for v in dims.values()):
        lines.append("\n")
        for key, label in [("obscene", "Profanity "), ("identity_attack", "Slurs    "), ("threat", "Threats  ")]:
            v = dims.get(key, 0)
            if v > 0.05:
                color = "red" if v > 0.5 else "yellow" if v > 0.2 else "dim"
                lines.append(f"  {_bar(v, 8)}  ", style="dim")
                lines.append(label, style="bold")
                lines.append(f"{v:.2f}\n", style=color)

    lines.append(f"\n  {advice}\n", style="dim")
    lines.append("\n")

    panel = Panel(
        lines,
        title="[bold]toxc check[/bold]",
        subtitle="[dim]youtube title & thumbnail safety[/dim]",
        border_style=tox_color,
        padding=(0, 1),
    )
    console.print(panel)
