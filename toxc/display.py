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
