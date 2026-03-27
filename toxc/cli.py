import typer
import sys
import json as _json
from typing import Optional
from pathlib import Path
from toxc.analyze import analyze, analyze_batch
from toxc.display import render_single, render_batch, console

app = typer.Typer(help="Toxicity & sentiment analysis for text.", add_completion=False)


def _output_json(results):
    console.print_json(_json.dumps(results, indent=2))


@app.command("analyze")
def analyze_cmd(
    text: Optional[str] = typer.Argument(None, help="Text to analyze"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File with one text per line"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    fast: bool = typer.Option(False, "--fast", help="Use VADER only (no DistilBERT, faster)"),
):
    use_detoxify = not fast
    texts = []

    if not sys.stdin.isatty() and text is None and file is None:
        texts = [line.strip() for line in sys.stdin if line.strip()]
    elif file:
        if not file.exists():
            typer.echo(f"File not found: {file}", err=True)
            raise typer.Exit(1)
        texts = [l.strip() for l in file.read_text().splitlines() if l.strip()]
    elif text:
        texts = [text]
    else:
        typer.echo("Provide text, --file, or pipe input. Try: toxc --help")
        raise typer.Exit(1)

    if len(texts) == 1:
        result = analyze(texts[0], use_detoxify)
        if json:
            _output_json(result)
        else:
            render_single(result)
    else:
        results = analyze_batch(texts, use_detoxify)
        if json:
            _output_json(results)
        else:
            render_batch(results)


@app.command("voice")
def voice_cmd(
    audio: Path = typer.Argument(..., help="Audio or video file to analyze"),
    html: Optional[Path] = typer.Option(None, "--html", help="Save HTML report to path"),
    model: str = typer.Option("small", "--model", "-m", help="Whisper model size (tiny/base/small/medium/large)"),
    fast: bool = typer.Option(False, "--fast", help="Use VADER only (no Detoxify)"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    if not audio.exists():
        typer.echo(f"File not found: {audio}", err=True)
        raise typer.Exit(1)

    from toxc.voice import transcribe_and_segment
    from toxc.report import aggregate, write_html

    sentences, duration = transcribe_and_segment(str(audio), model)

    if not sentences:
        typer.echo("No speech detected.", err=True)
        raise typer.Exit(1)

    use_detoxify = not fast
    texts = [s["text"] for s in sentences]
    results = analyze_batch(texts, use_detoxify)

    for result, sentence in zip(results, sentences):
        result["start"] = sentence["start"]
        result["end"] = sentence["end"]

    data = aggregate(results, str(audio), model, duration)

    if json:
        _output_json(data)
        return

    from toxc.display import render_voice_summary
    render_voice_summary(data)

    if html:
        write_html(data, str(html))
        console.print(f"\n[dim]Report saved → [bold]{html}[/bold][/dim]")


def run():
    # Allow `toxc "text"` without an explicit subcommand.
    # If the first argument isn't a known subcommand or flag, inject "analyze".
    known = {"voice", "analyze", "--help", "-h"}
    first = sys.argv[1] if len(sys.argv) > 1 else None
    if first is not None and first not in known:
        sys.argv.insert(1, "analyze")
    app()


if __name__ == "__main__":
    run()
