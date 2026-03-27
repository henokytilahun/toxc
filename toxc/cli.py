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


@app.command()
def main(
    text: Optional[str] = typer.Argument(None, help="Text to analyze"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File with one text per line"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    fast: bool = typer.Option(False, "--fast", help="Use VADER only (no DistilBERT, faster)"),
):
    use_detoxify = not fast
    texts = []

    # Piped input
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


def run():
    app()


if __name__ == "__main__":
    run()
