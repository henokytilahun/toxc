import os
import typer
import sys
import json as _json
from typing import Optional
from pathlib import Path
from toxc.analyze import analyze, analyze_batch
from toxc.display import render_single, render_batch, console

app = typer.Typer(help="Toxicity & sentiment analysis for text.", add_completion=False)
config_app = typer.Typer(help="Manage your channel profile.", add_completion=False)
app.add_typer(config_app, name="config")


def _output_json(results):
    console.print_json(_json.dumps(results, indent=2))


# ── config subcommands ────────────────────────────────────────────────────────

@config_app.command("show")
def config_show():
    """Show your saved channel profile."""
    from toxc.profile import load_profile, PROFILE_PATH
    profile = load_profile()
    if not profile:
        console.print(f"[dim]No profile saved. Run [bold]toxc config setup[/bold] to create one.[/dim]")
        raise typer.Exit()
    console.print(f"\n[dim]Profile: {PROFILE_PATH}[/dim]\n")
    for k, v in profile.items():
        console.print(f"  [bold]{k:<20}[/bold] {v}")
    console.print()


@config_app.command("set")
def config_set(
    channel_name:    Optional[str]   = typer.Option(None, "--channel-name"),
    monthly_views:   Optional[int]   = typer.Option(None, "--monthly-views"),
    subscribers:     Optional[int]   = typer.Option(None, "--subscribers"),
    cpm:             Optional[float] = typer.Option(None, "--cpm"),
    videos_per_month: Optional[int]  = typer.Option(None, "--videos-per-month"),
    category:        Optional[str]   = typer.Option(None, "--category"),
    past_strikes:    Optional[int]   = typer.Option(None, "--past-strikes"),
):
    """Update one or more profile fields without re-running full setup."""
    from toxc.profile import load_profile, save_profile
    profile = load_profile() or {}

    updates = {
        "channel_name":    channel_name,
        "monthly_views":   monthly_views,
        "subscribers":     subscribers,
        "cpm":             cpm,
        "videos_per_month": videos_per_month,
        "category":        category,
        "past_strikes":    past_strikes,
    }
    changed = {k: v for k, v in updates.items() if v is not None}
    if not changed:
        console.print("[dim]No fields specified. Use --cpm 4.50 etc.[/dim]")
        raise typer.Exit()

    profile.update(changed)
    save_profile(profile)
    for k, v in changed.items():
        console.print(f"  [green]✓[/green] {k} = {v}")
    console.print()


@config_app.command("setup")
def config_setup():
    """Run the interactive channel profile setup."""
    from toxc.profile import prompt_for_profile
    prompt_for_profile()


@config_app.command("reset")
def config_reset():
    """Delete your saved channel profile."""
    from toxc.profile import PROFILE_PATH
    if PROFILE_PATH.exists():
        PROFILE_PATH.unlink()
        console.print("[dim]Profile deleted.[/dim]")
    else:
        console.print("[dim]No profile to delete.[/dim]")


# ── analyze ───────────────────────────────────────────────────────────────────

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


# ── voice ─────────────────────────────────────────────────────────────────────

@app.command("voice")
def voice_cmd(
    source: str = typer.Argument(..., help="Audio/video file path or YouTube URL"),
    html: Optional[Path] = typer.Option(None, "--html", help="Save HTML report to path"),
    model: str = typer.Option("small", "--model", "-m", help="Whisper model size (tiny/base/small/medium/large)"),
    fast: bool = typer.Option(False, "--fast", help="Use VADER only (no Detoxify)"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    no_profile: bool = typer.Option(False, "--no-profile", help="Skip channel profile / financial analysis"),
):
    from toxc.voice import transcribe_and_segment, is_youtube_url
    from toxc.report import aggregate, write_html
    from toxc.profile import load_profile, prompt_for_profile

    audio_path = source
    yt_meta: dict = {}

    if is_youtube_url(source):
        from toxc.voice import fetch_youtube_audio
        audio_path, yt_meta = fetch_youtube_audio(source)
        display_name = yt_meta.get("title") or source
    else:
        p = Path(source)
        if not p.exists():
            typer.echo(f"File not found: {source}", err=True)
            raise typer.Exit(1)
        display_name = p.name

    # Load or prompt for channel profile
    profile: dict | None = None
    if not no_profile:
        profile = load_profile()
        if profile is None:
            profile = prompt_for_profile()

    try:
        sentences, duration = transcribe_and_segment(audio_path, model)

        if not sentences:
            typer.echo("No speech detected.", err=True)
            raise typer.Exit(1)

        use_detoxify = not fast
        texts = [s["text"] for s in sentences]
        results = analyze_batch(texts, use_detoxify)

        for result, sentence in zip(results, sentences):
            result["start"] = sentence["start"]
            result["end"] = sentence["end"]

        data = aggregate(results, display_name, model, duration, profile=profile)

        if yt_meta:
            data["youtube"] = yt_meta

        if json:
            _output_json(data)
            return

        from toxc.display import render_voice_summary
        render_voice_summary(data)

        if html:
            write_html(data, str(html))
            console.print(f"\n[dim]Report saved → [bold]{html}[/bold][/dim]")

    finally:
        if yt_meta.get("_tmpdir"):
            import shutil
            shutil.rmtree(yt_meta["_tmpdir"], ignore_errors=True)


# ── check ─────────────────────────────────────────────────────────────────────

@app.command("check")
def check_cmd(
    text: str = typer.Argument(..., help="Title, thumbnail text, or short phrase to check"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    fast: bool = typer.Option(False, "--fast", help="Use VADER only"),
):
    """Check a title or thumbnail text for YouTube ad safety."""
    result = analyze(text, use_detoxify=not fast)
    if json:
        _output_json(result)
    else:
        from toxc.display import render_check
        render_check(result)


def run():
    # Allow `toxc "text"` without an explicit subcommand.
    # If the first argument isn't a known subcommand or flag, inject "analyze".
    known = {"voice", "analyze", "check", "config", "--help", "-h"}
    first = sys.argv[1] if len(sys.argv) > 1 else None
    if first is not None and first not in known:
        sys.argv.insert(1, "analyze")
    app()


if __name__ == "__main__":
    run()
