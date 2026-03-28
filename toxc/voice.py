import os
import re
import tempfile
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import nltk
from rich.console import Console

_console = Console(stderr=True)

_YT_RE = re.compile(
    r"(https?://)?(www\.)?"
    r"(youtube\.com/(watch\?.*v=|shorts/|live/)|youtu\.be/)"
    r"[\w\-]+"
)


def is_youtube_url(source: str) -> bool:
    return bool(_YT_RE.match(source.strip()))


def fetch_youtube_audio(url: str) -> tuple[str, dict]:
    """
    Download the best audio track to a temp directory via yt-dlp.
    Returns (audio_file_path, metadata_dict).
    The caller is responsible for deleting the file (and its parent tmpdir).
    No FFmpeg post-processing — Whisper decodes the native format directly.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required for YouTube support. "
            "Install it with: pip install yt-dlp"
        )

    tmpdir = tempfile.mkdtemp(prefix="toxc_yt_")

    ydl_opts = {
        "format": "bestaudio/best",
        # %(ext)s lets yt-dlp keep the native extension (.webm, .m4a, etc.)
        "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    meta: dict = {}
    with _console.status("[dim]Fetching YouTube audio…[/dim]", spinner="dots"):
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            meta = {
                "title": info.get("title", ""),
                "channel": info.get("uploader", ""),
                "url": url,
                "duration": info.get("duration", 0),
                "thumbnail": info.get("thumbnail", ""),
            }

    # Find whatever file yt-dlp wrote (audio.webm, audio.m4a, …)
    files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
    if not files:
        raise RuntimeError("yt-dlp did not produce an audio file.")

    audio_path = os.path.join(tmpdir, files[0])
    # Stash the tmpdir so the CLI can clean up the whole directory
    meta["_tmpdir"] = tmpdir
    return audio_path, meta


def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _extract_words(whisper_result: dict) -> list[dict]:
    words = []
    for segment in whisper_result["segments"]:
        for word in segment.get("words", []):
            words.append({"word": word["word"], "start": word["start"], "end": word["end"]})
    return words


def _map_sentences(sentences: list[str], words: list[dict]) -> list[dict]:
    """Map each NLTK sentence back to Whisper word-level timestamps."""
    results = []
    word_idx = 0

    for sentence in sentences:
        target_len = len(re.sub(r"[^\w]", "", sentence).lower())
        consumed = ""
        start_time = None
        end_time = None

        while word_idx < len(words) and len(consumed) < target_len:
            w = words[word_idx]
            w_clean = re.sub(r"[^\w]", "", w["word"]).lower()
            if w_clean:
                if start_time is None:
                    start_time = w["start"]
                end_time = w["end"]
                consumed += w_clean
            word_idx += 1

        results.append({
            "text": sentence.strip(),
            "start": start_time or 0.0,
            "end": end_time or 0.0,
        })

    return results


def _diarize(audio_path: str, hf_token: str) -> list[tuple[float, float, str]]:
    """
    Run pyannote speaker diarization on `audio_path`.
    Returns a list of (start, end, speaker_label) tuples sorted by start time.

    Requires: pip install toxc[diarize]  and a HuggingFace token with
              pyannote/speaker-diarization-3.1 access granted at
              https://huggingface.co/pyannote/speaker-diarization-3.1
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise RuntimeError(
            "pyannote.audio is required for speaker diarization.\n"
            "Install it with:  pip install toxc[diarize]\n"
            "Then accept the model license at https://hf.co/pyannote/speaker-diarization-3.1\n"
            "and save your token:  toxc config set --hf-token hf_xxxx"
        )

    with _console.status("[dim]Loading speaker diarization model…[/dim]", spinner="dots"):
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )

    with _console.status("[dim]Running speaker diarization…[/dim]", spinner="dots"):
        diarization = pipeline(str(audio_path))

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), speaker))
    return sorted(turns, key=lambda t: t[0])


def _assign_speakers(sentences: list[dict], turns: list[tuple[float, float, str]]) -> list[dict]:
    """
    Assign a speaker label to each sentence using midpoint-overlap lookup.
    Falls back to nearest segment when the midpoint doesn't land inside any turn.
    """
    if not turns:
        return sentences

    for s in sentences:
        mid = (s["start"] + s["end"]) / 2

        # Primary: find the turn that contains the midpoint
        speaker = None
        for seg_start, seg_end, spk in turns:
            if seg_start <= mid <= seg_end:
                speaker = spk
                break

        # Fallback: nearest turn boundary
        if speaker is None:
            nearest = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))
            speaker = nearest[2]

        s["speaker"] = speaker

    return sentences


def transcribe_and_segment(
    audio_path: str,
    model_size: str = "small",
    hf_token: str | None = None,
) -> tuple[list[dict], float]:
    """
    Transcribe audio and segment into sentences with timestamps.

    Returns (sentences, duration).
    Each sentence: {"text": str, "start": float, "end": float}
    When hf_token is provided, each sentence also has "speaker": str.
    """
    _ensure_nltk()

    with _console.status(f"[dim]Loading Whisper ({model_size})…[/dim]", spinner="dots"):
        import whisper
        model = whisper.load_model(model_size)

    with _console.status("[dim]Transcribing…[/dim]", spinner="dots"):
        result = model.transcribe(str(audio_path), word_timestamps=True)

    duration = result["segments"][-1]["end"] if result["segments"] else 0.0
    words = _extract_words(result)
    sentences = nltk.sent_tokenize(result["text"].strip())

    if not words:
        mapped = [{"text": s, "start": 0.0, "end": duration} for s in sentences]
    else:
        mapped = _map_sentences(sentences, words)

    if hf_token:
        try:
            turns = _diarize(audio_path, hf_token)
            mapped = _assign_speakers(mapped, turns)
        except Exception as e:
            _console.print(f"[yellow]⚠ Speaker diarization failed: {e}[/yellow]")
            _console.print("[dim]Continuing without speaker labels.[/dim]")

    return mapped, duration
