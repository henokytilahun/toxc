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
    Download the best audio track to a temp file via yt-dlp.
    Returns (temp_file_path, metadata_dict).
    The caller is responsible for deleting the temp file.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError(
            "yt-dlp is required for YouTube support. "
            "Install it with: pip install yt-dlp"
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp.name,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
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

    # yt-dlp appends .mp3 to the template path when post-processing
    audio_path = tmp.name if os.path.exists(tmp.name) else tmp.name + ".mp3"
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


def transcribe_and_segment(audio_path: str, model_size: str = "small") -> tuple[list[dict], float]:
    """
    Returns (sentences, duration).
    Each sentence: {"text": str, "start": float, "end": float}
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
        return [{"text": s, "start": 0.0, "end": duration} for s in sentences], duration

    return _map_sentences(sentences, words), duration
