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


_MAX_SENTENCE_WORDS = 50


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


def _segment_whisper_result(result: dict) -> list[dict]:
    """Split Whisper output into sentence-level chunks with timestamps.

    Processes each Whisper segment individually so that long stretches
    of unpunctuated speech don't collapse into a single giant block.
    Applies a hard word-count cap as a final safety net.
    """
    all_mapped: list[dict] = []

    for segment in result["segments"]:
        seg_text = segment["text"].strip()
        if not seg_text:
            continue

        seg_words = [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for w in segment.get("words", [])
        ]

        sents = nltk.sent_tokenize(seg_text)

        final_sents: list[str] = []
        for s in sents:
            tok = s.split()
            if len(tok) > _MAX_SENTENCE_WORDS:
                for i in range(0, len(tok), _MAX_SENTENCE_WORDS):
                    final_sents.append(" ".join(tok[i : i + _MAX_SENTENCE_WORDS]))
            else:
                final_sents.append(s)

        if seg_words:
            all_mapped.extend(_map_sentences(final_sents, seg_words))
        else:
            for s in final_sents:
                all_mapped.append({
                    "text": s,
                    "start": segment["start"],
                    "end": segment["end"],
                })

    return all_mapped


def _diarize(
    audio_path: str,
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[float, float, str]]:
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
        import subprocess, torchaudio
        # Convert to WAV (mono PCM) so torchaudio can load it.
        # Preserve original sample rate for better speaker discrimination.
        wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_tmp.close()
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path), "-ac", "1",
                 "-acodec", "pcm_s16le",
                 "-y", "-loglevel", "error", wav_tmp.name],
                capture_output=True, check=True,
            )
            waveform, sample_rate = torchaudio.load(wav_tmp.name)
            # Pass waveform dict to bypass pyannote's torchcodec decoder
            diarize_kwargs = {}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarization = pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                **diarize_kwargs,
            )
        finally:
            if os.path.exists(wav_tmp.name):
                os.unlink(wav_tmp.name)

    # pyannote v4 wraps the result in DiarizeOutput; unwrap to Annotation
    if hasattr(diarization, 'itertracks'):
        annotation = diarization
    elif hasattr(diarization, 'speaker_diarization'):
        annotation = diarization.speaker_diarization
    else:
        raise TypeError(
            f"Unexpected diarization result type: {type(diarization).__name__}. "
            "You may need to update pyannote.audio."
        )

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), speaker))
    sorted_turns = sorted(turns, key=lambda t: t[0])

    speakers = set(t[2] for t in sorted_turns)
    _console.print(
        f"[dim]  Diarization: {len(speakers)} speaker(s) detected, "
        f"{len(sorted_turns)} turn(s)[/dim]"
    )
    for spk in sorted(speakers):
        spk_turns = [t for t in sorted_turns if t[2] == spk]
        total_s = sum(t[1] - t[0] for t in spk_turns)
        _console.print(f"[dim]    {spk}: {len(spk_turns)} turns, {total_s:.0f}s total[/dim]")

    return sorted_turns


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
    min_speakers: int | None = None,
    max_speakers: int | None = None,
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
    mapped = _segment_whisper_result(result)

    if hf_token:
        try:
            turns = _diarize(audio_path, hf_token, min_speakers=min_speakers, max_speakers=max_speakers)
            mapped = _assign_speakers(mapped, turns)
        except Exception as e:
            _console.print(f"[yellow]⚠ Speaker diarization failed: {e}[/yellow]")
            _console.print("[dim]Continuing without speaker labels.[/dim]")

    return mapped, duration
