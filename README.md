# toxc

**CLI toxicity & sentiment analysis for text and audio/video.**

Analyze a single string, a batch of lines, or a full audio/video file — toxc transcribes speech with Whisper, scores every sentence for toxicity and sentiment, and renders an interactive HTML report.

```bash
pip install toxc
```

---

## Text Analysis

**Analyze a string**
```bash
toxc "you're such an idiot"
```
```
╭────────────────────── toxc ──────────────────────╮
│                                                  │
│  ████████░░  Toxicity   0.81  TOXIC              │
│  ██░░░░░░░░  Sentiment  -0.51  Negative          │
│                                                  │
│  ██░░░░░░  Severe     0.18                       │
│  ███░░░░░  Obscene    0.31                       │
│  █░░░░░░░  Threat     0.09                       │
│  ████░░░░  Insult     0.62                       │
│                                                  │
│  Verdict: Toxic — personal attack detected       │
│                                                  │
╰──────────────────────────────────────────────────╯
```

**Pipe input / batch mode**
```bash
cat comments.txt | toxc
```
```
  Text                          Toxicity   Sentiment      Verdict
 ──────────────────────────────────────────────────────────────────
  I love this project!            0.00    +0.85 Positive   Clean
  You're such an idiot            0.79    -0.51 Negative   Toxic
  The weather is fine today       0.00    +0.20 Positive   Clean
  I will destroy you              0.81    -0.54 Negative   Toxic
```

**File input**
```bash
toxc --file comments.csv
```

**JSON output**
```bash
toxc "some text" --json | jq .toxicity
```

**Fast mode** (VADER only, no model download)
```bash
toxc "some text" --fast
```

---

## Voice & Video Analysis

Transcribe any audio or video file and get per-sentence toxicity scores, a terminal summary, and an interactive HTML report.

```bash
toxc voice interview.mp4 --html report.html
```

```
╭──────────────────── toxc voice ────────────────────╮
│                                                    │
│  ██░░░░░░░░  Toxicity   0.12  CLEAN                │
│  ████░░░░░░  Sentiment  +0.34  Positive            │
│                                                    │
│  Sentences  3 toxic / 24 total                     │
│                                                    │
│  Top moments                                       │
│  1:42  0.84  You are completely wrong about…       │
│  3:05  0.71  This is utterly ridiculous and…       │
│                                                    │
│  Verdict: Clean — voice analysis                   │
│                                                    │
╰────────────────────────────────────────────────────╯
│
Report saved → report.html
```

### HTML Report

The `--html` flag generates a full interactive dashboard:

| Panel | Contents |
|---|---|
| Sticky sidebar | Overall score, dimension mini-bars, stats, section nav, top moment links |
| Timeline | Proportional bar chart — each segment colored and scaled by toxicity score |
| Analysis | Dual-axis line chart (toxicity + sentiment over time) · Score distribution histogram |
| Dimension heatmap | 5 sub-dimensions × every sentence — hover to inspect, click to jump |
| Top 5 moment cards | Score, timestamp, verdict, dimension chips |
| All sentences | Full table with toxicity bar, sentiment, and verdict |

Light/dark theme toggle (defaults dark), scroll-to-sentence cross-linking from every component.

### Options

```
toxc voice AUDIO [OPTIONS]

Arguments:
  AUDIO           Path to audio or video file (mp3, mp4, wav, m4a, …)

Options:
  --html PATH     Save interactive HTML report to path
  -m, --model     Whisper model: tiny | base | small | medium | large  [default: small]
  --fast          Use VADER only (skip Detoxify)
  --json          Output full analysis as JSON
```

### Whisper model guide

| Model | VRAM | Speed | Best for |
|---|---|---|---|
| `tiny` | ~1 GB | Fastest | Quick checks, clear audio |
| `base` | ~1 GB | Fast | Good general baseline |
| `small` | ~2 GB | Balanced | **Default** — good accuracy |
| `medium` | ~5 GB | Slow | Accented speech, mixed languages |
| `large` | ~10 GB | Slowest | Max accuracy |

---

## How it works

```
Audio/Video
    │
    ▼
Whisper (transcription + word timestamps)
    │
    ▼
NLTK sentence segmentation → timed sentence list
    │
    ▼
VADER  ──► sentiment score per sentence
Detoxify ► toxicity + 5 sub-dimensions per sentence
    │
    ▼
Aggregate (length-weighted averages, top moments, peaks by dim)
    │
    ├── Terminal summary (Rich)
    ├── HTML report (interactive dashboard)
    └── JSON (--json flag)
```

**Models used:**

| Model | Role | Speed |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech-to-text with word timestamps | Depends on model size |
| [VADER](https://github.com/cjhutto/vaderSentiment) | Sentiment scoring | Instant, offline |
| [Detoxify](https://github.com/unitaryai/detoxify) | Toxicity + 5 sub-dimensions | ~1s first run (downloads DistilBERT) |

**Toxicity dimensions:** `insult` · `obscene` · `threat` · `identity_attack` · `severe_toxicity`

Use `--fast` to skip Detoxify and run VADER only — good for quick checks or CI pipelines.

---

## Install

```bash
pip install toxc
```

Requires Python 3.9+.

- First text analysis run downloads the DistilBERT model (~250 MB, cached after)
- Voice analysis requires `ffmpeg` installed on your system

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg
```

---

## All Options

```
toxc [TEXT] [OPTIONS]          — analyze text
toxc voice AUDIO [OPTIONS]     — analyze audio/video

Text options:
  TEXT            Text to analyze (or omit for pipe/file input)
  -f, --file      Path to file with one text per line
  --json          Output as JSON
  --fast          Use VADER only (no Detoxify)

Voice options:
  AUDIO           Audio or video file path
  --html PATH     Save HTML report
  -m, --model     Whisper model size  [default: small]
  --fast          Use VADER only
  --json          Output full JSON analysis
```

---

## Built with

- [OpenAI Whisper](https://github.com/openai/whisper) — speech-to-text transcription
- [Detoxify](https://github.com/unitaryai/detoxify) — DistilBERT toxicity classifier
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment
- [NLTK](https://www.nltk.org/) — sentence segmentation
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [Rich](https://github.com/Textualize/rich) — terminal formatting

---

## Background

Built on research from a comparative toxicity study across Twitter/X and Bluesky (~19k posts). The short finding: platform culture shapes toxicity patterns more than moderation rules do.

---

MIT License · [@henokytilahun](https://henokytilahun.com)
