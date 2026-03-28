# toxc
[![DevProof Score](https://orenda.vision/api/badge/score/henokytilahun/toxc)](https://orenda.vision/score/henokytilahun/toxc)

**CLI toxicity & sentiment analysis for text, audio, and video — with YouTube ad safety scoring, speaker diarization, and local LLM policy review.**

Analyze a string, a batch of lines, or a full video. toxc transcribes speech with Whisper, scores every sentence for toxicity and sentiment, maps the results to YouTube's advertiser-friendliness tiers, and renders an interactive HTML report that tells you exactly what a video will cost you if you upload it now — and the edits that fix it.

Three optional LLM passes layer on top of the rule-based model:

- **Pass 2 (`--context-check`)** — Ollama verifies each flagged sentence in context, separating genuine harm from hyperbole and sarcasm, then generates ad-safe rewrites that preserve your voice.
- **Pass 3 (`--policy-review`)** — Ollama reads your full transcript against YouTube's actual Advertiser-Friendly Content Guidelines and renders a holistic verdict: overall monetization risk, specific policy violations, title safety, and voice-preserving rewrites.
- **Speaker diarization (`--diarize`)** — pyannote identifies who said what, so per-speaker toxicity breakdowns tell you whether it's the host, the guest, or a clip being played back that's causing the flag.

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

Transcribe any audio or video file — or pass a YouTube URL directly — and get per-sentence toxicity scores, a terminal summary, and an interactive HTML report.

```bash
# Local file
toxc voice interview.mp4 --html report.html

# YouTube URL (no download needed)
toxc voice "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --html report.html

# YouTube Shorts or youtu.be links also work
toxc voice "https://youtu.be/dQw4w9WgXcQ" --html report.html
```

On first run, toxc will prompt for your channel profile (takes ~30 seconds). This unlocks financial impact estimates in the report. Skip with `--no-profile`.

---

## Context Check — Local LLM False-Positive Detection

Detoxify scores sentences in isolation — it has no concept of sarcasm, sports metaphors, hyperbole, or compliments. A fitness creator saying *"he absolutely destroys this argument"* scores 0.71 threat. A gaming channel saying *"you're killing it"* scores 0.68 toxicity. These are false positives, and one bad flag is enough for a creator to distrust the whole tool.

The `--context-check` flag runs a second pass through a local Ollama LLM. Every sentence that scored above 0.35 is sent to the model with one sentence of context on each side. The LLM returns:

- Whether the sentence is **genuine harm** or a **false positive**
- The **intent** (compliment, hyperbole, sarcasm, criticism, harmful, ...)
- An **adjusted score** reflecting context
- A **reason** in plain English
- Two **ad-safe rewrites** that preserve the creator's voice and energy

```bash
# Enable context check (requires Ollama running locally)
toxc voice myvideo.mp4 --html report.html --context-check

# Use a specific model
toxc voice myvideo.mp4 --html report.html --context-check --ollama-model mistral
```

**Example report output (What to Fix section):**
```
✓ Context check ran with llama3.2 · 5 false positives cleared of 7 flagged · 2 confirmed genuine

⚠ 2 CONFIRMED FLAGS
  1:43  "shut the hell up" → saves ~$394/video  harmful
        Direct insult with profanity — genuine flag
        Safe:   "come on, be serious"
        Alt:    "that's genuinely ridiculous"

✓ 5 CLEARED BY CONTEXT CHECK
  0:23  "you're absolutely killing it"  Compliment
        Was 0.71 → adjusted 0.05 · Hyperbolic praise — clearly a compliment in context
  1:12  "this is completely brutal"  Criticism
        Was 0.58 → adjusted 0.09 · Brutal means incisive/thorough, not violent
  ...
```

The adjusted scores flow through to the overall toxicity composite and risk level — so if context check clears most flags, the headline risk level drops accordingly.

---

## AI Policy Review — LLM Holistic Verdict

The `--policy-review` flag runs a third pass where Ollama reads the **full transcript** alongside YouTube's actual Advertiser-Friendly Content Guidelines and renders a verdict the same way a human reviewer would.

This catches things rule-based models fundamentally can't:

- **Policy-aware judgment** — the LLM reads the actual guidelines, not a proxy score
- **Intent across the full narrative arc** — a video can have zero toxic sentences and still be demonetized for overall topic sensitivity
- **Title risk assessment** — titles are held to stricter standards than video content; the LLM evaluates yours specifically
- **Voice-preserving rewrites** — two options per flag: same energy (ad-safe) and a more conservative fallback

```bash
toxc voice myvideo.mp4 --policy-review --html report.html

# Combine all three passes for maximum accuracy
toxc voice myvideo.mp4 --context-check --policy-review --diarize --html report.html

# Full analysis on a YouTube URL with speaker hints
toxc voice "https://youtu.be/dQw4w9WgXcQ" --context-check --policy-review --diarize --min-speakers 2 --html report.html
```

When `--context-check` has run first, the policy review receives the adjusted (de-false-positived) scores and Pass 2 intent labels — the LLM's verdict is correspondingly sharper.

**Example report output (AI Policy Review section):**
```
● Limited ads likely   87% confidence   Edit before publishing

"This video is largely clean commentary. The main risk is the title — 'STOPPED'
in all caps combined with a named individual reads as confrontational to YouTube's
classifiers. The content itself is fine."

⚠ 2 policy flags
  Harassment/Cyberbullying   1:12   medium
  "Why Mike needs to be stopped"
  → Safe: "Why Mike's take doesn't hold up"
  → Conservative: "The problem with Mike's argument"

  Clickbait / confrontational framing   title   medium
  "Why Mike Israetel Needs To Be STOPPED"
  → Safe: "Why Mike Israetel Is Wrong About This"
  → Conservative: "Responding to Mike Israetel"

✓ 5 cleared as false positives
  "you're destroying this argument"
  "insane value in this video"
  ...
```

---

## Speaker Diarization — Who Said What

The `--diarize` flag runs pyannote speaker diarization alongside Whisper transcription, assigning a speaker label to every sentence. The report then shows per-speaker toxicity breakdowns:

```
── Speaker Analysis ──────────────────────────────
  Speaker       Avg Toxicity   Sentences   Flagged   Verdict
  SPEAKER_00    0.04           42          0         Clean
  SPEAKER_01    0.31           31          3         Review

⚠ Riskiest speaker: SPEAKER_01
  · peak at "you don't know what you're talking about…"
```

This matters for three common creator formats:

- **Podcast / interview** — the guest flags, the host is clean. You know exactly who to ask to re-record a line.
- **Commentary over clips** — your voice is clean; the source audio isn't. Very different monetization risk.
- **Debate / two-person video** — per-speaker breakdown tells you which side of the conversation needs edits.

Speaker labels also flow into the AI Policy Review transcript — the LLM sees `[SPEAKER_00]` and `[SPEAKER_01]` prefixes on every line, which sharpens intent disambiguation significantly.

### Setup

Diarization requires a free HuggingFace token and a one-time model license acceptance:

```bash
# 1. Install the dependency
pip install "toxc[diarize]"

# 2. Accept the model license (free, one-time)
#    https://hf.co/pyannote/speaker-diarization-3.1

# 3. Generate a HuggingFace token
#    https://hf.co/settings/tokens

# 4. Save your token
toxc config set --hf-token hf_xxxxxxxxxxxx
```

After that:
```bash
toxc voice interview.mp4 --diarize --html report.html

# Hint expected speaker count (helps when pyannote merges similar voices)
toxc voice podcast.mp4 --diarize --min-speakers 2 --html report.html
toxc voice panel.mp4 --diarize --min-speakers 3 --max-speakers 5 --html report.html
```

After diarization completes, toxc prints a summary of detected speakers and their speaking time so you can verify the results at a glance.

If the token is missing or pyannote fails, toxc continues without speaker labels and prints a warning — the report still generates normally.

---

## Channel Profile & Financial Impact

The first time you run `toxc voice`, you'll see:

```
  Before we analyze, tell us about your channel:
  (Press Enter to skip any field — estimates will be used)

  Monthly views      › 125000
  Subscribers        › 48000
  Avg CPM ($)        › 4.50
  Videos per month   › 4
  Content category   › [gaming / commentary / education / news / kids / other]
  Past strikes?      › [None / 1 / 2+]

  Save profile for future runs? [y/n]: y
```

This transforms the report from "here's what's toxic" into "here's exactly what this video will cost you":

```
── Financial Impact ─────────────────────────────────────
  Full monetization     $562   ← where you want to be
  Limited ads (yellow)  $168   ← ~70% revenue loss
  Demonetized             $0   ← current risk ◀

  Revenue at risk / video    $562
  Annual impact if pattern   -$9,516/yr
```

**Category-aware thresholds** — a gaming channel saying "idiot" is scored differently from a kids' channel. toxc adjusts its risk thresholds based on your content category.

**Strike escalation warnings** — if you already have a strike, the report flags exactly what a second one means:

```
⚠ A second strike means a 2-week upload freeze, loss of all monetization during
  that period, and leaves you one strike from permanent termination.
```

### Managing your profile

```bash
toxc config show                                 # view saved profile
toxc config set --cpm 4.50 --subscribers 48000   # update individual fields
toxc config set --hf-token hf_xxxx               # save HuggingFace token
toxc config setup                                # re-run interactive setup
toxc config reset                                # delete profile
```

---

## HTML Report

The `--html` flag generates a full interactive dashboard organized into three tabs:

### Overview tab

| Panel | Contents |
|---|---|
| Hero cards | Toxicity score · Ad safety · Flagged sentences · Avg sentiment · Context cleared (when `--context-check` ran) |
| Speaker Analysis | Per-speaker toxicity table with avg score, sentence count, flagged count, verdict · riskiest speaker callout *(when `--diarize` ran)* |
| Channel Risk Profile | Channel size, category, strike status, category-specific note, escalation warning |
| Financial Impact | Three-tier revenue scenario (full / limited / demonetized) with current risk highlighted, revenue-at-risk and annual impact rows |
| Consequence Breakdown | Monetization status · Strike risk · Age restriction · Advertiser opt-out risk — all color-coded by severity |
| AI Policy Review | LLM verdict badge · confidence · publish recommendation · overall summary · policy violations with timestamps and rewrites · title risk · false positives cleared *(when `--policy-review` ran)* |
| What to Fix | Confirmed genuine flags with intent label, LLM reason, dollar savings, and two copyable rewrites · Cleared false positives with original vs. adjusted score |
| Risk Signals | Tiered by YouTube policy: Instant demonetization (identity attacks, threats, severe) · Limited ads (profanity) · Pattern signals (density, first-7s) — each with raw score |

### Visualizations tab *(collapsible sections)*

| Panel | Contents |
|---|---|
| Timeline | Proportional bar chart — each segment colored and scaled by toxicity, first-7-seconds zone highlighted |
| Analysis | Dual-axis Catmull-Rom line chart (toxicity + sentiment over time) · Score distribution histogram |
| Dimension Heatmap | 5 sub-dimensions × every sentence — hover to inspect, click to jump |
| Top Moments | Adjusted score, raw score, timestamp, verdict, dimension chips, context badge, rewrite preview |

### Transcript tab

Full sentence table with sticky headers, zebra striping, and — when `--context-check` ran — a Context column showing ✓ Cleared (with intent) or ⚠ Confirmed per row, plus before/after score comparison.

Light/dark theme toggle (defaults dark, persists via localStorage), sidebar nav switches tabs and scrolls simultaneously.

### Voice options

```
toxc voice SOURCE [OPTIONS]

Arguments:
  SOURCE              Audio/video file path OR a YouTube URL

Options:
  --html PATH         Save interactive HTML report to path
  -m, --model         Whisper model: tiny | base | small | medium | large  [default: small]
  --fast              Use VADER only (skip Detoxify)
  --no-profile        Skip channel profile prompts
  -cc, --context-check  Run local LLM (Ollama) to verify flagged sentences and generate safe rewrites
  -pr, --policy-review  Run local LLM (Ollama) for a full YouTube policy review
  --ollama-model      Ollama model to use for context check / policy review  [default: llama3.2]
  -d, --diarize       Enable speaker diarization via pyannote (requires HuggingFace token)
  --min-speakers N    Minimum expected speakers for diarization (e.g. 2 for a podcast)
  --max-speakers N    Maximum expected speakers for diarization
  --json              Output full analysis as JSON
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
Audio/Video file  ─OR─  YouTube URL
    │                        │
    │                   yt-dlp (audio only, temp file)
    │                        │
    └────────────────────────┘
    │
    ▼
Whisper (transcription + word timestamps)
    │
    ├── pyannote (optional, --diarize)
    │   └─ speaker segments → assign SPEAKER_XX to each sentence
    │
    ▼
NLTK sentence segmentation → timed sentence list
    │
    ▼
VADER    ──► sentiment score per sentence
Detoxify ──► toxicity + 5 sub-dimensions per sentence
    │
    ▼
Pass 2 (optional, --context-check)
Ollama ──► genuine harm vs. false positive per flagged sentence
           ├─ intent (compliment / hyperbole / sarcasm / harmful / ...)
           ├─ adjusted toxicity score
           └─ two ad-safe rewrites preserving creator voice
    │
    ▼
Pass 3 (optional, --policy-review)
Ollama ──► reads full transcript + YouTube Advertiser-Friendly Content Guidelines
           ├─ holistic monetization verdict (green / yellow / red)
           ├─ specific policy violations with timestamps
           ├─ title risk assessment
           ├─ false positives cleared at the narrative level
           └─ voice-preserving rewrites for confirmed flags
    │
    ▼
Per-speaker aggregation (if diarization ran)
  avg toxicity · flagged count · peak sentence per speaker
    │
    ▼
Composite toxicity score (using adjusted scores when available)
  density (25%) + peak-of-top-5% (50%) + toxic-rate (25%)
    │
    ▼
Monetization risk assessment (category-aware thresholds)
  Tier 1: identity attacks / threats / severe → instant demonetization
  Tier 2: profanity → limited ads (yellow icon)
  Tier 3: density pattern / first-7s / category
    │
    ▼
Financial impact calculations (if channel profile present)
    │
    ├── Terminal summary (Rich)
    ├── HTML report (tabbed interactive dashboard)
    └── JSON (--json flag)
```

**Models used:**

| Model | Role | Speed |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech-to-text with word timestamps | Depends on model size |
| [VADER](https://github.com/cjhutto/vaderSentiment) | Sentiment scoring | Instant, offline |
| [Detoxify](https://github.com/unitaryai/detoxify) | Toxicity + 5 sub-dimensions | ~1s first run (downloads DistilBERT) |
| [Ollama](https://ollama.com) *(optional)* | Pass 2 context check + Pass 3 policy review | ~1–3s/sentence locally |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) *(optional)* | Speaker diarization | ~30–60s per video |

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

### Voice dependencies

```bash
pip install "toxc[voice]"
# installs: openai-whisper, nltk, yt-dlp
```

### LLM passes (context check + policy review)

```bash
pip install ollama
brew install ollama   # or https://ollama.com/download
ollama pull llama3.2
```

If Ollama is not running or not installed, `--context-check` and `--policy-review` are silently skipped with a warning — the report still generates normally using Detoxify scores.

**Supported models** — any model pulled via `ollama pull`:

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | ~2 GB | **Default** — fast, accurate |
| `mistral` | ~4 GB | Strong reasoning |
| `gemma2` | ~5 GB | Good at nuance |
| `phi3` | ~2 GB | Lightweight alternative |

### Speaker diarization

```bash
pip install "toxc[diarize]"
# installs: pyannote.audio

# One-time setup
# 1. Accept model license: https://hf.co/pyannote/speaker-diarization-3.1
# 2. Generate token: https://hf.co/settings/tokens
toxc config set --hf-token hf_xxxxxxxxxxxx
```

---

## All Commands

```
toxc [TEXT] [OPTIONS]           analyze text
toxc voice SOURCE [OPTIONS]     analyze audio/video
toxc check TEXT [OPTIONS]       quick YouTube title/thumbnail safety check
toxc config show                view saved channel profile
toxc config set [OPTIONS]       update profile fields
toxc config setup               run interactive setup
toxc config reset               delete profile

Text options:
  TEXT                Text to analyze (or omit for pipe/file input)
  -f, --file          Path to file with one text per line
  --json              Output as JSON
  --fast              Use VADER only (no Detoxify)

Voice options:
  SOURCE              Audio/video file path or YouTube URL
  --html PATH         Save HTML report
  -m, --model         Whisper model size  [default: small]
  --fast              Use VADER only
  --no-profile        Skip channel profile prompts
  -cc, --context-check  Verify flagged sentences with local Ollama LLM
  -pr, --policy-review  Full YouTube policy review with local Ollama LLM
  --ollama-model      Ollama model to use  [default: llama3.2]
  -d, --diarize       Speaker diarization via pyannote
  --min-speakers N    Minimum expected speakers (helps separate similar voices)
  --max-speakers N    Maximum expected speakers
  --json              Output full JSON analysis

Config set options:
  --channel-name      Channel display name
  --monthly-views     Monthly view count
  --subscribers       Subscriber count
  --cpm               Average CPM in dollars
  --videos-per-month  Videos published per month
  --category          gaming / commentary / education / news / kids / other
  --past-strikes      0, 1, or 2
  --hf-token          HuggingFace token for speaker diarization
```

---

## Built with

- [OpenAI Whisper](https://github.com/openai/whisper) — speech-to-text transcription
- [Detoxify](https://github.com/unitaryai/detoxify) — DistilBERT toxicity classifier
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment
- [NLTK](https://www.nltk.org/) — sentence segmentation
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube audio download
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization *(optional)*
- [Ollama](https://ollama.com) — local LLM inference for context check and policy review *(optional)*
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [Rich](https://github.com/Textualize/rich) — terminal formatting

---

## Roadmap

### Near-term

- **`--cloud` flag** — use the Claude API instead of Ollama for LLM passes. No local GPU required, better reasoning on edge cases, pay-per-use. Local stays the default.
- **Progress bar for context check** — show per-sentence progress when checking long videos (currently shows a spinner)
- **Thumbnail safety check** — extend `toxc check` to accept an image path or URL and flag text/imagery that violates YouTube's thumbnail policies
- **CSV / JSON export from the report** — download button in the HTML report that exports the full sentence table as a spreadsheet

### Mid-term

- **Bulk channel analysis** — `toxc channel @handle --last 20` to scan a creator's recent uploads and surface which videos are currently at monetization risk
- **Watch mode** — `toxc watch @handle` polls for new uploads and runs analysis automatically, good for teams reviewing content before it goes live
- **GitHub Action** — `toxc-action` to add a toxicity gate to a CI pipeline for podcast shows, corporate content teams, or automated dubbing workflows
- **Multi-language support** — Whisper already transcribes 90+ languages; extend thresholds and rewrite suggestions to Spanish, Portuguese, German, French
- **Segment-level editing hints** — instead of flagging a whole sentence, highlight the specific word or phrase that triggered the score (requires token-level attribution)

### Longer-term

- **Browser extension** — run toxc inline on any YouTube video page, show the risk badge next to the title without leaving the browser
- **TikTok / Twitch / podcast support** — platform-specific monetization policies differ from YouTube; model each one's advertiser-friendliness rules separately
- **REST API** — self-hostable HTTP wrapper around `toxc voice` for integration with video editing tools, CMS platforms, and MCN dashboards
- **Appeal letter generator** — when a video is already demonetized, use the context check data to draft a YouTube appeals submission with specific sentence-level evidence

---

## Background

Built on research from a comparative toxicity study across Twitter/X and Bluesky (~19k posts). The short finding: platform culture shapes toxicity patterns more than moderation rules do.

---

MIT License · [@henokytilahun](https://henokytilahun.com)
