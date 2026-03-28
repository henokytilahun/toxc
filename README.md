# toxc
[![DevProof Score](https://orenda.vision/api/badge/score/henokytilahun/toxc)](https://orenda.vision/score/henokytilahun/toxc)

**Pre-upload YouTube ad safety analysis — powered by local LLM, not guesswork.**

Detoxify scores sentences in isolation. It has no concept of sarcasm, sports metaphors, or compliments. *"He absolutely destroys this argument"* scores 0.71 threat. *"You're killing it"* scores 0.68 toxicity. A creator sees those flags, loses trust in the tool, and uploads blind anyway.

toxc fixes that. It transcribes your video with Whisper, runs every sentence through a toxicity model, then sends flagged sentences to a local Ollama LLM with context — so *"destroying an argument"* gets cleared as a competitive metaphor, not a threat. The result is an interactive HTML report that tells you exactly what your video will cost you if you upload it now, and the two-word edits that fix it.

Built on a comparative toxicity study across ~19k posts on Twitter/X and Bluesky. The finding that shaped this tool: rule-based toxicity models fail systematically on platform-specific speech patterns. Context is everything.

```bash
pip install toxc
```

---

## The quickest win: check your title first

Before you even transcribe anything, run your title and thumbnail text through the ad safety checker. Titles are held to stricter standards than video content — and this takes two seconds.

```bash
toxc check "Why Mike Israetel Needs To Be STOPPED"
```
```
╭──────────────── toxc check ────────────────╮
│                                            │
│  YouTube Risk   CAUTION                    │
│                                            │
│  ████████░░  Toxicity    0.61              │
│  ████░░░░░░  Profanity   0.38              │
│  ██░░░░░░░░  Insult      0.21              │
│                                            │
│  May trigger limited ads — rephrase if     │
│  possible                                  │
│                                            │
╰────────────────────────────────────────────╯
```

Safe title, nothing to fix:
```bash
toxc check "Why Mike Israetel Is Wrong About Training Volume"
```
```
  YouTube Risk   SAFE
  Toxicity       0.04   ✓ Safe for title and thumbnail use
```

---

## Voice & video analysis

```bash
# Local file
toxc voice interview.mp4 --html report.html

# YouTube URL — no download needed
toxc voice "https://youtu.be/dQw4w9WgXcQ" --html report.html
```

On first run, toxc prompts for your channel profile (30 seconds). This is what turns a toxicity score into a revenue number. Skip with `--no-profile`.

**What the report shows:**

```
Toxicity    0.15   Clean
Ad Safety   LOW    Full monetization likely

Flagged moments
  1:43  0.61  "shut the hell up"
  3:22  0.48  "this is completely brutal"

Financial Impact
  Full monetization     $562
  Limited ads (yellow)  $168   ← ~70% revenue loss
  Demonetized             $0   ← current risk ◀
```

---

## Text analysis

```bash
toxc "you're such an idiot"
```
```
╭────────────────── toxc ──────────────────╮
│                                          │
│  ████████░░  Toxicity   0.81  TOXIC      │
│  ██░░░░░░░░  Sentiment  -0.51 Negative   │
│                                          │
│  ██░░░░░░  Severe      0.18              │
│  ███░░░░░  Obscene     0.31              │
│  █░░░░░░░  Threat      0.09              │
│  ████░░░░  Insult      0.62              │
│                                          │
│  Verdict: Toxic — personal attack        │
│                                          │
╰──────────────────────────────────────────╯
```

```bash
cat comments.txt | toxc          # pipe / batch
toxc --file comments.csv         # file input
toxc "some text" --json          # JSON output
toxc "some text" --fast          # VADER only, no model needed
```

---

## LLM passes

Three optional passes layer on top of the rule-based model, each requiring Ollama running locally. They're independent — use one, two, or all three.

### Pass 2 — Context check (`--context-check`)

Every sentence above 0.35 toxicity is sent to the LLM with one sentence of context on each side. Returns: genuine harm vs false positive, intent label, adjusted score, and two ad-safe rewrites that preserve your voice.

```bash
toxc voice myvideo.mp4 --context-check --html report.html
```

```
✓ Context check ran with llama3.2 · 5 false positives cleared of 7 flagged · 2 confirmed genuine

⚠ 2 confirmed flags
  1:43  "shut the hell up"  harmful
        Direct insult with profanity
        Safe:   "come on, be serious"
        Alt:    "that's genuinely ridiculous"

✓ 5 cleared
  0:23  "you're absolutely killing it"   Was 0.71 → 0.05  Hyperbolic praise
  1:12  "this is completely brutal"      Was 0.58 → 0.09  Means incisive, not violent
```

The adjusted scores flow back into the composite — if context check clears most flags, the headline risk level drops accordingly.

### Pass 3 — Policy review (`--policy-review`)

The LLM reads your **full transcript** alongside YouTube's actual Advertiser-Friendly Content Guidelines and renders a holistic verdict. This catches things the sentence-level model can't: a video can have zero toxic sentences and still get demonetized for overall topic sensitivity.

```bash
toxc voice myvideo.mp4 --policy-review --html report.html
```

```
● Limited ads likely   87% confidence   Edit before publishing

"Largely clean commentary. Main risk is the title — 'STOPPED' in all caps
reads as confrontational to YouTube's classifiers. Content itself is fine."

⚠ 1 policy flag
  Harassment / Cyberbullying   title   medium
  "Why Mike Israetel Needs To Be STOPPED"
  → Safe: "Why Mike Israetel Is Wrong About This"
  → Conservative: "Responding to Mike Israetel"
```

### Full pipeline

```bash
toxc voice myvideo.mp4 --context-check --policy-review --html report.html
```

When `--context-check` runs first, `--policy-review` receives the de-false-positived scores and intent labels — the LLM's verdict is correspondingly sharper.

### Setting up Ollama

```bash
brew install ollama              # macOS — or https://ollama.com/download
ollama pull llama3.2             # ~2 GB, recommended
pip install ollama               # Python client
```

| Model | Size | Notes |
|---|---|---|
| `llama3.2` | ~2 GB | **Default** — fast, accurate |
| `mistral` | ~4 GB | Strong reasoning |
| `gemma2` | ~5 GB | Good at nuance |

If Ollama isn't running, LLM passes are skipped with a warning — the report still generates using Detoxify scores.

---

## Speaker diarization *(optional)*

Identifies who said what. Useful for podcasts, interviews, and reaction content where the host's audio is clean but the guest's or source clip isn't.

```bash
pip install "toxc[diarize]"
toxc voice interview.mp4 --diarize --html report.html
```

The report adds a per-speaker breakdown:

```
── Speaker Analysis ─────────────────────────────

  Speaker       Avg Toxicity   Sentences   Flagged   Verdict
  SPEAKER_00    0.04           42          0         Clean
  SPEAKER_01    0.31           31          3         Review

⚠ Riskiest: SPEAKER_01 · peak at "you don't know what you're talking about…"
```

The transcript tab gains a Speaker column. The LLM policy review sees `[SPEAKER_XX]` prefixes on every line, which sharpens intent disambiguation — the model can tell that SPEAKER_00 calling SPEAKER_01's argument "brutal" is a compliment, not a threat.

**One-time setup** (free, but requires a HuggingFace account):
1. Accept the model license at [hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
2. Generate a token at [hf.co/settings/tokens](https://hf.co/settings/tokens)
3. `toxc config set --hf-token hf_xxxxxxxxxxxx`

Diarization is fully optional — toxc works without it.

---

## Channel profile & financial impact

```
  Monthly views      › 125000
  Subscribers        › 48000
  Avg CPM ($)        › 4.50
  Videos per month   › 4
  Content category   › gaming / commentary / education / news / kids / other
  Past strikes?      › none / 1 / 2+
```

Saved after first run. Enables:
- **Per-video revenue scenarios** — full / limited / demonetized, with current risk highlighted
- **Annual impact** — projects the per-video loss across your upload schedule
- **Category-aware thresholds** — a gaming channel saying "idiot" is scored differently from a kids' channel
- **Strike escalation warnings** — if you have a strike, the report tells you exactly what a second one means

```bash
toxc config show                                  # view profile
toxc config set --cpm 4.50 --subscribers 48000    # update fields
toxc config set --hf-token hf_xxxx                # save HF token
toxc config setup                                 # re-run interactive
toxc config reset                                 # delete profile
```

---

## Install

```bash
pip install toxc                    # text analysis, no downloads
pip install "toxc[voice]"           # + Whisper, NLTK, yt-dlp
pip install "toxc[diarize]"         # + pyannote speaker diarization
pip install ollama && ollama pull llama3.2  # + LLM passes
```

Requires Python 3.9+. Voice analysis requires `ffmpeg`:

```bash
brew install ffmpeg      # macOS
apt install ffmpeg       # Ubuntu/Debian
```

First text run downloads the DistilBERT model (~250 MB, cached after).

### Whisper model guide

| Model | VRAM | Speed | Best for |
|---|---|---|---|
| `tiny` | ~1 GB | Fastest | Quick checks, clear audio |
| `base` | ~1 GB | Fast | Good general baseline |
| `small` | ~2 GB | Balanced | **Default** |
| `medium` | ~5 GB | Slow | Accented speech |
| `large` | ~10 GB | Slowest | Max accuracy |

---

## How it works

```
Audio/Video  ──or──  YouTube URL
      │                    │
      │               yt-dlp (audio only)
      └────────────────────┘
      │
      ▼
Whisper  ──► words + timestamps
pyannote ──► speaker segments     (optional, --diarize)
      │
      ▼
NLTK sentence segmentation → timed sentence list
      │
      ▼
VADER    ──► sentiment per sentence
Detoxify ──► toxicity + 5 sub-dimensions per sentence
      │
      ▼
Pass 2 (optional, --context-check)
Ollama ──► genuine harm vs. false positive
          ├─ intent label
          ├─ adjusted score
          └─ two ad-safe rewrites
      │
      ▼
Pass 3 (optional, --policy-review)
Ollama ──► full transcript + YouTube guidelines
          ├─ holistic monetization verdict
          ├─ specific policy violations
          ├─ title risk
          └─ voice-preserving rewrites
      │
      ▼
Composite score  density(25%) + peak-5%(50%) + rate(25%)
Monetization risk  Tier 1: hate/threats/severe → demonetized
                   Tier 2: profanity → limited ads
                   Tier 3: density / first-7s / category
      │
      ├── Terminal summary
      ├── HTML report (interactive dashboard)
      └── JSON (--json)
```

| Model | Role |
|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) | Speech-to-text with word timestamps |
| [VADER](https://github.com/cjhutto/vaderSentiment) | Sentiment scoring — instant, offline |
| [Detoxify](https://github.com/unitaryai/detoxify) | Toxicity + 5 sub-dimensions |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization *(optional)* |
| [Ollama](https://ollama.com) | Local LLM — context check + policy review *(optional)* |

---

## All commands

```
toxc [TEXT]                     analyze text (or pipe input)
toxc voice SOURCE               analyze audio/video or YouTube URL
toxc check TEXT                 quick YouTube title/thumbnail safety check
toxc config show/setup/reset    manage channel profile

Text:    --file, --json, --fast
Voice:   --html, --model, --fast, --no-profile, --json
         --context-check (-cc), --policy-review (-pr), --ollama-model
         --diarize (-d), --min-speakers, --max-speakers
Config:  --channel-name, --monthly-views, --subscribers, --cpm,
         --videos-per-month, --category, --past-strikes, --hf-token
```

---

## Roadmap

**Near-term**
- `--cloud` flag — Claude API instead of Ollama, no local GPU required
- Thumbnail safety check — `toxc check` extended to image files
- CSV/JSON export button in the HTML report

**Mid-term**
- `toxc channel @handle --last 20` — bulk scan of recent uploads
- Watch mode — poll for new uploads and run automatically
- Multi-language rewrites — Whisper already transcribes 90+ languages

**Longer-term**
- Browser extension — risk badge inline on any YouTube page
- REST API — self-hostable HTTP wrapper for MCN/CMS integrations
- Appeal letter generator — draft a YouTube appeals submission from the context-check evidence

---

## Built with

[OpenAI Whisper](https://github.com/openai/whisper) · [Detoxify](https://github.com/unitaryai/detoxify) · [VADER](https://github.com/cjhutto/vaderSentiment) · [NLTK](https://www.nltk.org/) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [pyannote.audio](https://github.com/pyannote/pyannote-audio) · [Ollama](https://ollama.com) · [Typer](https://typer.tiangolo.com/) · [Rich](https://github.com/Textualize/rich)

---

MIT License · [@henokytilahun](https://henokytilahun.com)
