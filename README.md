# toxc

**Fast CLI toxicity & sentiment analysis for text.**

```bash
pip install toxc
```

---

## Usage

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

**Pipe input (batch mode)**
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

**JSON output** (pipe into `jq`, scripts, etc.)
```bash
toxc "some text" --json | jq .toxicity
```

**Fast mode** (VADER only, no model download)
```bash
toxc "some text" --fast
```

---

## How it works

`toxc` combines two models:

| Model | Role | Speed |
|---|---|---|
| [VADER](https://github.com/cjhutto/vaderSentiment) | Sentiment scoring | Instant, offline |
| [Detoxify](https://github.com/unitaryai/detoxify) | Toxicity + 5 sub-dimensions | ~1s first run (downloads DistilBERT) |

Toxicity dimensions: `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`

Use `--fast` to skip Detoxify and run VADER only — good for quick checks or CI pipelines.

---

## Install

```bash
pip install toxc
```

Requires Python 3.9+. First run downloads the DistilBERT model (~250MB, cached after).

---

## Options

```
toxc [TEXT] [OPTIONS]

Arguments:
  TEXT          Text to analyze (optional if using --file or pipe)

Options:
  -f, --file    Path to file with one text per line
  --json        Output as JSON
  --fast        Use VADER only (no Detoxify)
  --help        Show this message and exit
```

---

## Built with

- [Detoxify](https://github.com/unitaryai/detoxify) — DistilBERT toxicity classifier
- [VADER](https://github.com/cjhutto/vaderSentiment) — rule-based sentiment
- [Typer](https://typer.tiangolo.com/) — CLI framework
- [Rich](https://github.com/Textualize/rich) — terminal formatting

---

## Background

Built on research from a comparative toxicity study across Twitter/X and Bluesky (~19k posts). The short finding: platform culture shapes toxicity patterns more than moderation rules do.

---

MIT License · [@henokytilahun](https://henokytilahun.com)
