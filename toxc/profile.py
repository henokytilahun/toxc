import json
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt

console = Console()

PROFILE_PATH = Path.home() / ".config" / "toxc" / "profile.json"

CATEGORIES = ["gaming", "commentary", "education", "news", "kids", "other"]

# Typical CPM ranges by niche (mid-point estimates, USD)
# Sources: publicly reported creator data, Social Blade, various creator disclosures
CATEGORY_CPM_DEFAULTS = {
    "kids":       1.5,   # Low — COPPA restricted, limited advertiser pool
    "gaming":     3.0,   # Low-mid — large audience, lower purchase intent
    "commentary": 4.0,   # Mid — general audience
    "other":      3.5,   # Mid baseline
    "news":       6.0,   # Higher — engaged, affluent audience
    "education":  8.0,   # High — strong purchase intent, professional audience
}

# Per-category thresholds. The identity/threat/severe tiers are universally strict
# because hate speech and threats are brand-safety issues regardless of category.
# Obscene thresholds vary more — gaming/commentary audiences expect stronger language.
CATEGORY_THRESHOLDS = {
    #                  high_tox  med_tox  identity  threat  severe  obscene_high  obscene_med  flagged_rate
    "kids":       dict(high_tox=0.20, medium_tox=0.08, identity=0.04, threat=0.06, severe=0.08, obscene_high=0.15, obscene_medium=0.05, flagged_rate=0.03),
    "education":  dict(high_tox=0.45, medium_tox=0.18, identity=0.06, threat=0.10, severe=0.12, obscene_high=0.35, obscene_medium=0.10, flagged_rate=0.08),
    "news":       dict(high_tox=0.60, medium_tox=0.28, identity=0.08, threat=0.12, severe=0.15, obscene_high=0.55, obscene_medium=0.12, flagged_rate=0.15),
    "commentary": dict(high_tox=0.60, medium_tox=0.28, identity=0.08, threat=0.12, severe=0.15, obscene_high=0.55, obscene_medium=0.12, flagged_rate=0.15),
    "gaming":     dict(high_tox=0.68, medium_tox=0.32, identity=0.08, threat=0.12, severe=0.15, obscene_high=0.68, obscene_medium=0.18, flagged_rate=0.18),
    "other":      dict(high_tox=0.60, medium_tox=0.28, identity=0.08, threat=0.12, severe=0.15, obscene_high=0.55, obscene_medium=0.12, flagged_rate=0.15),
}

_SIZE_LABELS = [
    (1_000_000, "Large (1M+ subs)"),
    (100_000,   "Mid-large (100K+ subs)"),
    (10_000,    "Mid-tier (10K–100K subs)"),
    (1_000,     "Small (1K–10K subs)"),
    (1,         "Starter (<1K subs)"),
    (0,         "Unknown"),
]

_CATEGORY_NOTES = {
    "Kids":        "Strictest scrutiny — COPPA compliance required. Zero tolerance.",
    "Education":   "Higher standards — family-friendly content expected.",
    "Gaming":      "Moderate scrutiny — gaming language has more tolerance.",
    "Commentary":  "Standard YouTube advertiser scrutiny.",
    "News":        "Higher tolerance for difficult topics in news context.",
    "Other":       "Standard YouTube advertiser thresholds.",
}


def load_profile() -> dict | None:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except Exception:
            return None
    return None


def save_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


def prompt_for_profile() -> dict:
    """Interactive onboarding — returns a (possibly partial) profile dict."""
    console.print()
    console.print("[bold]  Before we analyze, tell us about your channel:[/bold]")
    console.print("  [dim](Press Enter to skip any field — estimates will be used)[/dim]")
    console.print()

    profile: dict = {}

    def ask_int(label: str) -> int | None:
        raw = Prompt.ask(f"  {label}", default="").strip().replace(",", "")
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

    def ask_float(label: str) -> float | None:
        raw = Prompt.ask(f"  {label}", default="").strip()
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    v = ask_int("Monthly views      ›")
    if v is not None:
        profile["monthly_views"] = v

    v = ask_int("Subscribers        ›")
    if v is not None:
        profile["subscribers"] = v

    v = ask_int("Videos per month   ›")
    if v is not None:
        profile["videos_per_month"] = v

    # Ask category first so we can show a sensible CPM default
    console.print("  [dim]  Content category options:[/dim]")
    cat_notes = {
        "gaming":      "gaming, let's plays, streams",
        "commentary":  "opinion, reaction, vlog",
        "education":   "tutorials, explainers, how-to",
        "news":        "news, current events, politics",
        "kids":        "children's content (COPPA applies)",
        "other":       "anything else",
    }
    for name, desc in cat_notes.items():
        console.print(f"  [dim]    {name:<14} {desc}[/dim]")

    while True:
        raw = Prompt.ask("  Content category   ›", default="").strip().lower()
        if not raw:
            profile["category"] = "other"
            break
        if raw in CATEGORIES:
            profile["category"] = raw
            break
        console.print(f"  [yellow]  \"{raw}\" isn't an option. Choose from: {', '.join(CATEGORIES)}[/yellow]")

    # CPM — show a category-based estimate so the user knows what's reasonable
    cat = profile.get("category", "other")
    cpm_default = CATEGORY_CPM_DEFAULTS.get(cat, 3.5)
    console.print(f"  [dim]  Typical CPM for {cat}: ~${cpm_default:.2f}[/dim]")
    console.print(f"  [dim]  Find yours: YouTube Studio → Analytics → Revenue → CPM[/dim]")
    v = ask_float(f"Avg CPM ($)        › [Enter for ~${cpm_default:.2f}]")
    if v is not None:
        profile["cpm"] = v
        profile["cpm_estimated"] = False
    else:
        profile["cpm"] = cpm_default
        profile["cpm_estimated"] = True

    # Past strikes — show what each means
    console.print("  [dim]  Strike status:[/dim]")
    console.print("  [dim]    none   clean record[/dim]")
    console.print("  [dim]    1      2-week freeze + demonetization if you get another[/dim]")
    console.print("  [dim]    2+     one strike from permanent termination[/dim]")
    while True:
        raw = Prompt.ask("  Past strikes?      › [none / 1 / 2+]", default="").strip().lower()
        strike_map = {"none": 0, "0": 0, "": 0, "1": 1, "2": 2, "2+": 2, "3": 2, "multiple": 2}
        if raw in strike_map:
            profile["past_strikes"] = strike_map[raw]
            break
        console.print(f"  [yellow]  Enter none, 1, or 2+[/yellow]")

    console.print()

    if profile:
        save_yn = Prompt.ask("  Save profile for future runs?", choices=["y", "n"], default="y")
        if save_yn == "y":
            save_profile(profile)
            console.print(f"  [dim]Saved → {PROFILE_PATH}[/dim]")
    console.print()

    return profile


def financial_impact(profile: dict, risk_level: str) -> dict:
    """Compute estimated revenue impact given channel profile and risk level."""
    monthly_views = profile.get("monthly_views")
    cpm = profile.get("cpm")

    if not monthly_views or not cpm:
        return {}

    videos_per_month = profile.get("videos_per_month") or 4
    views_per_video = monthly_views / videos_per_month
    revenue_full = views_per_video * (cpm / 1000)
    revenue_limited = revenue_full * 0.30  # limited ads ≈ 30 % of normal CPM

    if risk_level == "HIGH":
        projected = 0.0
        at_risk = revenue_full
    elif risk_level == "MEDIUM":
        projected = revenue_limited
        at_risk = revenue_full - revenue_limited
    else:
        projected = revenue_full
        at_risk = 0.0

    annual_impact = at_risk * videos_per_month * 12

    return {
        "revenue_per_video":   round(revenue_full, 2),
        "limited_ads_revenue": round(revenue_limited, 2),
        "projected_revenue":   round(projected, 2),
        "revenue_at_risk":     round(at_risk, 2),
        "annual_impact":       round(annual_impact, 2),
        "views_per_video":     round(views_per_video),
    }


def channel_context(profile: dict, risk_level: str) -> dict:
    """Build the channel context dict that gets embedded in the HTML report."""
    subs = profile.get("subscribers", 0)
    size_label = next(label for threshold, label in _SIZE_LABELS if subs >= threshold)

    strikes = profile.get("past_strikes", 0)
    if strikes == 0:
        strike_label = "None — clean record"
        strike_severity = "low"
    elif strikes == 1:
        strike_label = "1 of 3 — elevated caution advised"
        strike_severity = "medium"
    else:
        strike_label = "2 of 3 — one strike from permanent ban"
        strike_severity = "high"

    category = profile.get("category", "other").title()

    escalation: str | None = None
    if strikes == 1 and risk_level in ("HIGH", "MEDIUM"):
        escalation = (
            "A second strike means a 2-week upload freeze, loss of all monetization "
            "during that period, and leaves you one strike from permanent termination."
        )
    elif strikes >= 2 and risk_level != "LOW":
        escalation = (
            "CRITICAL: Another strike means permanent channel termination. "
            "Do not upload without editing all flagged moments."
        )

    return {
        "name":           profile.get("channel_name", ""),
        "subscribers":    profile.get("subscribers", 0),
        "monthly_views":  profile.get("monthly_views", 0),
        "cpm":            profile.get("cpm", 0),
        "videos_per_month": profile.get("videos_per_month", 0),
        "category":       category,
        "past_strikes":   strikes,
        "size_label":     size_label,
        "strike_label":   strike_label,
        "strike_severity": strike_severity,
        "category_note":  _CATEGORY_NOTES.get(category, _CATEGORY_NOTES["Other"]),
        "financial":      financial_impact(profile, risk_level),
        "strike_escalation": escalation,
    }


def get_thresholds(profile: dict | None) -> dict:
    """Return monetization risk thresholds for the given channel category."""
    category = (profile or {}).get("category", "other").lower()
    return CATEGORY_THRESHOLDS.get(category, CATEGORY_THRESHOLDS["other"])
