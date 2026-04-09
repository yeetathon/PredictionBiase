"""Edge Intelligence module — scrapes the web for fresh AFL signals.

Fetches injury reports, team news, suspensions, and selection updates from
public AFL news sources and RSS feeds.  Results are cached locally at
data/cache/edge_signals.json and refreshed after
settings.scrape_cache_ttl_minutes minutes.

Sources (in priority order)
---------------------------
1. AFL Official RSS  — https://www.afl.com.au/rss
2. Fox Sports AFL RSS — https://www.foxsports.com.au/rss.xml
3. AFL team-news HTML pages
4. AFL fantasy/injury news

All sources degrade gracefully: if one fails, the others still run.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as _ET

import requests
from bs4 import BeautifulSoup
from loguru import logger

from app.core.config import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENT = "AFL-PredictionBiase/1.0 (research tool)"
_REQUEST_TIMEOUT = 15  # seconds
_INTER_SOURCE_SLEEP = 0.5  # seconds between source fetches

_CACHE_DIR = Path("data/cache")
_CACHE_FILE = _CACHE_DIR / "edge_signals.json"

_AFL_TEAMS: List[str] = [
    "Adelaide Crows",
    "Brisbane Lions",
    "Carlton",
    "Collingwood",
    "Essendon",
    "Fremantle",
    "Geelong Cats",
    "Gold Coast Suns",
    "Greater Western Sydney Giants",
    "Hawthorn",
    "Melbourne",
    "North Melbourne",
    "Port Adelaide",
    "Richmond",
    "St Kilda",
    "Sydney Swans",
    "West Coast Eagles",
    "Western Bulldogs",
]

# Aliases / partial names that appear in news text → canonical name
_TEAM_ALIASES: Dict[str, str] = {
    # Adelaide
    "adelaide": "Adelaide Crows",
    "crows": "Adelaide Crows",
    # Brisbane
    "brisbane": "Brisbane Lions",
    "lions": "Brisbane Lions",
    # Carlton
    "carlton": "Carlton",
    "blues": "Carlton",
    # Collingwood
    "collingwood": "Collingwood",
    "magpies": "Collingwood",
    "pies": "Collingwood",
    # Essendon
    "essendon": "Essendon",
    "bombers": "Essendon",
    "dons": "Essendon",
    # Fremantle
    "fremantle": "Fremantle",
    "dockers": "Fremantle",
    "freo": "Fremantle",
    # Geelong
    "geelong": "Geelong Cats",
    "cats": "Geelong Cats",
    # Gold Coast
    "gold coast": "Gold Coast Suns",
    "suns": "Gold Coast Suns",
    # GWS
    "gws": "Greater Western Sydney Giants",
    "greater western sydney": "Greater Western Sydney Giants",
    "giants": "Greater Western Sydney Giants",
    # Hawthorn
    "hawthorn": "Hawthorn",
    "hawks": "Hawthorn",
    # Melbourne
    "melbourne": "Melbourne",
    "demons": "Melbourne",
    "dees": "Melbourne",
    # North Melbourne
    "north melbourne": "North Melbourne",
    "north": "North Melbourne",
    "kangaroos": "North Melbourne",
    "roos": "North Melbourne",
    # Port Adelaide
    "port adelaide": "Port Adelaide",
    "port": "Port Adelaide",
    "power": "Port Adelaide",
    # Richmond
    "richmond": "Richmond",
    "tigers": "Richmond",
    # St Kilda
    "st kilda": "St Kilda",
    "saints": "St Kilda",
    # Sydney
    "sydney": "Sydney Swans",
    "swans": "Sydney Swans",
    # West Coast
    "west coast": "West Coast Eagles",
    "eagles": "West Coast Eagles",
    # Western Bulldogs
    "western bulldogs": "Western Bulldogs",
    "bulldogs": "Western Bulldogs",
    "dogs": "Western Bulldogs",
    "footscray": "Western Bulldogs",
}

# Keyword lists for signal classification
_INJURY_KEYWORDS = [
    "injured", "injury", "out", "doubt", "unavailable", "concussion",
    "hamstring", "knee", "ankle", "shoulder", "ribs", "fractured", "torn",
    "strain",
]
_SUSPENSION_KEYWORDS = [
    "suspended", "banned", "tribunal", "mrp", "week suspension", "rubbed out",
]
_LATE_OUT_KEYWORDS = [
    "late out", "last-minute", "withdraws", "withdrew", "scratched",
]
_SELECTION_KEYWORDS = [
    "named", "selected", "recalled", "returns", "debutant", "dropped",
    "omitted", "replaced",
]
_TEAM_NEWS_KEYWORDS = [
    "team list", "team sheet", "squad", "lineup", "announced",
]
_WEATHER_KEYWORDS = [
    "weather", "rain", "wind", "storm", "forecast", "conditions",
]

# Severity weight per signal type (used in impact calculation)
_SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 0.20,
    "high": 0.12,
    "medium": 0.06,
    "low": 0.02,
}


# ---------------------------------------------------------------------------
# EdgeSignal dataclass
# ---------------------------------------------------------------------------

@dataclass
class EdgeSignal:
    """A single piece of intelligence that may affect a prediction."""

    signal_id: str          # SHA-256 hash of source_url + headline
    signal_type: str        # injury / late_out / suspension / selection / team_news / weather_alert / breaking
    severity: str           # critical / high / medium / low
    team_name: str          # canonical AFL team name (empty if unknown)
    player_name: str        # player affected (empty if N/A)
    headline: str           # short headline text
    summary: str            # 1-2 sentence summary
    source_url: str         # URL where the signal was found
    source_name: str        # human-readable source label
    published_at: datetime  # when the item was published / date-scraped
    scraped_at: datetime    # when our scraper collected it
    fixture_relevance: List[str] = field(default_factory=list)  # fixture labels
    is_stale: bool = False  # True if older than scrape_max_age_hours

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        d = asdict(self)
        d["published_at"] = self.published_at.isoformat()
        d["scraped_at"] = self.scraped_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EdgeSignal":
        """Reconstruct from a dict (e.g. loaded from the JSON cache)."""
        d = dict(d)
        for key in ("published_at", "scraped_at"):
            raw = d.get(key)
            if isinstance(raw, str):
                try:
                    d[key] = datetime.fromisoformat(raw)
                except ValueError:
                    d[key] = datetime.now(timezone.utc)
            elif not isinstance(raw, datetime):
                d[key] = datetime.now(timezone.utc)
        if "fixture_relevance" not in d:
            d["fixture_relevance"] = []
        return cls(**d)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_signal_id(source_url: str, headline: str) -> str:
    """Return a stable 16-char hex id derived from source + headline."""
    raw = f"{source_url}|{headline}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def _normalise_text(text: str) -> str:
    """Lower-case, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def _classify_signal_type(headline: str, summary: str) -> str:
    """Return the best-matching signal type for a headline + summary."""
    combined = _normalise_text(f"{headline} {summary}")

    if any(kw in combined for kw in _LATE_OUT_KEYWORDS):
        return "late_out"
    if any(kw in combined for kw in _SUSPENSION_KEYWORDS):
        return "suspension"
    if any(kw in combined for kw in _INJURY_KEYWORDS):
        return "injury"
    if any(kw in combined for kw in _TEAM_NEWS_KEYWORDS):
        return "team_news"
    if any(kw in combined for kw in _SELECTION_KEYWORDS):
        return "selection"
    if any(kw in combined for kw in _WEATHER_KEYWORDS):
        return "weather_alert"
    return "breaking"


def _classify_severity(signal_type: str, headline: str, summary: str) -> str:
    """Assign a severity level based on signal type and text content."""
    combined = _normalise_text(f"{headline} {summary}")

    if signal_type in ("suspension", "late_out"):
        return "critical"
    if signal_type == "injury":
        # Multiple players out → critical; named player out → high
        multi_indicators = ["multiple", "several", "outs", "short list"]
        if any(ind in combined for ind in multi_indicators):
            return "critical"
        definite_out = ["ruled out", "will miss", "season-ending", "won't play", "wont play"]
        if any(d in combined for d in definite_out):
            return "high"
        return "medium"
    if signal_type == "selection":
        return "medium"
    if signal_type == "team_news":
        return "medium"
    if signal_type == "weather_alert":
        severe_weather = ["storm", "heavy rain", "dangerous", "extreme wind"]
        if any(w in combined for w in severe_weather):
            return "high"
        return "low"
    return "low"


def _extract_team_name(text: str) -> str:
    """Return the first canonical AFL team name found in *text*, or ''."""
    lowered = _normalise_text(text)
    # Try multi-word aliases first (longest first to avoid partial matches)
    for alias in sorted(_TEAM_ALIASES, key=len, reverse=True):
        if alias in lowered:
            return _TEAM_ALIASES[alias]
    return ""


def _extract_player_name(text: str) -> str:
    """Heuristically extract the most likely player name from a headline.

    Looks for a consecutive pair of title-case words not matching known team /
    common stop words.  Returns the first candidate found, or ''.
    """
    # Match two consecutive capitalised words (first-name + surname pattern)
    candidates = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", text)
    stop_words = {
        "Adelaide Crows", "Brisbane Lions", "Carlton", "Collingwood",
        "Essendon", "Fremantle", "Geelong Cats", "Gold Coast Suns",
        "Greater Western Sydney Giants", "Hawthorn", "Melbourne",
        "North Melbourne", "Port Adelaide", "Richmond", "St Kilda",
        "Sydney Swans", "West Coast Eagles", "Western Bulldogs",
        "AFL", "Round", "Season", "Friday", "Saturday", "Sunday",
        "Monday", "Tuesday", "Wednesday", "Thursday",
    }
    for candidate in candidates:
        if candidate not in stop_words and len(candidate.split()) <= 3:
            return candidate
    return ""


def _parse_published_date(date_str: str) -> datetime:
    """Parse an RSS/Atom date string into a timezone-aware datetime."""
    if not date_str:
        return datetime.now(timezone.utc)
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # ISO 8601 fallback (Atom feeds)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+00:00", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _is_afl_content(headline: str, summary: str) -> bool:
    """Return True if the item appears to be AFL-related."""
    combined = _normalise_text(f"{headline} {summary}")
    afl_markers = [
        "afl", "australian football", "footy",
        "ruck", "midfielder", "forward", "defender",
    ] + [t.lower() for t in _AFL_TEAMS]
    # Also include aliases
    afl_markers += list(_TEAM_ALIASES.keys())
    return any(m in combined for m in afl_markers)


def _build_requests_session() -> requests.Session:
    """Return a configured requests.Session with the standard user-agent."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    })
    return session


# ---------------------------------------------------------------------------
# Per-source scrapers
# ---------------------------------------------------------------------------

def _parse_rss_xml(content: bytes) -> List[Dict[str, str]]:
    """
    Parse RSS 2.0 or Atom feed XML using stdlib ElementTree.
    Returns list of dicts with keys: title, summary, link, published.
    """
    entries: List[Dict[str, str]] = []
    try:
        root = _ET.fromstring(content)
    except _ET.ParseError:
        return entries

    ns_atom = "http://www.w3.org/2005/Atom"
    ns_content = "http://purl.org/rss/1.0/modules/content/"

    # Atom feed
    if root.tag == f"{{{ns_atom}}}feed" or "feed" in root.tag.lower():
        for entry in root.findall(f"{{{ns_atom}}}entry"):
            title_el = entry.find(f"{{{ns_atom}}}title")
            summary_el = entry.find(f"{{{ns_atom}}}summary") or entry.find(f"{{{ns_atom}}}content")
            link_el = entry.find(f"{{{ns_atom}}}link")
            pub_el = entry.find(f"{{{ns_atom}}}published") or entry.find(f"{{{ns_atom}}}updated")
            entries.append({
                "title": (title_el.text or "") if title_el is not None else "",
                "summary": (summary_el.text or "") if summary_el is not None else "",
                "link": (link_el.get("href", "") if link_el is not None else ""),
                "published": (pub_el.text or "") if pub_el is not None else "",
            })
        return entries

    # RSS 2.0
    for item in root.iter("item"):
        title_el = item.find("title")
        summary_el = item.find("description")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        # Also try content:encoded for richer summary
        content_el = item.find(f"{{{ns_content}}}encoded")
        raw_summary = (content_el.text or "") if content_el is not None else ""
        if not raw_summary:
            raw_summary = (summary_el.text or "") if summary_el is not None else ""
        entries.append({
            "title": (title_el.text or "") if title_el is not None else "",
            "summary": raw_summary,
            "link": (link_el.text or "") if link_el is not None else "",
            "published": (pub_el.text or "") if pub_el is not None else "",
        })
    return entries


def _scrape_rss(
    url: str,
    source_name: str,
    session: requests.Session,
    now: datetime,
    afl_only: bool = False,
) -> List[EdgeSignal]:
    """Parse an RSS/Atom feed and return EdgeSignal objects.

    Parameters
    ----------
    url:
        The RSS feed URL.
    source_name:
        Human-readable label for the source (e.g. "AFL.com.au").
    session:
        Shared requests.Session.
    now:
        Current UTC datetime.
    afl_only:
        When True, skip entries that don't appear to be AFL-related.
    """
    signals: List[EdgeSignal] = []

    try:
        response = session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("EdgeIntelligence: RSS fetch failed url={} error={}", url, exc)
        return signals

    try:
        raw_entries = _parse_rss_xml(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.warning("EdgeIntelligence: RSS XML parse failed url={} error={}", url, exc)
        return signals

    logger.debug("EdgeIntelligence: RSS {} entries={}", source_name, len(raw_entries))

    for entry in raw_entries:
        headline: str = entry.get("title", "") or ""
        summary: str = entry.get("summary", "") or ""
        link: str = entry.get("link", url) or url

        # Strip HTML tags from summary
        if summary:
            try:
                summary = BeautifulSoup(summary, "lxml").get_text(separator=" ")
            except Exception:  # noqa: BLE001
                summary = re.sub(r"<[^>]+>", " ", summary)
            summary = re.sub(r"\s+", " ", summary).strip()

        # Trim summary to ~300 chars
        if len(summary) > 300:
            summary = summary[:297] + "..."

        if not headline:
            continue
        if afl_only and not _is_afl_content(headline, summary):
            continue

        published_at = _parse_published_date(entry.get("published", ""))
        signal_type = _classify_signal_type(headline, summary)
        severity = _classify_severity(signal_type, headline, summary)
        team_name = _extract_team_name(f"{headline} {summary}")
        player_name = _extract_player_name(headline)
        signal_id = _make_signal_id(link, headline)
        age_hours = (now - published_at.replace(tzinfo=timezone.utc)
                     if published_at.tzinfo is None
                     else (now - published_at)).total_seconds() / 3600
        is_stale = age_hours > settings.scrape_max_age_hours

        signals.append(EdgeSignal(
            signal_id=signal_id,
            signal_type=signal_type,
            severity=severity,
            team_name=team_name,
            player_name=player_name,
            headline=headline.strip(),
            summary=summary.strip(),
            source_url=link,
            source_name=source_name,
            published_at=published_at if published_at.tzinfo else published_at.replace(tzinfo=timezone.utc),
            scraped_at=now,
            fixture_relevance=[],
            is_stale=is_stale,
        ))

    return signals


def _scrape_afl_team_news_html(
    session: requests.Session,
    now: datetime,
) -> List[EdgeSignal]:
    """Scrape AFL.com.au team-news pages for selection/injury text.

    Tries multiple URLs and returns whatever succeeds.
    """
    signals: List[EdgeSignal] = []
    candidate_urls = [
        ("https://www.afl.com.au/news/team-sheets", "AFL.com.au Team Sheets"),
        ("https://www.afl.com.au/matches/team-selector", "AFL.com.au Team Selector"),
    ]

    for url, label in candidate_urls:
        try:
            response = session.get(url, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("EdgeIntelligence: team-news page unavailable url={} error={}", url, exc)
            continue

        try:
            soup = BeautifulSoup(response.text, "lxml")
        except Exception as exc:  # noqa: BLE001
            logger.debug("EdgeIntelligence: HTML parse failed url={} error={}", url, exc)
            continue

        # Pull candidate text blocks — headings and article-level paragraphs
        text_blocks: List[Tuple[str, str]] = []

        # h2 / h3 headings that look like team news headlines
        for tag in soup.find_all(["h2", "h3", "h4"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and _is_afl_content(text, ""):
                # Try to grab the next sibling paragraph as summary
                sibling = tag.find_next_sibling(["p", "div"])
                sibling_text = sibling.get_text(separator=" ", strip=True) if sibling else ""
                text_blocks.append((text, sibling_text[:300]))

        # Standalone paragraphs inside article/section with team keywords
        for tag in soup.find_all(["p"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 40 and _is_afl_content(text, ""):
                text_blocks.append((text[:120], text[:300]))

        # Deduplicate by headline
        seen: set = set()
        for headline_text, summary_text in text_blocks:
            headline_text = re.sub(r"\s+", " ", headline_text).strip()
            summary_text = re.sub(r"\s+", " ", summary_text).strip()
            key = _make_signal_id(url, headline_text)
            if key in seen:
                continue
            seen.add(key)

            signal_type = _classify_signal_type(headline_text, summary_text)
            # Only keep relevant signal types from HTML pages
            if signal_type not in ("injury", "selection", "team_news", "late_out", "suspension"):
                continue

            severity = _classify_severity(signal_type, headline_text, summary_text)
            team_name = _extract_team_name(f"{headline_text} {summary_text}")
            player_name = _extract_player_name(headline_text)

            signals.append(EdgeSignal(
                signal_id=key,
                signal_type=signal_type,
                severity=severity,
                team_name=team_name,
                player_name=player_name,
                headline=headline_text,
                summary=summary_text if summary_text else headline_text,
                source_url=url,
                source_name=label,
                published_at=now,  # HTML pages have no per-item publish date
                scraped_at=now,
                fixture_relevance=[],
                is_stale=False,
            ))

        if signals:
            logger.debug(
                "EdgeIntelligence: HTML scrape {} signals={}", label, len(signals)
            )
            # One successful URL is sufficient
            break

    return signals


def _scrape_afl_fantasy_news(
    session: requests.Session,
    now: datetime,
) -> List[EdgeSignal]:
    """Attempt to fetch AFL fantasy / SuperCoach injury news via HTML."""
    signals: List[EdgeSignal] = []
    url = "https://www.afl.com.au/fantasy/news"

    try:
        response = session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.debug(
            "EdgeIntelligence: AFL fantasy news unavailable url={} error={}", url, exc
        )
        return signals

    try:
        soup = BeautifulSoup(response.text, "lxml")
    except Exception as exc:  # noqa: BLE001
        logger.debug("EdgeIntelligence: fantasy news HTML parse failed error={}", exc)
        return signals

    seen: set = set()
    for tag in soup.find_all(["h2", "h3", "h4", "p"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < 30:
            continue
        if not _is_afl_content(text, ""):
            continue

        headline_text = text[:120].strip()
        summary_text = text[:300].strip()
        signal_type = _classify_signal_type(headline_text, summary_text)

        if signal_type not in ("injury", "late_out", "suspension", "selection"):
            continue

        key = _make_signal_id(url, headline_text)
        if key in seen:
            continue
        seen.add(key)

        severity = _classify_severity(signal_type, headline_text, summary_text)
        team_name = _extract_team_name(headline_text)
        player_name = _extract_player_name(headline_text)

        signals.append(EdgeSignal(
            signal_id=key,
            signal_type=signal_type,
            severity=severity,
            team_name=team_name,
            player_name=player_name,
            headline=headline_text,
            summary=summary_text,
            source_url=url,
            source_name="AFL Fantasy",
            published_at=now,
            scraped_at=now,
            fixture_relevance=[],
            is_stale=False,
        ))

    logger.debug("EdgeIntelligence: AFL fantasy news signals={}", len(signals))
    return signals


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _ensure_cache_dir() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache() -> Tuple[Optional[datetime], List[EdgeSignal]]:
    """Load cached signals from disk.

    Returns
    -------
    (cached_at, signals)
        cached_at is None if no valid cache exists.
    """
    if not _CACHE_FILE.exists():
        return None, []
    try:
        raw = _CACHE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        cached_at_str = data.get("cached_at")
        if not cached_at_str:
            return None, []
        cached_at = datetime.fromisoformat(cached_at_str)
        signals = [EdgeSignal.from_dict(s) for s in data.get("signals", [])]
        return cached_at, signals
    except Exception as exc:  # noqa: BLE001
        logger.warning("EdgeIntelligence: cache load failed error={}", exc)
        return None, []


def _save_cache(signals: List[EdgeSignal]) -> None:
    """Persist signals to the JSON cache file."""
    _ensure_cache_dir()
    try:
        data = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "signals": [s.to_dict() for s in signals],
        }
        _CACHE_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("EdgeIntelligence: cache saved signals={}", len(signals))
    except Exception as exc:  # noqa: BLE001
        logger.warning("EdgeIntelligence: cache save failed error={}", exc)


def _cache_is_fresh(cached_at: datetime) -> bool:
    """Return True if the cache is younger than scrape_cache_ttl_minutes."""
    now = datetime.now(timezone.utc)
    cached_at_aware = (
        cached_at if cached_at.tzinfo else cached_at.replace(tzinfo=timezone.utc)
    )
    age_minutes = (now - cached_at_aware).total_seconds() / 60
    return age_minutes < settings.scrape_cache_ttl_minutes


# ---------------------------------------------------------------------------
# Team-name fuzzy matching
# ---------------------------------------------------------------------------

def _token_overlap_score(a: str, b: str) -> float:
    """Return the Jaccard token-overlap score between two strings."""
    tokens_a = set(_normalise_text(a).split())
    tokens_b = set(_normalise_text(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def fuzzy_match_team(name: str) -> str:
    """Map an arbitrary team-name string to a canonical AFL team name.

    First tries the alias dictionary (exact substring), then falls back to
    Jaccard token overlap against the canonical list.

    Returns
    -------
    str
        Canonical team name, or '' if no match meets the threshold.
    """
    if not name:
        return ""

    # 1. Direct canonical match
    if name in _AFL_TEAMS:
        return name

    lowered = _normalise_text(name)

    # 2. Alias lookup
    for alias, canonical in sorted(_TEAM_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in lowered:
            return canonical

    # 3. Token-overlap fallback (threshold 0.4)
    best_score = 0.0
    best_team = ""
    for canonical in _AFL_TEAMS:
        score = _token_overlap_score(name, canonical)
        if score > best_score:
            best_score = score
            best_team = canonical
    if best_score >= 0.4:
        return best_team

    return ""


# ---------------------------------------------------------------------------
# Fixture relevance tagging
# ---------------------------------------------------------------------------

def _tag_fixture_relevance(
    signals: List[EdgeSignal],
    fixtures: Optional[List[Dict[str, Any]]] = None,
) -> List[EdgeSignal]:
    """Attach fixture labels to signals whose team_name matches a fixture.

    Modifies signals in-place (also returns the list for convenience).
    *fixtures* is expected to be a list of dicts with keys
    ``home_team_name`` and ``away_team_name``.
    """
    if not fixtures:
        return signals

    for sig in signals:
        if not sig.team_name:
            continue
        labels: List[str] = []
        for fix in fixtures:
            home = fix.get("home_team_name", "")
            away = fix.get("away_team_name", "")
            if sig.team_name in (home, away):
                labels.append(f"{home} vs {away}")
        sig.fixture_relevance = labels

    return signals


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class EdgeIntelligenceService:
    """Orchestrates multi-source AFL edge-signal collection with caching.

    Usage
    -----
    ::

        svc = EdgeIntelligenceService()
        signals = svc.fetch_signals()
        impact  = svc.get_impact_summary("Hawthorn", "Collingwood")
    """

    def __init__(self) -> None:
        self._session = _build_requests_session()
        logger.debug(
            "EdgeIntelligenceService initialised: "
            "scraping_enabled={} ttl_minutes={} max_age_hours={}",
            settings.enable_scraping,
            settings.scrape_cache_ttl_minutes,
            settings.scrape_max_age_hours,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_signals(self, teams: Optional[List[str]] = None) -> List[EdgeSignal]:
        """Fetch and return current AFL edge signals.

        Results are cached for ``settings.scrape_cache_ttl_minutes`` minutes.
        If all sources fail, returns ``[]`` with a warning log — never raises.

        Parameters
        ----------
        teams:
            Optional list of canonical AFL team names.  When provided, only
            signals relevant to those teams (or with no team attribution) are
            returned.  Pass ``None`` to return all signals.
        """
        if not settings.enable_scraping:
            logger.info("EdgeIntelligence: scraping disabled via settings")
            return []

        # Check cache
        cached_at, cached_signals = _load_cache()
        if cached_at is not None and _cache_is_fresh(cached_at):
            logger.debug(
                "EdgeIntelligence: returning cached signals count={}",
                len(cached_signals),
            )
            return self._filter_by_teams(cached_signals, teams)

        # Scrape fresh signals
        signals = self._scrape_all()

        # Persist to cache (even if empty — prevents hammering on repeated failures)
        _save_cache(signals)

        return self._filter_by_teams(signals, teams)

    def get_team_signals(self, team_name: str) -> List[EdgeSignal]:
        """Return all current signals relevant to *team_name*.

        Uses fuzzy matching so "Geelong" matches "Geelong Cats" etc.
        """
        canonical = fuzzy_match_team(team_name) or team_name
        all_signals = self.fetch_signals()
        return [
            s for s in all_signals
            if not s.team_name or s.team_name == canonical
               or fuzzy_match_team(s.team_name) == canonical
        ]

    def get_fixture_signals(self, home_team: str, away_team: str) -> List[EdgeSignal]:
        """Return all signals relevant to the home_team vs away_team fixture."""
        canonical_home = fuzzy_match_team(home_team) or home_team
        canonical_away = fuzzy_match_team(away_team) or away_team
        all_signals = self.fetch_signals()

        relevant: List[EdgeSignal] = []
        for sig in all_signals:
            if not sig.team_name:
                # No team attributed — include if it's a high-priority signal
                if sig.severity in ("critical", "high"):
                    relevant.append(sig)
                continue
            canonical_sig = fuzzy_match_team(sig.team_name) or sig.team_name
            if canonical_sig in (canonical_home, canonical_away):
                relevant.append(sig)

        return relevant

    def get_impact_summary(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Summarise the net prediction impact of current edge signals.

        Returns
        -------
        dict with keys:

        home_impact : float in [-1, 1]
            Positive = signals favour home team; negative = disfavour.
        away_impact : float in [-1, 1]
            Positive = signals favour away team; negative = disfavour.
        signals : List[EdgeSignal]
            All signals relevant to this fixture.
        confidence_penalty : float in [0, 0.1]
            Suggested reduction to model confidence given signal uncertainty.
        """
        canonical_home = fuzzy_match_team(home_team) or home_team
        canonical_away = fuzzy_match_team(away_team) or away_team
        signals = self.get_fixture_signals(canonical_home, canonical_away)

        home_delta = 0.0
        away_delta = 0.0
        confidence_penalty = 0.0

        for sig in signals:
            weight = _SEVERITY_WEIGHTS.get(sig.severity, 0.02)
            canonical_sig = fuzzy_match_team(sig.team_name) or sig.team_name

            if sig.signal_type in ("injury", "late_out", "suspension"):
                # Bad news for the team affected
                if canonical_sig == canonical_home:
                    home_delta -= weight
                elif canonical_sig == canonical_away:
                    away_delta -= weight
                confidence_penalty = min(0.10, confidence_penalty + weight * 0.5)

            elif sig.signal_type in ("selection", "team_news"):
                # Neutral or slight positive (named team sheet = confirmed squad)
                if canonical_sig == canonical_home:
                    home_delta += weight * 0.3
                elif canonical_sig == canonical_away:
                    away_delta += weight * 0.3

            elif sig.signal_type == "weather_alert":
                # Weather affects both equally but increases uncertainty
                confidence_penalty = min(0.10, confidence_penalty + weight * 0.8)

        # Clamp to [-1, 1]
        home_impact = max(-1.0, min(1.0, home_delta))
        away_impact = max(-1.0, min(1.0, away_delta))

        logger.debug(
            "EdgeIntelligence: impact_summary fixture={} vs {} "
            "home_impact={:.3f} away_impact={:.3f} "
            "confidence_penalty={:.3f} n_signals={}",
            canonical_home,
            canonical_away,
            home_impact,
            away_impact,
            confidence_penalty,
            len(signals),
        )

        return {
            "home_impact": home_impact,
            "away_impact": away_impact,
            "signals": signals,
            "confidence_penalty": round(confidence_penalty, 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrape_all(self) -> List[EdgeSignal]:
        """Run all scrapers and merge deduplicated results."""
        now = datetime.now(timezone.utc)
        all_signals: List[EdgeSignal] = []
        seen_ids: set = set()

        # --- Source 1: AFL Official RSS ---
        logger.info("EdgeIntelligence: scraping AFL Official RSS")
        try:
            afl_rss = _scrape_rss(
                url="https://www.afl.com.au/rss",
                source_name="AFL.com.au",
                session=self._session,
                now=now,
                afl_only=False,  # AFL's own feed is always AFL content
            )
            self._merge_signals(all_signals, afl_rss, seen_ids)
            logger.info("EdgeIntelligence: AFL RSS signals={}", len(afl_rss))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EdgeIntelligence: AFL RSS scraper error={}", exc)

        time.sleep(_INTER_SOURCE_SLEEP)

        # --- Source 2: Fox Sports AFL RSS ---
        logger.info("EdgeIntelligence: scraping Fox Sports AFL RSS")
        try:
            fox_rss = _scrape_rss(
                url="https://www.foxsports.com.au/rss.xml",
                source_name="Fox Sports",
                session=self._session,
                now=now,
                afl_only=True,  # Fox covers multiple sports — filter AFL only
            )
            self._merge_signals(all_signals, fox_rss, seen_ids)
            logger.info("EdgeIntelligence: Fox Sports signals={}", len(fox_rss))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EdgeIntelligence: Fox Sports RSS scraper error={}", exc)

        time.sleep(_INTER_SOURCE_SLEEP)

        # --- Source 3: AFL team-news HTML pages ---
        logger.info("EdgeIntelligence: scraping AFL team-news HTML pages")
        try:
            html_signals = _scrape_afl_team_news_html(self._session, now)
            self._merge_signals(all_signals, html_signals, seen_ids)
            logger.info("EdgeIntelligence: HTML team-news signals={}", len(html_signals))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EdgeIntelligence: HTML team-news scraper error={}", exc)

        time.sleep(_INTER_SOURCE_SLEEP)

        # --- Source 4: AFL fantasy/injury news ---
        logger.info("EdgeIntelligence: scraping AFL fantasy/injury news")
        try:
            fantasy_signals = _scrape_afl_fantasy_news(self._session, now)
            self._merge_signals(all_signals, fantasy_signals, seen_ids)
            logger.info("EdgeIntelligence: AFL fantasy news signals={}", len(fantasy_signals))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EdgeIntelligence: AFL fantasy news scraper error={}", exc)

        if not all_signals:
            logger.warning(
                "EdgeIntelligence: all sources returned zero signals — "
                "returning empty list"
            )

        logger.info(
            "EdgeIntelligence: scrape complete total_signals={}",
            len(all_signals),
        )
        return all_signals

    @staticmethod
    def _merge_signals(
        target: List[EdgeSignal],
        new: List[EdgeSignal],
        seen_ids: set,
    ) -> None:
        """Append signals from *new* that are not already in *seen_ids*."""
        for sig in new:
            if sig.signal_id not in seen_ids:
                seen_ids.add(sig.signal_id)
                target.append(sig)

    @staticmethod
    def _filter_by_teams(
        signals: List[EdgeSignal],
        teams: Optional[List[str]],
    ) -> List[EdgeSignal]:
        """Return signals matching *teams* (or all signals if teams is None)."""
        if not teams:
            return signals
        canonical_teams = {fuzzy_match_team(t) or t for t in teams}
        return [
            s for s in signals
            if not s.team_name or (fuzzy_match_team(s.team_name) or s.team_name) in canonical_teams
        ]
