"""
DSG AFL stat registry — machine-readable catalogue of every stat available
through the Data Sports Group Australian Football API.

Each entry documents:
  entity        : which level the stat appears at ("team", "player", or both)
  pre_match_safe: True if it can be used as a pre-match predictor feature
                  (historical rolling average of this stat from prior games).
                  False for stats that are only known after the match ends or
                  that carry post-match contamination (brownlow, supercoach).
  feature_family: semantic grouping used by the feature pipeline to build
                  coherent feature families.
  target_markets: markets this stat most directly informs.
  leakage_note  : non-None string when there is a specific leakage risk to
                  document.

FEATURE_FAMILIES
----------------
  ball_winning     — ball-winning, possession volume and quality
  kick_quality     — efficiency and accuracy of disposal/kick choices
  territorial      — territory, forward-press, and entry quality
  stoppage         — stoppages, ruck contests, midfield control
  scoring          — scoring acts and scoring-chain involvement
  discipline       — turnovers, rule-break risk, composure
  opportunity      — minutes played, role, usage, workload
  evaluation_only  — post-match ratings; never a training feature
"""
from __future__ import annotations
from typing import Dict, Any

DSG_STAT_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Ball-winning / possession ──────────────────────────────────────────────
    "kicks": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "ball_winning",
        "target_markets": ["head_to_head", "line", "player_disposals"],
        "leakage_note": None,
    },
    "handballs": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "ball_winning",
        "target_markets": ["head_to_head", "player_disposals"],
        "leakage_note": None,
    },
    "disposals": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "ball_winning",
        "target_markets": ["head_to_head", "player_disposals"],
        "leakage_note": None,
    },
    "contested_possessions": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "ball_winning",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "uncontested_possessions": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "ball_winning",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },

    # ── Kick quality / style ───────────────────────────────────────────────────
    "effective_kicks": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "effective_handballs": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "effective_disposals": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head", "line", "player_disposals"],
        "leakage_note": None,
    },
    "clangers": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "marks": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "contested_marks": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "kick_quality",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },

    # ── Territory / forward pressure ──────────────────────────────────────────
    "inside_50s": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "territorial",
        "target_markets": ["head_to_head", "line", "total"],
        "leakage_note": None,
    },
    "rebound_50s": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "territorial",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "marks_inside_50": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "territorial",
        "target_markets": ["head_to_head", "line", "total"],
        "leakage_note": None,
    },

    # ── Stoppage / ruck / midfield control ────────────────────────────────────
    "clearances": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "stoppage",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "centre_clearances": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "stoppage",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "stoppage_clearances": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "stoppage",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "hitouts": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "stoppage",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "hitouts_to_advantage": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "stoppage",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },

    # ── Scoring / finishing ───────────────────────────────────────────────────
    "score": {
        "entity": ["team"],
        "pre_match_safe": True,
        "feature_family": "scoring",
        "target_markets": ["head_to_head", "line", "total"],
        "leakage_note": None,
    },
    "goals": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "scoring",
        "target_markets": ["head_to_head", "line", "total"],
        "leakage_note": None,
    },
    "behinds": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "scoring",
        "target_markets": ["head_to_head", "total"],
        "leakage_note": None,
    },
    "score_involvements": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "scoring",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "goal_assists": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "scoring",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },

    # ── Discipline / mistakes / risk ──────────────────────────────────────────
    "frees_for": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "discipline",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "frees_against": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "discipline",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "one_percenters": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "discipline",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "bounces": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "opportunity",
        "target_markets": ["head_to_head"],
        "leakage_note": "Mostly ruckmen; near-zero for non-ruck players.",
    },

    # ── Effort / physical ─────────────────────────────────────────────────────
    "tackles": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "discipline",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },
    "metres_gained": {
        "entity": ["team", "player"],
        "pre_match_safe": True,
        "feature_family": "territorial",
        "target_markets": ["head_to_head", "line"],
        "leakage_note": None,
    },
    "turnovers": {
        "entity": ["team"],
        "pre_match_safe": True,
        "feature_family": "discipline",
        "target_markets": ["head_to_head"],
        "leakage_note": None,
    },

    # ── Opportunity / role / usage ────────────────────────────────────────────
    "time_on_ground_pct": {
        "entity": ["player"],
        "pre_match_safe": True,
        "feature_family": "opportunity",
        "target_markets": ["player_disposals"],
        "leakage_note": "TOG% for the current match is unknown pre-match; "
                        "use rolling average from prior games only.",
    },

    # ── Evaluation-only (post-match ratings — NEVER use as training features) ─
    "brownlow_votes": {
        "entity": ["player"],
        "pre_match_safe": False,
        "feature_family": "evaluation_only",
        "target_markets": [],
        "leakage_note": "Awarded after the match. Use only for post-hoc model evaluation.",
    },
    "supercoach_score": {
        "entity": ["player"],
        "pre_match_safe": False,
        "feature_family": "evaluation_only",
        "target_markets": [],
        "leakage_note": "Fantasy scoring computed after match completion.",
    },
    "rating_points": {
        "entity": ["player"],
        "pre_match_safe": False,
        "feature_family": "evaluation_only",
        "target_markets": [],
        "leakage_note": "Computed post-match; leaks outcome information.",
    },
}


def get_pre_match_safe_stats(entity: str = "team") -> list[str]:
    """Return all stat names that are safe to use as pre-match predictors."""
    return [
        name for name, meta in DSG_STAT_REGISTRY.items()
        if entity in meta["entity"] and meta["pre_match_safe"]
    ]


def get_stats_by_family(family: str) -> list[str]:
    """Return all stat names belonging to a given feature family."""
    return [
        name for name, meta in DSG_STAT_REGISTRY.items()
        if meta["feature_family"] == family
    ]


def get_team_stat_cols() -> list[str]:
    """
    Ordered list of pre-match-safe team-level stats for the rolling feature loop.
    Ordered by feature family so related stats are computed together.
    Excludes evaluation-only and player-only stats.

    Intentionally excluded (low predictive value or fully captured by derived ratios):
      behinds          — redundant with scoring_efficiency derived feature
      uncontested_possessions — redundant with contested_possessions + disposals
      bounces          — near-zero for non-ruckmen, extreme noise for teams
      one_percenters   — highly variable, weak team-outcome signal
      goal_assists     — largely captured by score_involvements
      effective_handballs — captured by eff_handball_rate derived feature
    """
    return [
        # Ball-winning / possession
        "kicks", "handballs", "disposals",
        "effective_kicks", "effective_disposals",
        "contested_possessions", "clangers",
        # Kick quality / style
        "marks", "contested_marks",
        # Territory / forward pressure
        "inside_50s", "rebound_50s", "marks_inside_50",
        # Stoppage / ruck / midfield
        "clearances", "centre_clearances", "stoppage_clearances",
        "hitouts", "hitouts_to_advantage",
        # Scoring / finishing
        "score", "goals", "score_involvements",
        # Discipline / mistakes
        "frees_for", "frees_against",
        # Effort / physical
        "tackles", "metres_gained", "turnovers",
    ]


def get_player_secondary_stats() -> list[str]:
    """
    Secondary stats to compute rolling EWMA alongside the primary player prop
    stat (disposals). Each feeds into a derived per-disposal rate feature.

    Ordered by signal priority for disposals prediction:
      effective_disposals  — disposal efficiency (quality, not just volume)
      contested_possessions — contested workload / midfield hardness
      score_involvements   — scoring chain involvement (impact proxy)
      clangers             — error rate (discipline)
      tackles              — defensive work rate (role clarity)
      marks                — marking craft / field position
      inside_50s           — forward pressure / role type
      goal_assists         — creative output
      time_on_ground_pct   — usage / opportunity (pre-match rolling avg only)
    """
    return [
        "effective_disposals",
        "contested_possessions",
        "score_involvements",
        "clangers",
        "tackles",
        "marks",
        "inside_50s",
        "goal_assists",
        "time_on_ground_pct",
    ]
