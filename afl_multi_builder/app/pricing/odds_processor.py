"""
Odds processing layer.
Converts bookmaker odds to implied probabilities,
removes vig/overround, computes edge and EV.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

from app.core.metrics import (
    implied_probability, remove_vig, compute_edge, compute_ev, compute_kelly_fraction
)
from app.data_ingestion.loader import DataLoader


@dataclass
class ProcessedOdds:
    """Fully processed odds entry with probabilities, edge, and EV."""
    fixture_id: int
    market_type: str
    selection: str
    bookmaker: str
    decimal_odds: float
    raw_implied_prob: float          # 1/decimal_odds (includes vig)
    vig_adjusted_prob: float         # vig removed, fair market probability
    overround: float                  # total of all raw implied probs - 1
    model_probability: Optional[float] = None
    calibrated_probability: Optional[float] = None
    edge: Optional[float] = None
    ev: Optional[float] = None
    kelly_fraction: Optional[float] = None
    timestamp: str = ""
    status: str = "active"


class OddsProcessor:
    """
    Processes raw bookmaker odds for a fixture.
    Handles head_to_head, line, total, and player_disposals markets.
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()

    def process_fixture_odds(self, fixture_id: int) -> Dict[str, List[ProcessedOdds]]:
        """
        Process all odds for a fixture, grouped by market type.
        Returns dict: market_type -> list of ProcessedOdds.
        """
        odds_df = self.loader.load_odds_df()
        if odds_df.empty:
            return {}

        fx_odds = odds_df[odds_df["fixture_id"] == fixture_id].copy()
        if fx_odds.empty:
            return {}

        result: Dict[str, List[ProcessedOdds]] = {}
        for market_type in fx_odds["market_type"].unique():
            market_odds = fx_odds[fx_odds["market_type"] == market_type]
            processed = self._process_market(market_type, market_odds)
            if processed:
                result[market_type] = processed

        return result

    def _process_market(
        self, market_type: str, market_df: pd.DataFrame
    ) -> List[ProcessedOdds]:
        """
        Process odds for a single market type.

        Player disposals markets use independent over/under vig calculation:
        each side is priced independently by bookmakers and should NOT be
        force-normalised as a binary pair when only one side is present.
        """
        if market_type == "player_disposals":
            return self._process_player_market(market_df)

        # Standard binary-pair processing for H2H, line, total
        best_per_selection: Dict[str, Dict] = {}
        for _, row in market_df.iterrows():
            sel = str(row["selection"])
            d_odds = float(row["decimal_odds"])
            if sel not in best_per_selection or d_odds > best_per_selection[sel]["decimal_odds"]:
                best_per_selection[sel] = row.to_dict()

        if not best_per_selection:
            return []

        all_raw_probs = [implied_probability(v["decimal_odds"]) for v in best_per_selection.values()]
        overround = sum(all_raw_probs) - 1.0
        vig_adj = remove_vig(all_raw_probs)

        processed = []
        for i, (sel, data) in enumerate(best_per_selection.items()):
            d_odds = float(data["decimal_odds"])
            raw_prob = implied_probability(d_odds)
            vig_prob = vig_adj[i] if i < len(vig_adj) else raw_prob
            po = ProcessedOdds(
                fixture_id=int(data["fixture_id"]),
                market_type=market_type,
                selection=sel,
                bookmaker=str(data["bookmaker"]),
                decimal_odds=d_odds,
                raw_implied_prob=round(raw_prob, 5),
                vig_adjusted_prob=round(vig_prob, 5),
                overround=round(overround, 5),
                timestamp=str(data.get("timestamp", "")),
                status=str(data.get("status", "active")),
            )
            processed.append(po)

        return processed

    def _process_player_market(self, market_df: pd.DataFrame) -> List[ProcessedOdds]:
        """
        Player disposals: treat over/under as independent bets (each has its own vig).

        When both sides are available, normalise the pair to remove vig.
        When only one side is available, use raw implied as best estimate.
        This prevents the common bug of normalising a 3-selection market as binary.
        """
        # Group by (player_id, line) selection pairs
        # Typical selections: "player_{pid}_over_{line}", "player_{pid}_under_{line}"
        pair_map: Dict[str, Dict[str, Dict]] = {}   # "pid_line" -> {"over"/"under" -> best_row}
        other: Dict[str, Dict] = {}  # selections that don't match the pattern

        for _, row in market_df.iterrows():
            sel = str(row["selection"])
            d_odds = float(row["decimal_odds"])
            parts = sel.split("_")
            # Expected: player_{pid}_{over|under}_{line}
            if len(parts) >= 4 and parts[0] == "player":
                try:
                    pid = parts[1]
                    direction = parts[2]  # "over" or "under"
                    line = parts[3]
                    key = f"{pid}_{line}"
                    if key not in pair_map:
                        pair_map[key] = {}
                    if direction not in pair_map[key] or d_odds > pair_map[key][direction]["decimal_odds"]:
                        pair_map[key][direction] = row.to_dict()
                except (IndexError, ValueError):
                    other[sel] = row.to_dict() if sel not in other or d_odds > other[sel]["decimal_odds"] else other[sel]
            else:
                if sel not in other or d_odds > other[sel]["decimal_odds"]:
                    other[sel] = row.to_dict()

        processed = []
        for key, dirs in pair_map.items():
            if "over" in dirs and "under" in dirs:
                # Both sides present: normalise to remove vig
                over_row = dirs["over"]
                under_row = dirs["under"]
                raw_over = implied_probability(float(over_row["decimal_odds"]))
                raw_under = implied_probability(float(under_row["decimal_odds"]))
                total_raw = raw_over + raw_under
                overround = total_raw - 1.0
                vig_over = raw_over / total_raw if total_raw > 0 else raw_over
                vig_under = raw_under / total_raw if total_raw > 0 else raw_under
                for direction, row_data, vig_prob, raw_prob in [
                    ("over", over_row, vig_over, raw_over),
                    ("under", under_row, vig_under, raw_under),
                ]:
                    processed.append(ProcessedOdds(
                        fixture_id=int(row_data["fixture_id"]),
                        market_type="player_disposals",
                        selection=str(row_data.get("selection", "")),
                        bookmaker=str(row_data.get("bookmaker", "")),
                        decimal_odds=float(row_data["decimal_odds"]),
                        raw_implied_prob=round(raw_prob, 5),
                        vig_adjusted_prob=round(vig_prob, 5),
                        overround=round(overround, 5),
                        timestamp=str(row_data.get("timestamp", "")),
                        status=str(row_data.get("status", "active")),
                    ))
            else:
                # Only one side available: use raw implied (no normalisation possible)
                direction = "over" if "over" in dirs else "under"
                row_data = dirs[direction]
                raw_prob = implied_probability(float(row_data["decimal_odds"]))
                processed.append(ProcessedOdds(
                    fixture_id=int(row_data["fixture_id"]),
                    market_type="player_disposals",
                    selection=str(row_data.get("selection", "")),
                    bookmaker=str(row_data.get("bookmaker", "")),
                    decimal_odds=float(row_data["decimal_odds"]),
                    raw_implied_prob=round(raw_prob, 5),
                    vig_adjusted_prob=round(raw_prob, 5),  # best we can do with one side
                    overround=0.0,
                    timestamp=str(row_data.get("timestamp", "")),
                    status=str(row_data.get("status", "active")),
                ))

        # Process any non-standard selections
        for sel, data in other.items():
            d_odds = float(data["decimal_odds"])
            raw_prob = implied_probability(d_odds)
            processed.append(ProcessedOdds(
                fixture_id=int(data["fixture_id"]),
                market_type="player_disposals",
                selection=sel,
                bookmaker=str(data.get("bookmaker", "")),
                decimal_odds=d_odds,
                raw_implied_prob=round(raw_prob, 5),
                vig_adjusted_prob=round(raw_prob, 5),
                overround=0.0,
                timestamp=str(data.get("timestamp", "")),
                status=str(data.get("status", "active")),
            ))

        return processed

    def attach_model_probabilities(
        self,
        processed: List[ProcessedOdds],
        model_probs: Dict[str, float],
        calibrated_probs: Optional[Dict[str, float]] = None,
    ) -> List[ProcessedOdds]:
        """
        Attach model probabilities and compute edge/EV for each ProcessedOdds entry.
        model_probs: {selection -> probability}
        """
        for po in processed:
            if po.selection in model_probs:
                mp = model_probs[po.selection]
                cp = (calibrated_probs or {}).get(po.selection, mp)
                po.model_probability = round(mp, 5)
                po.calibrated_probability = round(cp, 5)
                po.edge = round(compute_edge(cp, po.vig_adjusted_prob), 5)
                po.ev = round(compute_ev(cp, po.decimal_odds), 5)
                po.kelly_fraction = round(compute_kelly_fraction(cp, po.decimal_odds), 5)
        return processed

    def get_all_odds_summary(self) -> pd.DataFrame:
        """Return a flat DataFrame of all processed odds for upcoming fixtures."""
        fixtures = self.loader.load_fixtures_df()
        odds_df = self.loader.load_odds_df()

        if fixtures.empty or odds_df.empty:
            return pd.DataFrame()

        upcoming = fixtures[fixtures["status"] == "upcoming"]["fixture_id"].tolist()
        if not upcoming:
            # Fall back to all fixtures if no upcoming
            upcoming = fixtures["fixture_id"].tolist()

        all_rows = []
        for fid in upcoming:
            market_dict = self.process_fixture_odds(fid)
            for market_type, entries in market_dict.items():
                for po in entries:
                    all_rows.append({
                        "fixture_id": po.fixture_id,
                        "market_type": po.market_type,
                        "selection": po.selection,
                        "bookmaker": po.bookmaker,
                        "decimal_odds": po.decimal_odds,
                        "raw_implied_prob": po.raw_implied_prob,
                        "vig_adjusted_prob": po.vig_adjusted_prob,
                        "overround": po.overround,
                        "status": po.status,
                    })

        return pd.DataFrame(all_rows)

    def get_market_consensus(
        self, fixture_id: int, market_type: str = "head_to_head"
    ) -> Optional[Dict[str, float]]:
        """
        Compute consensus probability for a market from all bookmakers.
        Returns {selection -> consensus_prob}.
        """
        odds_df = self.loader.load_odds_df()
        if odds_df.empty:
            return None
        fx_odds = odds_df[
            (odds_df["fixture_id"] == fixture_id) &
            (odds_df["market_type"] == market_type)
        ]
        if fx_odds.empty:
            return None

        consensus = {}
        for sel in fx_odds["selection"].unique():
            sel_odds = fx_odds[fx_odds["selection"] == sel]["decimal_odds"]
            # Average implied probability across bookmakers
            avg_imp = float(sel_odds.apply(lambda x: 1.0 / x).mean())
            consensus[sel] = avg_imp

        # Vig-adjust
        total = sum(consensus.values())
        if total > 0:
            consensus = {k: v / total for k, v in consensus.items()}

        return consensus
