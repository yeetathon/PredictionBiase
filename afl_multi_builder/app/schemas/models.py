"""Pydantic schemas for API request/response validation."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    timestamp: str


class PipelineRunRequest(BaseModel):
    fixture_ids: Optional[List[int]] = None
    include_player_legs: bool = True
    min_ev: Optional[float] = None
    max_correlation: Optional[float] = None


class LegResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    leg_id: str
    fixture_id: int
    player_id: Optional[int]
    team_id: Optional[int]
    market_type: str
    selection: str
    decimal_odds: float
    model_probability: float
    ev: float
    confidence_score: float
    explanation: str


class MultiGenerateRequest(BaseModel):
    legs: List[LegResponse]
    n_legs: Optional[int] = None
    mode: str = Field("value", pattern="^(value|safe|same_game)$")
    max_results: int = Field(10, ge=1, le=50)
    max_correlation: Optional[float] = None
    min_ev: Optional[float] = None


class MultiResponse(BaseModel):
    multi_id: str
    n_legs: int
    multi_type: str
    combined_odds: float
    adjusted_probability: float
    ev: float
    correlation_score: float
    correlation_label: str
    risk_score: float
    explanation: str
    leg_ids: List[str]


class PipelineResponse(BaseModel):
    run_id: str
    timestamp: str
    n_fixtures: int
    n_candidate_legs: int
    elapsed_seconds: float
    value_legs: List[LegResponse]
    safe_legs: List[LegResponse]
    value_multis: List[MultiResponse]
    safe_multis: List[MultiResponse]
    same_game_multis: List[MultiResponse]


class TrainingRunResponse(BaseModel):
    run_id: str
    timestamp: str
    status: str
    match_model: Optional[Dict[str, Any]] = None
    player_disposals_model: Optional[Dict[str, Any]] = None


class BacktestRunResponse(BaseModel):
    run_id: str
    timestamp: str
    status: str
    n_total_predictions: Optional[int] = None
    overall_metrics: Optional[Dict[str, float]] = None
    calibration_report: Optional[Dict[str, Any]] = None
    roi_by_edge_threshold: Optional[List[Dict[str, Any]]] = None
    clv_analysis: Optional[Dict[str, Any]] = None
    fold_metrics: Optional[List[Dict[str, Any]]] = None


class SummaryResponse(BaseModel):
    timestamp: str
    data_summary: Dict[str, Any]
    models_available: List[str]
    settings: Dict[str, Any]
