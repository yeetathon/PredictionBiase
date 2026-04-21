"""Tests for FastAPI endpoints.

The pipeline/training/backtest endpoints internally call DataLoader(), which
would try to instantiate AFLDataLoader without an authkey. We patch
_make_afl_provider to return an in-memory provider so all tests run offline.
"""
import pytest
import pandas as pd
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


def _in_memory_afl_provider():
    """Return a fully in-memory AFL data provider with realistic synthetic data."""

    completed = [
        {
            "fixture_id": i, "season": 2024, "round": i,
            "home_team_id": 1, "away_team_id": 2,
            "home_team_name": "Melbourne", "away_team_name": "Collingwood",
            "venue_id": 10, "date": f"2024-0{(i % 9) + 1}-{10 + i:02d}",
            "time": "19:30", "status": "completed",
            "result_home_score": (90 + i * 2) if i % 2 == 0 else (70 + i),
            "result_away_score": (70 + i) if i % 2 == 0 else (90 + i * 2),
            "home_score": float((90 + i * 2) if i % 2 == 0 else (70 + i)),
            "away_score": float((70 + i) if i % 2 == 0 else (90 + i * 2)),
            "home_win": int(i % 2 == 0),
        }
        for i in range(1, 11)
    ]
    upcoming = [
        {
            "fixture_id": 1001, "season": 2025, "round": 5,
            "home_team_id": 1, "away_team_id": 2,
            "home_team_name": "Melbourne", "away_team_name": "Collingwood",
            "venue_id": 10, "date": "2025-05-10", "time": "19:30",
            "status": "upcoming",
            "result_home_score": None, "result_away_score": None,
            "home_score": None, "away_score": None, "home_win": None,
        }
    ]
    all_fixtures = pd.DataFrame(completed + upcoming)

    teams = pd.DataFrame([
        {"team_id": 1, "name": "Melbourne",   "abbreviation": "MEL"},
        {"team_id": 2, "name": "Collingwood", "abbreviation": "COL"},
    ])

    players = pd.DataFrame([
        {"player_id": 101, "team_id": 1, "first_name": "Alice", "last_name": "Smith",
         "position": "midfielder", "jumper_number": 7},
        {"player_id": 102, "team_id": 2, "first_name": "Bob",   "last_name": "Jones",
         "position": "forward",    "jumper_number": 10},
    ])

    team_stats = pd.DataFrame([
        {
            "fixture_id": i, "team_id": 1 if j == 0 else 2,
            "is_home": j == 0,
            "kicks": 100 + i, "handballs": 80 + i, "disposals": 180 + i,
            "goals": 12 + i % 3, "behinds": 8,
            "score": 90 + i * 2 if j == 0 else 80 + i,
        }
        for i in range(1, 11)
        for j in range(2)
    ])

    player_stats = pd.DataFrame([
        {
            "fixture_id": i, "player_id": 101 if j == 0 else 102,
            "team_id": 1 if j == 0 else 2,
            "disposals": 25 + i % 5, "kicks": 15 + i % 3,
            "handballs": 10, "goals": 1, "marks": 5,
        }
        for i in range(1, 11)
        for j in range(2)
    ])

    class _Provider:
        @property
        def fixtures_df(self):      return all_fixtures
        @property
        def teams_df(self):         return teams
        @property
        def players_df(self):       return players
        @property
        def team_stats_df(self):    return team_stats
        @property
        def player_stats_df(self):  return player_stats

        def load_upcoming_fixtures_df(self):
            return all_fixtures[all_fixtures["status"] == "upcoming"].copy()

    return _Provider()


@pytest.fixture(scope="module")
def client():
    # Patch _make_afl_provider so DataLoader uses in-memory data
    # Also patch PreflightService so no real API key is needed
    with patch(
        "app.data_ingestion.loader._make_afl_provider",
        side_effect=_in_memory_afl_provider,
    ), patch(
        "app.services.pipeline.PreflightService",
    ) as mock_pf_cls:
        mock_pf_cls.return_value.run.return_value = None
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        r = client.get("/api/v1/health")
        data = r.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestSummaryEndpoint:
    def test_summary_returns_200(self, client):
        r = client.get("/api/v1/reports/summary")
        assert r.status_code == 200

    def test_summary_has_data(self, client):
        r = client.get("/api/v1/reports/summary")
        data = r.json()
        assert "data_summary" in data
        assert "models_available" in data
        assert "settings" in data


class TestPipelineEndpoint:
    def test_pipeline_returns_200(self, client):
        r = client.post("/api/v1/pipeline/run")
        assert r.status_code == 200

    def test_pipeline_response_structure(self, client):
        r = client.post("/api/v1/pipeline/run")
        data = r.json()
        assert "run_id" in data
        assert "value_legs" in data
        assert "value_multis" in data
        assert isinstance(data["value_legs"], list)


class TestLegsEndpoint:
    def test_get_legs_value(self, client):
        client.post("/api/v1/pipeline/run")
        r = client.get("/api/v1/legs?mode=value")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_legs_safe(self, client):
        r = client.get("/api/v1/legs?mode=safe")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestMultisEndpoint:
    def test_get_multis_value(self, client):
        r = client.get("/api/v1/multis?mode=value")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_get_multis_same_game(self, client):
        r = client.get("/api/v1/multis?mode=same_game")
        assert r.status_code == 200

    def test_generate_multis(self, client):
        r = client.post("/api/v1/multis/generate", json={
            "legs": [
                {
                    "leg_id": "L_test1", "fixture_id": 64, "player_id": None,
                    "team_id": 1, "market_type": "head_to_head",
                    "selection": "home_win", "decimal_odds": 1.90,
                    "model_probability": 0.60, "ev": 0.14,
                    "confidence_score": 65.0, "explanation": "Test",
                },
                {
                    "leg_id": "L_test2", "fixture_id": 65, "player_id": None,
                    "team_id": 2, "market_type": "head_to_head",
                    "selection": "home_win", "decimal_odds": 2.10,
                    "model_probability": 0.55, "ev": 0.16,
                    "confidence_score": 58.0, "explanation": "Test",
                },
            ],
            "mode": "value",
            "max_results": 5,
        })
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestTrainingEndpoint:
    def test_training_returns_200(self, client):
        r = client.post("/api/v1/training/run")
        assert r.status_code == 200

    def test_training_has_run_id(self, client):
        r = client.post("/api/v1/training/run")
        data = r.json()
        assert "run_id" in data


class TestBacktestEndpoint:
    def test_backtest_returns_200(self, client):
        r = client.post("/api/v1/backtest/run")
        assert r.status_code == 200

    def test_backtest_has_status(self, client):
        r = client.post("/api/v1/backtest/run")
        data = r.json()
        assert "status" in data
