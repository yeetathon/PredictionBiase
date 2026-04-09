"""Tests for FastAPI endpoints.

The pipeline/training/backtest endpoints internally call DataLoader(), which
would try to instantiate SportradarLoader without an API key. We patch
_make_afl_provider to return a DemoAFLDataProvider so all tests run offline.
"""
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

DEMO_DATA = Path(__file__).parent / "demo_data"


def _demo_afl_provider():
    from app.data_ingestion.demo_loader import DemoAFLDataProvider
    return DemoAFLDataProvider(DEMO_DATA)


@pytest.fixture(scope="module")
def client():
    # Patch _make_afl_provider so DataLoader uses demo data in the API process
    with patch(
        "app.data_ingestion.loader._make_afl_provider",
        side_effect=_demo_afl_provider,
    ):
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
