"""Model versioning and promotion service.

Maintains a registry of trained model versions in the ``model_versions`` DB
table and enforces the promotion rule:

    Promote challenger only if holdout Brier improves by > PROMOTION_THRESHOLD
    (default 0.002) compared to the current champion.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import ModelVersion


class ModelManager:
    """Manage model versions and champion promotion.

    Attributes
    ----------
    promotion_threshold:
        Minimum Brier score improvement required to promote a challenger model
        to champion status. Default comes from settings.
    artifacts_dir:
        Directory where serialised model files are stored.
    """

    def __init__(
        self,
        promotion_threshold: Optional[float] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        self.promotion_threshold = promotion_threshold or settings.model_promotion_brier_improvement
        self.artifacts_dir = artifacts_dir or settings.model_artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model_name: str,
        model_type: str,
        artifact_path: str,
        brier_score_val: float,
        brier_score_train: float = 0.0,
        log_loss_val: float = 0.0,
        n_train_samples: int = 0,
        n_val_samples: int = 0,
        hyperparameters: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> ModelVersion:
        """Register a freshly trained model as a challenger.

        The model starts with ``status="challenger"``; call :meth:`maybe_promote`
        to evaluate it against the current champion.
        """
        version_id = str(uuid.uuid4())[:12]
        with SessionLocal() as db:
            mv = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                model_type=model_type,
                artifact_path=artifact_path,
                brier_score_train=brier_score_train,
                brier_score_val=brier_score_val,
                log_loss_val=log_loss_val,
                n_train_samples=n_train_samples,
                n_val_samples=n_val_samples,
                hyperparameters=hyperparameters or {},
                feature_names=feature_names or [],
                extra_metadata=metadata or {},
                status="challenger",
                promoted_at=None,
            )
            db.add(mv)
            db.commit()
            db.refresh(mv)
            logger.info(
                "Registered challenger model: name={} version={} brier_val={:.4f}",
                model_name,
                version_id,
                brier_score_val,
            )
            return mv

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def maybe_promote(self, model_name: str) -> Dict[str, Any]:
        """Evaluate the latest challenger against the current champion.

        Returns a dict describing whether promotion occurred and why.
        """
        with SessionLocal() as db:
            champion = self._get_champion(db, model_name)
            challenger = self._get_latest_challenger(db, model_name)

            if challenger is None:
                return {"promoted": False, "reason": "no_challenger"}

            if champion is None:
                # First model — auto-promote
                return self._promote(db, challenger, reason="first_model")

            improvement = champion.brier_score_val - challenger.brier_score_val
            if improvement >= self.promotion_threshold:
                return self._promote(
                    db,
                    challenger,
                    reason=f"brier_improvement_{improvement:.4f}",
                    previous_champion=champion,
                )
            else:
                return {
                    "promoted": False,
                    "reason": "insufficient_improvement",
                    "champion_brier": champion.brier_score_val,
                    "challenger_brier": challenger.brier_score_val,
                    "improvement": improvement,
                    "threshold": self.promotion_threshold,
                }

    def _promote(
        self,
        db: Session,
        challenger: ModelVersion,
        reason: str,
        previous_champion: Optional[ModelVersion] = None,
    ) -> Dict[str, Any]:
        # Retire previous champion
        if previous_champion is not None:
            previous_champion.status = "retired"
            db.add(previous_champion)

        challenger.status = "champion"
        challenger.promoted_at = datetime.utcnow()
        db.add(challenger)
        db.commit()

        logger.info(
            "Promoted model {} version {} to champion (reason={})",
            challenger.model_name,
            challenger.version_id,
            reason,
        )
        return {
            "promoted": True,
            "reason": reason,
            "version_id": challenger.version_id,
            "brier_val": challenger.brier_score_val,
            "previous_champion_brier": previous_champion.brier_score_val if previous_champion else None,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_champion(self, model_name: str) -> Optional[ModelVersion]:
        with SessionLocal() as db:
            return self._get_champion(db, model_name)

    def list_versions(self, model_name: str, limit: int = 20) -> List[Dict]:
        with SessionLocal() as db:
            rows = (
                db.query(ModelVersion)
                .filter(ModelVersion.model_name == model_name)
                .order_by(ModelVersion.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._to_dict(r) for r in rows]

    def get_all_champions(self) -> List[Dict]:
        with SessionLocal() as db:
            rows = (
                db.query(ModelVersion)
                .filter(ModelVersion.status == "champion")
                .all()
            )
            return [self._to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_champion(self, db: Session, model_name: str) -> Optional[ModelVersion]:
        return (
            db.query(ModelVersion)
            .filter(
                ModelVersion.model_name == model_name,
                ModelVersion.status == "champion",
            )
            .order_by(ModelVersion.promoted_at.desc())
            .first()
        )

    def _get_latest_challenger(self, db: Session, model_name: str) -> Optional[ModelVersion]:
        return (
            db.query(ModelVersion)
            .filter(
                ModelVersion.model_name == model_name,
                ModelVersion.status == "challenger",
            )
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

    @staticmethod
    def _to_dict(mv: ModelVersion) -> Dict[str, Any]:
        return {
            "version_id": mv.version_id,
            "model_name": mv.model_name,
            "model_type": mv.model_type,
            "status": mv.status,
            "brier_score_val": mv.brier_score_val,
            "brier_score_train": mv.brier_score_train,
            "log_loss_val": mv.log_loss_val,
            "n_train_samples": mv.n_train_samples,
            "n_val_samples": mv.n_val_samples,
            "artifact_path": mv.artifact_path,
            "promoted_at": mv.promoted_at.isoformat() if mv.promoted_at else None,
            "created_at": mv.created_at.isoformat() if mv.created_at else None,
        }
