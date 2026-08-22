"""Save and load a fitted detector, with the facts needed to distrust it.

A model file on its own is not a model. Without the feature order it was fitted
on, the split it was fitted from and the metrics it earned, an artifact is a
black box that will happily score a vector of the wrong width and return a
number. So the metadata travels with it, `load` refuses a mismatch, and the same
metadata is what the service reports in every response.

The threshold is stored explicitly rather than left inside the estimator.
Isolation Forest places its own boundary from the `contamination` it was given,
and callers should be able to read the number the service is comparing against
rather than infer it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .features import FEATURE_NAMES
from .model import Detector

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "artifacts" / "detector.joblib"


@dataclass(frozen=True)
class Artifact:
    detector: Detector
    threshold: float
    feature_names: tuple[str, ...]
    kept_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    trained_at: str
    train_windows: int
    train_first_ts: str
    train_last_ts: str
    dataset_sha256: str
    metrics: dict

    def metadata(self) -> dict:
        return {
            "threshold": self.threshold,
            "feature_names": list(self.feature_names),
            "kept_features": list(self.kept_features),
            "dropped_features": list(self.dropped_features),
            "trained_at": self.trained_at,
            "train_windows": self.train_windows,
            "train_first_ts": self.train_first_ts,
            "train_last_ts": self.train_last_ts,
            "dataset_sha256": self.dataset_sha256,
            "metrics": self.metrics,
        }


def threshold_of(detector: Detector) -> float:
    """The score above which `Detector.predict` returns 1.

    `Detector.score` negates sklearn's `score_samples`, and sklearn flags where
    `score_samples - offset_ < 0`. Negating both sides puts the boundary at
    `-offset_` in this sign convention. There is a test that pins the identity,
    because a sign error here would move every decision the service makes while
    leaving every number it prints looking reasonable.
    """
    return float(-detector.forest.offset_)


def save(detector: Detector, *, train_windows: int, train_first_ts: str,
         train_last_ts: str, dataset_sha256: str, metrics: dict,
         path: Path = DEFAULT_PATH) -> Artifact:
    import joblib

    artifact = Artifact(
        detector=detector,
        threshold=threshold_of(detector),
        feature_names=FEATURE_NAMES,
        kept_features=tuple(detector.kept_names()),
        dropped_features=tuple(detector.dropped),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_windows=train_windows,
        train_first_ts=train_first_ts,
        train_last_ts=train_last_ts,
        dataset_sha256=dataset_sha256,
        metrics=metrics,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"detector": detector, **artifact.metadata()}, path)
    path.with_suffix(".json").write_text(
        json.dumps(artifact.metadata(), indent=2) + "\n", encoding="utf-8")
    return artifact


def load(path: Path = DEFAULT_PATH) -> Artifact:
    """Load an artifact, refusing one whose feature list is not this code's.

    An artifact trained before a feature was added still has a scaler and a
    forest, and both will accept a matrix of the wrong width in some sklearn
    versions and return a plausible number. Failing here is the difference
    between a wrong answer and no answer.
    """
    import joblib

    payload = joblib.load(path)
    names = tuple(payload["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(
            f"artifact was fitted on {len(names)} features and this code has "
            f"{len(FEATURE_NAMES)}. Retrain with tools/train_model.py."
        )
    return Artifact(
        detector=payload["detector"],
        threshold=float(payload["threshold"]),
        feature_names=names,
        kept_features=tuple(payload["kept_features"]),
        dropped_features=tuple(payload["dropped_features"]),
        trained_at=str(payload["trained_at"]),
        train_windows=int(payload["train_windows"]),
        train_first_ts=str(payload["train_first_ts"]),
        train_last_ts=str(payload["train_last_ts"]),
        dataset_sha256=str(payload["dataset_sha256"]),
        metrics=dict(payload["metrics"]),
    )
