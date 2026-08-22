"""Isolation Forest over window features. Deliberately boring.

The evidence this project is meant to produce is the method, not the algorithm:
a pinned dataset, labels that came from somewhere other than the signal, a
time-based split, and a baseline to lose to. Swapping the estimator changes
almost none of that, which is why the estimator is the least interesting file
here.

Fitting happens on training windows only, and the scaler is fitted on training
windows only as well. Fitting a scaler on everything is a quiet and very common
leak: the test set's mean and variance end up inside the transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, Window


@dataclass
class Detector:
    contamination: float = 0.05
    n_estimators: int = 200
    random_state: int = 0

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    @staticmethod
    def matrix(windows: list[Window]) -> np.ndarray:
        return np.asarray([w.vector() for w in windows], dtype=float)

    def fit(self, train: list[Window]) -> "Detector":
        x = self.scaler.fit_transform(self.matrix(train))
        self.forest.fit(x)
        return self

    def score(self, windows: list[Window]) -> list[float]:
        """Higher means more anomalous, so the sign is flipped from sklearn's."""
        x = self.scaler.transform(self.matrix(windows))
        return [-v for v in self.forest.score_samples(x)]

    def predict(self, windows: list[Window]) -> list[int]:
        x = self.scaler.transform(self.matrix(windows))
        return [1 if v == -1 else 0 for v in self.forest.predict(x)]

    def feature_ranking(self, windows: list[Window]) -> list[tuple[str, float]]:
        """Crude but honest attribution: mean absolute z of the flagged windows.

        Not a SHAP value and not claimed to be one. It answers "what was unusual
        about the windows this flagged", which is the question a person looking
        at an alert actually asks.
        """
        x = self.scaler.transform(self.matrix(windows))
        flagged = x[np.asarray(self.predict(windows), dtype=bool)]
        if flagged.size == 0:
            return []
        magnitude = np.abs(flagged).mean(axis=0)
        return sorted(zip(FEATURE_NAMES, magnitude), key=lambda p: -p[1])
