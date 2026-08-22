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
    # Feature names to exclude before fitting. Only for the sensitivity check in
    # tools/report.py --without: naming a feature here after reading a result is
    # how a model gets tuned on its own test set, so it is never the default and
    # every run that uses it prints that it did.
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.scaler = StandardScaler()
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.kept: list[int] = list(range(len(FEATURE_NAMES)))
        self.dropped: list[str] = []

    @staticmethod
    def matrix(windows: list[Window]) -> np.ndarray:
        return np.asarray([w.vector() for w in windows], dtype=float)

    def fit(self, train: list[Window]) -> "Detector":
        """Fit on training windows only, scaler included.

        Features that never move in training are dropped rather than scaled.
        StandardScaler gives a zero-variance column a scale of 1.0, so any value
        it takes later is passed through raw: `disk_percent_mean` sitting at a
        constant 46 in training turns a 0.1 percentage point of disk growth into
        a z of 0.1 while a genuinely varying feature's z is a fraction of that.
        The column cannot carry information the forest can use, and it can carry
        magnitude, which is the worst combination available.
        """
        raw = self.matrix(train)
        std = raw.std(axis=0)
        usable = [
            i for i in range(raw.shape[1])
            if std[i] > 0 and FEATURE_NAMES[i] not in self.exclude
        ]
        self.kept = usable
        self.dropped = [FEATURE_NAMES[i] for i in range(raw.shape[1])
                        if i not in set(usable)]
        x = self.scaler.fit_transform(raw[:, self.kept])
        self.forest.fit(x)
        return self

    def _prepare(self, windows: list[Window]) -> np.ndarray:
        return self.scaler.transform(self.matrix(windows)[:, self.kept])

    def score(self, windows: list[Window]) -> list[float]:
        """Higher means more anomalous, so the sign is flipped from sklearn's."""
        return [-v for v in self.forest.score_samples(self._prepare(windows))]

    def predict(self, windows: list[Window]) -> list[int]:
        return [1 if v == -1 else 0 for v in self.forest.predict(self._prepare(windows))]

    def kept_names(self) -> list[str]:
        return [FEATURE_NAMES[i] for i in self.kept]

    def feature_ranking(self, windows: list[Window]) -> list[tuple[str, float]]:
        """Crude but honest attribution: mean absolute z of the flagged windows.

        Not a SHAP value and not claimed to be one. It answers "what was unusual
        about the windows this flagged", which is the question a person looking
        at an alert actually asks.

        Read it alongside `shift_report`. A feature can top this list because the
        flagged windows really were unusual on it, or because the whole scoring
        period sits at a different level from the training period, and only the
        second of those is a reason to distrust the number.
        """
        x = self._prepare(windows)
        flagged = x[np.asarray(self.predict(windows), dtype=bool)]
        if flagged.size == 0:
            return []
        magnitude = np.abs(flagged).mean(axis=0)
        return sorted(zip(self.kept_names(), magnitude), key=lambda p: -p[1])

    def shift_report(self, windows: list[Window]) -> list[tuple[str, float]]:
        """How far each feature's mean has moved, in training standard deviations.

        This is the covariate-shift check, and on this dataset it is the most
        important number in the file. Train and test are different weeks of the
        same machine, so a feature can sit at a different level in both without
        any window being anomalous. Where that offset is large it moves every
        test score by the same amount, which leaves ROC AUC and average
        precision untouched (they only read the ordering) and quietly
        invalidates precision, recall and F1, which read a fixed threshold.

        A large value here is not a bug to be scaled away. It is the reason to
        trust the ranking metrics over the thresholded ones.
        """
        x = self._prepare(windows)
        return sorted(zip(self.kept_names(), x.mean(axis=0)),
                      key=lambda p: -abs(p[1]))
