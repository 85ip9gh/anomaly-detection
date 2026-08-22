from datetime import datetime, timedelta, timezone

import pytest

from anomaly import artifact as artifact_module
from anomaly.features import FEATURE_NAMES, to_windows
from anomaly.model import Detector
from tests.test_service import reading


def fitted():
    rows = [reading(i, 5.0 + (i % 4)) for i in range(300)]
    return Detector().fit(to_windows(rows))


def test_the_stored_threshold_reproduces_predict():
    """A sign error here would move every decision and change no printed number.

    `Detector.score` negates sklearn's `score_samples`, and sklearn flags where
    `score_samples - offset_ < 0`, so the boundary sits at `-offset_` in this
    convention. The service compares against the stored number rather than
    calling `predict`, so the two have to agree.
    """
    detector = fitted()
    rows = [reading(i, 5.0 + (i % 4)) for i in range(300)]
    windows = to_windows(rows)
    threshold = artifact_module.threshold_of(detector)
    by_threshold = [1 if s > threshold else 0 for s in detector.score(windows)]
    assert by_threshold == detector.predict(windows)


def test_a_roundtrip_keeps_the_decisions_identical(tmp_path):
    detector = fitted()
    windows = to_windows([reading(i, 5.0 + (i % 4)) for i in range(300)])
    before = detector.score(windows)
    path = tmp_path / "detector.joblib"
    artifact_module.save(detector, train_windows=10, train_first_ts="a",
                         train_last_ts="b", dataset_sha256="x", metrics={},
                         path=path)
    loaded = artifact_module.load(path)
    assert loaded.detector.score(windows) == before
    assert loaded.threshold == artifact_module.threshold_of(detector)


def test_an_artifact_from_a_different_feature_set_is_refused(tmp_path, monkeypatch):
    """Silently scoring the wrong width is worse than not answering."""
    path = tmp_path / "detector.joblib"
    artifact_module.save(fitted(), train_windows=1, train_first_ts="a",
                         train_last_ts="b", dataset_sha256="x", metrics={},
                         path=path)
    monkeypatch.setattr(artifact_module, "FEATURE_NAMES",
                        FEATURE_NAMES + ("a_feature_added_later",))
    with pytest.raises(ValueError, match="fitted on"):
        artifact_module.load(path)
