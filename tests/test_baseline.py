from datetime import datetime, timedelta, timezone

from anomaly.baseline import RollingSigmaBaseline
from anomaly.features import Window, FEATURE_NAMES


def window(i, cpu):
    ts = datetime(2026, 8, 19, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats["cpu_percent_mean"] = cpu
    return Window(start=ts.isoformat().replace("+00:00", "Z"),
                  end=ts.isoformat().replace("+00:00", "Z"),
                  host="cubebox", readings=30, features=feats)


def test_the_first_windows_cannot_be_judged_and_are_not():
    windows = [window(i, 5.0 + (i % 3)) for i in range(20)]
    baseline = RollingSigmaBaseline(history=12)
    assert baseline.score(windows)[:12] == [0.0] * 12


def test_a_spike_after_a_quiet_stretch_fires():
    windows = [window(i, 5.0 + (i % 3) * 0.1) for i in range(13)]
    windows.append(window(13, 90.0))
    assert RollingSigmaBaseline(history=12).predict(windows)[-1] == 1


def test_a_flat_series_never_fires():
    windows = [window(i, 5.0) for i in range(30)]
    assert sum(RollingSigmaBaseline(history=12).predict(windows)) == 0


def test_scoring_uses_only_the_past():
    """Appending later windows must not change an earlier window's score."""
    early = [window(i, 5.0 + (i % 4)) for i in range(20)]
    baseline = RollingSigmaBaseline(history=12)
    before = baseline.score(early)
    after = baseline.score(early + [window(i, 99.0) for i in range(20, 25)])
    assert before == after[:20]
