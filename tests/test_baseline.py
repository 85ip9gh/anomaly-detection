from datetime import datetime, timedelta, timezone

from anomaly.baseline import RollingSigmaBaseline
from anomaly.features import Window, FEATURE_NAMES


def window(i, cpu, day=19):
    """A five minute window at slot `i`, shaped like one `to_windows` produces.

    `end` is the last reading's timestamp, 290 seconds after the first, not the
    slot boundary. That gap matters now that the baseline resets its history
    across holes in the archive: a window whose end equals its start looks like
    a five minute hole to everything downstream.
    """
    ts = datetime(2026, 8, day, tzinfo=timezone.utc) + timedelta(minutes=5 * i)
    feats = {name: 0.0 for name in FEATURE_NAMES}
    feats["cpu_percent_mean"] = cpu
    return Window(start=ts.isoformat().replace("+00:00", "Z"),
                  end=(ts + timedelta(seconds=290)).isoformat().replace("+00:00", "Z"),
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


def test_history_resets_across_a_hole_in_the_archive():
    """A machine that was switched off for two days has no trailing hour.

    Without the reset the first window back compares against how the machine
    looked before the outage, fires, and the baseline loses precision to a
    handicap rather than to the model. Beating a handicapped opponent proves
    nothing, which is the whole reason this file exists.
    """
    quiet = [window(i, 5.0 + (i % 3) * 0.1) for i in range(13)]
    after_gap = [window(i, 90.0, day=21) for i in range(3)]
    scores = RollingSigmaBaseline(history=12).score(quiet + after_gap)
    assert scores[-3:] == [0.0, 0.0, 0.0]


def test_without_the_reset_the_same_windows_do_fire():
    """The reset is doing work, not quietly zeroing everything."""
    quiet = [window(i, 5.0 + (i % 3) * 0.1) for i in range(13)]
    after_gap = [window(i, 90.0, day=21) for i in range(3)]
    unreset = RollingSigmaBaseline(history=12, reset_on_gap=False)
    # Index 13 is the first window after the gap. Later ones do not fire even
    # without the reset, because by then the spike is inside its own history.
    assert unreset.predict(quiet + after_gap)[13] == 1
