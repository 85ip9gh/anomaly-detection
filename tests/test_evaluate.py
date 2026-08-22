import pytest

from anomaly.evaluate import score, usable


def test_counts_and_metrics():
    s = score([1, 1, 0, 0], [1, 0, 1, 0])
    assert (s.true_positive, s.false_positive, s.false_negative, s.true_negative) == (1, 1, 1, 1)
    assert s.precision == 0.5 and s.recall == 0.5 and s.f1 == 0.5


def test_flagging_everything_buys_recall_and_nothing_else():
    s = score([1, 0, 0, 0], [1, 1, 1, 1])
    assert s.recall == 1.0
    assert s.precision == 0.25


def test_no_predictions_is_zero_not_a_crash():
    s = score([1, 0], [0, 0])
    assert s.precision == 0.0 and s.recall == 0.0 and s.f1 == 0.0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError):
        score([1, 0], [1])


def test_a_handful_of_positives_is_not_evaluable():
    ok, why = usable([(None, 1), (None, 0), (None, 0)])
    assert not ok and "fewer than" in why
    ok, _ = usable([(None, 1)] * 5 + [(None, 0)] * 5)
    assert ok
