import pytest

from anomaly.evaluate import score, usable
from tests.test_baseline import window


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


def test_a_set_with_no_negatives_is_refused():
    """All-positive is as unusable as all-negative, and less obviously so.

    Precision cannot be wrong when there is nothing to be wrong about, and
    roc_auc_score raises rather than returning a number, so the ranking column
    silently vanishes from the report while precision and recall stay on the
    page looking like results.
    """
    pairs = [(window(i, 5.0), 1) for i in range(20)]
    ok, why = usable(pairs)
    assert not ok
    assert "negative" in why


def test_a_balanced_set_is_accepted():
    pairs = [(window(i, 5.0), i % 2) for i in range(20)]
    ok, why = usable(pairs)
    assert ok
    assert "10 positive" in why and "10 negative" in why
