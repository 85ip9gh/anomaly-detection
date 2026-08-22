from anomaly.dataset import split_by_time
from tests.test_baseline import window


def test_split_is_chronological_and_disjoint():
    windows = [window(i, 5.0) for i in range(10)]
    train, test = split_by_time(windows, train_fraction=0.7)
    assert len(train) == 7 and len(test) == 3
    assert max(w.start for w in train) < min(w.start for w in test)


def test_split_does_not_depend_on_input_order():
    windows = [window(i, 5.0) for i in range(10)]
    a, b = split_by_time(windows)
    c, d = split_by_time(list(reversed(windows)))
    assert [w.start for w in a] == [w.start for w in c]
    assert [w.start for w in b] == [w.start for w in d]
