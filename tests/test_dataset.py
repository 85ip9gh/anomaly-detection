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


def test_the_cut_snaps_to_a_hole_in_the_archive():
    """Train and test should hold different stretches of machine time.

    Cutting at exactly 70% would put the first half of one continuous stretch in
    training and the second half of the same stretch in testing, which is the
    weakest possible test: the model is asked about a situation it was fitted on
    the beginning of.
    """
    early = [window(i, 5.0) for i in range(6)]           # 6 windows, day 19
    late = [window(i, 5.0, day=21) for i in range(4)]    # 4 windows, day 21
    train, test = split_by_time(early + late, train_fraction=0.7)
    assert len(train) == 6 and len(test) == 4
    assert {w.start[:10] for w in train} == {"2026-08-19"}
    assert {w.start[:10] for w in test} == {"2026-08-21"}


def test_snapping_can_be_turned_off():
    early = [window(i, 5.0) for i in range(6)]
    late = [window(i, 5.0, day=21) for i in range(4)]
    train, test = split_by_time(early + late, train_fraction=0.7, snap_to_gap=False)
    assert len(train) == 7 and len(test) == 3


def test_a_gapless_series_is_split_where_asked():
    windows = [window(i, 5.0) for i in range(10)]
    train, test = split_by_time(windows, train_fraction=0.7)
    assert len(train) == 7 and len(test) == 3
