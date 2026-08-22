from datetime import datetime, timedelta, timezone

from anomaly.features import CADENCE_SECONDS, to_windows, window_features


def reading(i, **over):
    ts = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc) + timedelta(seconds=10 * i)
    row = {
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "host": "cubebox",
        "cpu_percent": 5.0, "memory_percent": 30.0, "swap_percent": 0.0,
        "process_count": 250, "uptime_seconds": 1000.0 + 10 * i, "disk_percent": 44.0,
        "net_bytes_recv": 1000 * i, "net_bytes_sent": 500 * i,
        "net_dropin": 0, "net_errin": 0,
    }
    row.update(over)
    return row


def test_windows_do_not_overlap_or_span_a_gap():
    rows = [reading(i) for i in range(30)]
    late = reading(60)  # 5 minutes after the last one, far past the gap tolerance
    windows = to_windows(rows + [late], size=10)
    assert [w.readings for w in windows] == [10, 10, 10]
    starts = [w.start for w in windows]
    assert starts == sorted(starts)
    assert all(a.end < b.start for a, b in zip(windows, windows[1:]))


def test_counter_reset_is_not_traffic():
    rows = [reading(i) for i in range(5)]
    rows[3]["net_bytes_recv"] = 0        # reboot: the counter starts over
    rows[4]["net_bytes_recv"] = 1000
    feats = window_features(rows)
    assert feats["net_bytes_recv_rate_max"] == 100.0  # 1000 bytes over 10 s, not a spike


def test_reboot_flag_follows_uptime_going_backwards():
    rows = [reading(i) for i in range(5)]
    assert window_features(rows)["reboot"] == 0.0
    rows[3]["uptime_seconds"] = 5.0
    assert window_features(rows)["reboot"] == 1.0


def test_coverage_falls_when_readings_are_missing():
    full = [reading(i) for i in range(10)]
    assert window_features(full)["coverage"] == 1.0
    sparse = [full[0], full[3], full[6], full[9]]
    assert window_features(sparse)["coverage"] < 0.5


def test_cadence_matches_the_collector():
    assert CADENCE_SECONDS == 10.0
