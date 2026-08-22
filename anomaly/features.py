"""Turn a stream of readings into fixed-length windows of features.

A single reading says almost nothing: CPU at 80% is a build, a video call, or
nothing at all. The window is the unit that carries shape, so it is the unit the
model scores and the unit a label attaches to.

Two things here are easy to get wrong and are handled explicitly:

  Counters vs gauges. bytes_recv is cumulative and resets when the host reboots,
  so a raw difference produces a huge negative number exactly at the moment
  something interesting happened. Rates are computed per second and a negative
  difference is treated as a counter reset rather than as traffic.

  Coverage. A window is not comparable to its neighbours if half its readings
  are missing, and missing readings are themselves a signal (the collector or
  the machine was down). `coverage` records the fraction of expected readings
  present, so a gap is a feature rather than a silent hole.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from statistics import fmean, pstdev

# 10 s cadence for sentinel.system, so 30 readings is a 5 minute window.
CADENCE_SECONDS = 10.0
WINDOW_READINGS = 30

GAUGES = ("cpu_percent", "memory_percent", "swap_percent", "process_count", "disk_percent")
COUNTERS = ("net_bytes_recv", "net_bytes_sent", "net_dropin", "net_errin")

FEATURE_NAMES = (
    tuple(f"{g}_{stat}" for g in GAUGES for stat in ("mean", "max", "std"))
    + tuple(f"{c}_rate_mean" for c in COUNTERS)
    + tuple(f"{c}_rate_max" for c in COUNTERS)
    + ("process_count_delta", "coverage", "reboot")
)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    host: str
    readings: int
    features: dict[str, float] = field(default_factory=dict)

    def vector(self) -> list[float]:
        return [self.features[name] for name in FEATURE_NAMES]


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]


def _rates(rows: list[dict], key: str) -> list[float]:
    """Per-second rate between consecutive readings, counter resets dropped."""
    out = []
    for a, b in zip(rows, rows[1:]):
        va, vb = a.get(key), b.get(key)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        dt = (parse_ts(b["ts"]) - parse_ts(a["ts"])).total_seconds()
        if dt <= 0:
            continue
        delta = vb - va
        if delta < 0:  # counter reset, almost always a reboot
            continue
        out.append(delta / dt)
    return out


def window_features(rows: list[dict]) -> dict[str, float]:
    feats: dict[str, float] = {}
    for gauge in GAUGES:
        s = _series(rows, gauge)
        feats[f"{gauge}_mean"] = fmean(s) if s else 0.0
        feats[f"{gauge}_max"] = max(s) if s else 0.0
        feats[f"{gauge}_std"] = pstdev(s) if len(s) > 1 else 0.0
    for counter in COUNTERS:
        r = _rates(rows, counter)
        feats[f"{counter}_rate_mean"] = fmean(r) if r else 0.0
        feats[f"{counter}_rate_max"] = max(r) if r else 0.0

    procs = _series(rows, "process_count")
    feats["process_count_delta"] = (procs[-1] - procs[0]) if len(procs) > 1 else 0.0

    span = (parse_ts(rows[-1]["ts"]) - parse_ts(rows[0]["ts"])).total_seconds()
    expected = max(span / CADENCE_SECONDS + 1, 1.0)
    feats["coverage"] = min(len(rows) / expected, 1.0)

    uptimes = _series(rows, "uptime_seconds")
    feats["reboot"] = 1.0 if any(b < a for a, b in zip(uptimes, uptimes[1:])) else 0.0
    return feats


def to_windows(rows: list[dict], size: int = WINDOW_READINGS,
               max_gap_seconds: float = CADENCE_SECONDS * 6) -> list[Window]:
    """Non-overlapping windows, never spanning a gap longer than max_gap_seconds.

    Windows do not overlap because an overlapping window shares readings with the
    one before it, and a time-based train/test split then leaks the tail of
    training into the head of testing.
    """
    rows = sorted(rows, key=lambda r: r["ts"])
    segments: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        if current:
            gap = (parse_ts(row["ts"]) - parse_ts(current[-1]["ts"])).total_seconds()
            if gap > max_gap_seconds:
                segments.append(current)
                current = []
        current.append(row)
    if current:
        segments.append(current)

    windows: list[Window] = []
    for segment in segments:
        for i in range(0, len(segment) - size + 1, size):
            chunk = segment[i:i + size]
            windows.append(Window(
                start=chunk[0]["ts"],
                end=chunk[-1]["ts"],
                host=chunk[0].get("host") or "unknown",
                readings=len(chunk),
                features=window_features(chunk),
            ))
    return windows
