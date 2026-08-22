"""Load the exported slice and split it by time, never at random.

The split rule is the one thing in this file worth arguing about. A random split
on a time series puts readings from 14:05 in training and 14:04 in testing, and
the model then scores well because it has already seen the neighbourhood of
every test point. The number that comes out is real arithmetic on a meaningless
experiment. Splitting by time is the only split that answers the question the
service will actually face: given the past, judge something that has not
happened yet.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from .features import CADENCE_SECONDS, Window, parse_ts, to_windows

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
LABELS = REPO / "labels" / "labels.csv"

# The same threshold `to_windows` uses to refuse to span a hole. Six missed
# readings is a minute of silence, which is far more than jitter and far less
# than the hours this archive actually goes dark for.
GAP_SECONDS = CADENCE_SECONDS * 6


def slice_path(data: Path = DATA, kind: str = "system") -> Path:
    """The exported slice, gzipped or not.

    Both spellings are accepted because the gzipped one only arrived when the
    dataset grew from hours to days. A checkout that predates that still loads.
    """
    for name in (f"{kind}-slice.ndjson.gz", f"{kind}-slice.ndjson"):
        candidate = data / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no {kind} slice in {data}. Run tools/export_slice.py first."
    )


def load_rows(path: Path | None = None) -> list[dict]:
    path = path or slice_path()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_windows(path: Path | None = None) -> list[Window]:
    return to_windows(load_rows(path))


def load_labels(path: Path = LABELS) -> dict[str, dict]:
    """Keyed by window start timestamp, which is what the label CSV records."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["window_start"]] = row
    return out


def segment_boundaries(windows: list[Window],
                       max_gap_seconds: float = GAP_SECONDS) -> list[int]:
    """Indices where a new contiguous stretch of archive begins.

    The collector only publishes while its host is awake, so the archive is not
    one continuous series: it is several stretches separated by hours or days of
    nothing. Those gaps are where a split belongs.
    """
    ordered = sorted(windows, key=lambda w: w.start)
    cuts = []
    for i in range(1, len(ordered)):
        gap = (parse_ts(ordered[i].start) - parse_ts(ordered[i - 1].end)).total_seconds()
        if gap > max_gap_seconds:
            cuts.append(i)
    return cuts


def split_by_time(windows: list[Window], train_fraction: float = 0.7,
                  snap_to_gap: bool = True) -> tuple[list[Window], list[Window]]:
    """Split by time, and prefer to cut where the archive is already broken.

    Cutting at ``train_fraction`` exactly will usually land in the middle of a
    contiguous stretch of machine time, which puts the first hours of one event
    in training and the last hours of the same event in testing. That is not the
    random-split leak (no labels are used to fit), but it is still the weakest
    possible test: the model is asked about a situation it was fitted on the
    beginning of.

    Snapping the cut to the nearest gap in the archive removes that. Train and
    test then contain different stretches of machine time, separated by hours the
    machine spent switched off, which is as clean a boundary as this data has.
    The realised fraction moves as a result, so callers should report the sizes
    rather than assume 70/30. With no gaps at all the behaviour is unchanged.
    """
    ordered = sorted(windows, key=lambda w: w.start)
    cut = int(len(ordered) * train_fraction)
    if snap_to_gap:
        candidates = segment_boundaries(ordered)
        # A boundary at 0 or at the end would empty one side, which is never a
        # split. Prefer the closest surviving candidate to the requested cut.
        candidates = [c for c in candidates if 0 < c < len(ordered)]
        if candidates:
            cut = min(candidates, key=lambda c: abs(c - cut))
    return ordered[:cut], ordered[cut:]


def labelled(windows: list[Window], labels: dict[str, dict]) -> list[tuple[Window, int]]:
    out = []
    for w in windows:
        row = labels.get(w.start)
        if row and row.get("label") in ("0", "1"):
            out.append((w, int(row["label"])))
    return out
