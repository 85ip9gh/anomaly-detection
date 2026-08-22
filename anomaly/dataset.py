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
import json
from pathlib import Path

from .features import Window, to_windows

REPO = Path(__file__).resolve().parents[1]
SLICE = REPO / "data" / "system-slice.ndjson"
LABELS = REPO / "labels" / "labels.csv"


def load_rows(path: Path = SLICE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_windows(path: Path = SLICE) -> list[Window]:
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


def split_by_time(windows: list[Window], train_fraction: float = 0.7
                  ) -> tuple[list[Window], list[Window]]:
    ordered = sorted(windows, key=lambda w: w.start)
    cut = int(len(ordered) * train_fraction)
    return ordered[:cut], ordered[cut:]


def labelled(windows: list[Window], labels: dict[str, dict]) -> list[tuple[Window, int]]:
    out = []
    for w in windows:
        row = labels.get(w.start)
        if row and row.get("label") in ("0", "1"):
            out.append((w, int(row["label"])))
    return out
