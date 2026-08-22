"""The labelling tool, written before the model on purpose.

The PRD's own risk section says labelling is the step most likely to be skipped
and that skipping it removes the point of the project. The counter to that is
making labelling cheap: this prints one window at a time with the numbers a
person needs to judge it, and writes the answer straight to a versioned CSV.

Two modes:

    review     walk windows in time order and label them by keystroke
    queue      print the windows an unsupervised pass finds most unusual

`queue` exists to order the work, not to do it. A window appearing in the queue
is not evidence of anything: labelling it 1 because the model ranked it high is
the circularity this whole design is trying to avoid. The queue is for deciding
which five minutes of the day to go and check against something external, a run
log, a calendar, a commit time.

Usage:
    python -m anomaly.label_cli review
    python -m anomaly.label_cli queue --top 20
"""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path

from .dataset import LABELS, load_labels, load_windows, split_by_time
from .features import Window, parse_ts

FIELDS = ("window_start", "window_end", "host", "label", "event", "source", "note")
LOCAL_OFFSET = timedelta(hours=-3)  # ADT, the collector host's wall clock


def describe(w: Window) -> str:
    f = w.features
    local = parse_ts(w.start) + LOCAL_OFFSET
    return (
        f"{w.start}  (local {local:%Y-%m-%d %H:%M})  host={w.host}\n"
        f"  cpu     mean {f['cpu_percent_mean']:6.1f}%  max {f['cpu_percent_max']:6.1f}%\n"
        f"  memory  mean {f['memory_percent_mean']:6.1f}%  max {f['memory_percent_max']:6.1f}%\n"
        f"  procs   mean {f['process_count_mean']:6.1f}   delta {f['process_count_delta']:+.0f}\n"
        f"  net in  mean {f['net_bytes_recv_rate_mean'] / 1e6:6.2f} MB/s  "
        f"max {f['net_bytes_recv_rate_max'] / 1e6:6.2f} MB/s\n"
        f"  swap    mean {f['swap_percent_mean']:6.2f}%  coverage {f['coverage']:.2f}"
    )


def write_labels(rows: list[dict], path: Path = LABELS) -> None:
    path.parent.mkdir(exist_ok=True)
    rows = sorted(rows, key=lambda r: r["window_start"])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def review(args) -> None:
    windows = load_windows()
    existing = load_labels()
    rows = list(existing.values())
    for w in windows:
        if w.start in existing and not args.relabel:
            continue
        print("\n" + describe(w))
        answer = input("  [n]ormal / [a]nomaly / [s]kip / [q]uit > ").strip().lower()
        if answer.startswith("q"):
            break
        if answer.startswith("s"):
            continue
        label = 1 if answer.startswith("a") else 0
        event = input("  event (what was happening) > ").strip() if label else ""
        source = input("  source (where that came from) > ").strip() if label else "reviewed-quiet"
        rows = [r for r in rows if r["window_start"] != w.start]
        rows.append({"window_start": w.start, "window_end": w.end, "host": w.host,
                     "label": str(label), "event": event, "source": source, "note": ""})
        write_labels(rows)
    print(f"\nwrote {len(rows)} labels to {LABELS}")


def queue(args) -> None:
    from .model import Detector

    windows = load_windows()
    train, _ = split_by_time(windows)
    detector = Detector().fit(train)
    scored = sorted(zip(windows, detector.score(windows)), key=lambda p: -p[1])
    labelled = load_labels()
    print(f"{len(windows)} windows, {len(labelled)} already labelled.")
    print("Ranked by an unsupervised score. Rank is a reading order, not a label.\n")
    shown = 0
    for w, s in scored:
        if w.start in labelled and not args.all:
            continue
        print(f"[score {s:.3f}] {describe(w)}\n")
        shown += 1
        if shown >= args.top:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("review")
    r.add_argument("--relabel", action="store_true")
    r.set_defaults(func=review)
    q = sub.add_parser("queue")
    q.add_argument("--top", type=int, default=20)
    q.add_argument("--all", action="store_true")
    q.set_defaults(func=queue)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
