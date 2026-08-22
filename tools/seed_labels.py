"""Derive the starting label set from an external run ledger, not from the signal.

This is the file that decides whether the whole project means anything. If
labels come from looking at the telemetry and calling the spiky parts anomalous,
then the model is scored on its agreement with a spike detector and the F1 is
arithmetic performed on a tautology. So every interval here is justified by
something outside the telemetry entirely:

  jobspy-search/logs/nightly-history.tsv       when a scan was started
  jobspy-search/logs/nightly-2026-08-21.log    "finished in 12 min"
  Vault/job-search-data/*.csv modification times, which bound the extraction
                                               stage: the screened file is
                                               written when scraping ends and
                                               the extracted file when the model
                                               pass ends

None of those knows anything about CPU or memory. That is the point.

The intervals are in the collector host's local time (ADT, UTC-3) because that
is what the ledger records; they are converted once, here, and stored as UTC.

Windows are labelled 1 when they overlap a known run, and 0 only where the
ledger records no run AND the period is otherwise accounted for. Windows that
are neither are left out of the file rather than assumed quiet: "no scan was
running" is not the same as "nothing was happening", and a silent 0 on a window
that was actually a Docker build is a wrong label that nothing would ever catch.

Usage:
    python tools/seed_labels.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.dataset import load_windows  # noqa: E402
from anomaly.features import parse_ts  # noqa: E402
from anomaly.label_cli import write_labels  # noqa: E402

LOCAL = timezone(timedelta(hours=-3))  # ADT


def at(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL)


# (start_local, end_local, label, event, source)
INTERVALS = [
    (at("2026-08-19 02:19"), at("2026-08-19 15:36"), 1,
     "jobspy nightly scan, remote scope, model extraction stage",
     "nightly-history.tsv start 02:00; screened csv mtime 02:19 ends scraping; "
     "extracted csv mtime 15:36 ends extraction"),
    (at("2026-08-21 17:00"), at("2026-08-21 17:11"), 1,
     "jobspy evening Halifax pass",
     "nightly-history.tsv start 17:00; nightly-2026-08-21.log 'finished in 12 min'; "
     "shortlist csv mtime 17:11"),
    (at("2026-08-21 12:10"), at("2026-08-21 16:34"), 0,
     "",
     "no run in nightly-history.tsv between 08-20 23:30 and 08-21 17:00; "
     "desk work only, corroborated by jobspy-search commits at 16:44"),
    (at("2026-08-19 15:41"), at("2026-08-19 19:04"), 0,
     "",
     "extraction finished at 15:36 and the shortlist was written 15:41; "
     "next run in nightly-history.tsv is 23:30"),
]


def main() -> None:
    windows = load_windows()
    rows = []
    for w in windows:
        start, end = parse_ts(w.start), parse_ts(w.end)
        for i_start, i_end, label, event, source in INTERVALS:
            # overlap, not containment: a 5 minute window straddling the end of a
            # run is genuinely part anomalous, and calling it normal teaches the
            # opposite of what happened.
            if start < i_end and end > i_start:
                rows.append({
                    "window_start": w.start, "window_end": w.end, "host": w.host,
                    "label": str(label), "event": event, "source": source,
                    "note": "seeded by tools/seed_labels.py",
                })
                break
    write_labels(rows)
    positives = sum(1 for r in rows if r["label"] == "1")
    print(f"{len(windows)} windows, {len(rows)} labelled, "
          f"{positives} positive, {len(rows) - positives} negative, "
          f"{len(windows) - len(rows)} deliberately left unlabelled")


if __name__ == "__main__":
    main()
