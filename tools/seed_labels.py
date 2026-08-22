"""Derive the label set from an external run ledger, not from the signal.

This is the file that decides whether the whole project means anything. If
labels come from looking at the telemetry and calling the spiky parts anomalous,
then the model is scored on its agreement with a spike detector and the F1 is
arithmetic performed on a tautology. So every interval here is justified by
something outside the telemetry entirely:

  jobspy-search/logs/nightly-history.tsv       when each scan was started
  Vault/job-search-data/*.csv modification times, which bound each stage: a
                                               `-screened.csv` is written when
                                               scraping ends and a
                                               `-extracted.csv` when the model
                                               pass ends

Neither knows anything about CPU or memory. That is the point.

## What the label means

  1  a jobspy scan pipeline was running on cubebox
  0  no jobspy scan pipeline was running on cubebox

Deliberately narrow. It is *not* "the machine was anomalous" and *not* "the
machine was idle", and the narrowness is what makes it fully evidenced: the
ledger settles both classes on its own, with no judgement call about how busy is
busy.

The cost of that narrowness is real and worth stating: a 0 window may still have
had a Docker build, an agent session or a video call on it. Those are unlabelled
load sitting inside the negative class, and they can only make the task harder,
never easier. A detector that scores well here has separated the scan pipeline
from everything else the machine does, which is the stronger claim anyway.

## Three states, not two

Windows are labelled 1, labelled 0, or left out of the file. Two things are
deliberately left out rather than guessed at:

  A buffer of BUFFER_MINUTES either side of every run. Start times come from a
  scheduler and end times from file modification times, so the boundary is
  known to a minute or two at best, and a window that straddles it is genuinely
  part-run.

  Every window on 2026-08-22, the date this exporter was written. That night's
  machine state includes a Docker Desktop restart and a full Kafka replay
  performed *by the session building this dataset*. Labelling it 0 would be
  recording the measurement as if it were the thing measured.

The intervals are in the collector host's local time (ADT, UTC-3) because that
is what the ledger records; they are converted once, here, and stored as UTC.

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

BUFFER_MINUTES = 30

# The session that wrote this file restarted Docker Desktop and replayed the
# whole Kafka backlog on this date, so its telemetry is the measurement's own
# footprint rather than a normal night.
CONTAMINATED_FROM = datetime(2026, 8, 22, 0, 0, tzinfo=LOCAL)


def at(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL)


# (start_local, end_local, event, source)
#
# One entry per scheduled run in nightly-history.tsv. Where the ledger records a
# `local` and a `national`/`remote` scope started together, they are one interval
# because the second stage begins as the first ends and the machine never went
# quiet in between. Start is the ledger's own start time; end is the modification
# time of the last artefact that run produced.
#
# Two ledger rows have no interval here because cubebox published no telemetry
# on those dates at all: 2026-08-17 02:00 and 2026-08-18 02:00. Their scans ran
# and the shortlists exist, but the collector was not up, so there is nothing to
# label. Leaving them out is not a judgement about the runs, only about the
# archive.
RUNS = [
    (at("2026-08-16 02:00"), at("2026-08-16 04:08"),
     "jobspy nightly scan, local then remote scope",
     "nightly-history.tsv start 02:00 local and remote; "
     "local-indeed.csv mtime 02:00 ends local scraping; "
     "local-indeed-screened-extracted.csv 02:12 ends local extraction; "
     "remote-indeed.csv 02:13 ends remote scraping; "
     "remote-indeed-screened-extracted.csv 04:08 ends remote extraction"),

    (at("2026-08-17 02:00"), at("2026-08-17 04:10"),
     "jobspy nightly scan, local then remote scope",
     "nightly-history.tsv start 02:00; local extracted 02:13; "
     "remote-indeed.csv 02:14; remote extracted 04:10"),

    (at("2026-08-18 02:00"), at("2026-08-18 04:41"),
     "jobspy nightly scan, local then remote scope",
     "nightly-history.tsv start 02:00; local extracted 02:17; "
     "remote-indeed.csv 02:18; remote extracted 04:41"),

    (at("2026-08-19 02:00"), at("2026-08-19 15:41"),
     "jobspy nightly scan, remote scope, long model extraction stage",
     "nightly-history.tsv start 02:00; local extracted 02:18; "
     "remote-indeed.csv 02:19 ends scraping; "
     "remote-indeed-screened-extracted.csv 15:36 ends extraction; "
     "remote-indeed-shortlist.csv 15:41 written last"),

    (at("2026-08-19 23:30"), at("2026-08-20 08:15"),
     "jobspy evening pass, local combined then national",
     "nightly-history.tsv start 23:30 local and national; "
     "local-indeed 23:30, companies 23:39, careerbeacon 23:42, "
     "linkedin and combined 23:57; local-combined-extracted 08-20 00:27; "
     "national-indeed 00:29, companies 00:40, linkedin and combined 02:47, "
     "national-combined-extracted 07:29, national-combined-shortlist 08:15"),

    (at("2026-08-20 17:00"), at("2026-08-20 17:09"),
     "jobspy evening Halifax pass",
     "nightly-history.tsv start 17:00; "
     "local-indeed-screened-extracted.csv and shortlist mtime 17:09"),

    (at("2026-08-20 23:30"), at("2026-08-21 06:41"),
     "jobspy evening pass, local combined then national",
     "nightly-history.tsv start 23:30 local and national; "
     "local-indeed 23:30, companies 23:51, careerbeacon 23:56, "
     "linkedin and combined 08-21 00:27, local-combined-extracted 01:32; "
     "national-indeed 01:35, companies 01:58, linkedin 06:16, "
     "national-combined-extracted 06:41"),

    (at("2026-08-21 17:00"), at("2026-08-21 17:11"),
     "jobspy evening Halifax pass",
     "nightly-history.tsv start 17:00; "
     "local-indeed-screened-extracted.csv and shortlist mtime 17:11"),
]

QUIET_SOURCE = (
    "no run recorded in nightly-history.tsv covering this window, and no "
    "job-search-data artefact was written during it"
)


def classify(start: datetime, end: datetime) -> tuple[int, str, str] | None:
    """Label one window, or return None to leave it out of the file."""
    if end > CONTAMINATED_FROM:
        return None

    buffer = timedelta(minutes=BUFFER_MINUTES)
    for run_start, run_end, event, source in RUNS:
        # Overlap, not containment: a five minute window straddling the end of a
        # run is genuinely part run, and calling it quiet teaches the opposite of
        # what happened.
        if start < run_end and end > run_start:
            return 1, event, source
    for run_start, run_end, _, _ in RUNS:
        if start < run_end + buffer and end > run_start - buffer:
            return None
    return 0, "", QUIET_SOURCE


def main() -> None:
    windows = load_windows()
    rows = []
    for w in windows:
        start, end = parse_ts(w.start), parse_ts(w.end)
        verdict = classify(start, end)
        if verdict is None:
            continue
        label, event, source = verdict
        rows.append({
            "window_start": w.start, "window_end": w.end, "host": w.host,
            "label": str(label), "event": event, "source": source,
            "note": "seeded by tools/seed_labels.py",
        })
    write_labels(rows)

    positives = sum(1 for r in rows if r["label"] == "1")
    print(f"{len(windows)} windows, {len(rows)} labelled, "
          f"{positives} positive, {len(rows) - positives} negative, "
          f"{len(windows) - len(rows)} deliberately left unlabelled")

    if rows:
        first, last = rows[0]["window_start"], rows[-1]["window_start"]
        print(f"labelled range {first} to {last}")
    covered = {e for _, _, e, _ in ((r[0], r[1], r[2], r[3]) for r in RUNS)}
    print(f"{len(RUNS)} ledger runs described, {len(covered)} distinct events")


if __name__ == "__main__":
    main()
