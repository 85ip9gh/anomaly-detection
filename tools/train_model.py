"""Fit the detector the service will serve, and record what it was fitted on.

Same split as `tools/report.py`, deliberately. Training the shipped model on
everything including the held-out period would make the service better and the
published numbers meaningless, because the metrics stamped into the artifact
would then describe a different model from the one answering requests. The
service is allowed to be slightly worse than it could be; it is not allowed to
carry numbers it did not earn.

Usage:
    python tools/train_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly.artifact import DEFAULT_PATH, save  # noqa: E402
from anomaly.dataset import DATA, load_windows, split_by_time  # noqa: E402
from anomaly.model import Detector  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    windows = load_windows()
    train, test = split_by_time(windows)
    detector = Detector().fit(train)

    manifest = json.loads((DATA / "system-slice.manifest.json").read_text(encoding="utf-8"))

    results_path = REPO / "docs" / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if results.get("mode") != "held-out":
            raise SystemExit(
                "docs/results.json is not a held-out run, so it must not be "
                "stamped into an artifact. Run tools/report.py first."
            )
        metrics = {
            "held_out": True,
            "baseline": results["baseline"],
            "model": results["model"],
            "trivial_always_flag": results["trivial_always_flag"],
            "positive_rate_test": results["positive_rate_test"],
            "labelled_test": results["labelled_test"],
        }
    else:
        # An artifact with no metrics is honest. An artifact with invented ones
        # is what this branch exists to prevent.
        metrics = {"held_out": False, "note": "no report run; metrics unknown"}

    artifact = save(
        detector,
        train_windows=len(train),
        train_first_ts=train[0].start,
        train_last_ts=train[-1].end,
        dataset_sha256=manifest["sha256"],
        metrics=metrics,
    )
    print(f"fitted on {len(train)} windows, {len(test)} held out")
    print(f"threshold {artifact.threshold:.6f}, "
          f"{len(artifact.kept_features)} features kept, "
          f"{len(artifact.dropped_features)} dropped")
    print(f"wrote {DEFAULT_PATH} and {DEFAULT_PATH.with_suffix('.json')}")


if __name__ == "__main__":
    main()
