"""Run the baseline and the model on the same split and print both, side by side.

Two kinds of number come out of this, and they answer different questions.

  Precision, recall and F1 at each detector's own threshold. This is what the
  service would actually do, and it is the number the PRD asks for. It is also
  unkind to any detector whose threshold is set by an assumption rather than by
  the data, which is exactly the case for an isolation forest given a
  contamination rate up front.

  ROC AUC and average precision, which use the score rather than the decision.
  These say whether a detector ranks anomalous windows above normal ones at all,
  independent of where the line is drawn. A detector can rank perfectly and
  still score a terrible F1 if its threshold is in the wrong place, and knowing
  which of those two problems you have is the difference between retuning and
  starting over.

Usage:
    python tools/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from anomaly.baseline import RollingSigmaBaseline  # noqa: E402
from anomaly.dataset import labelled, load_labels, load_windows, split_by_time  # noqa: E402
from anomaly.evaluate import score, usable  # noqa: E402
from anomaly.model import Detector  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def ranking(truth: list[int], scores: list[float]) -> dict:
    if len(set(truth)) < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
    }


def main() -> None:
    windows = load_windows()
    labels = load_labels()
    train, test = split_by_time(windows, train_fraction=0.7)

    diagnostic = "--diagnostic" in sys.argv
    pairs = labelled(test, labels)
    ok, why = usable(pairs)
    print(f"windows {len(windows)}  train {len(train)}  test {len(test)}")
    print(f"labelled in test: {why}")
    if not ok and not diagnostic:
        print("\nNot evaluating. The harness runs; the labels do not support a number.")
        print("Rerun with --diagnostic for in-sample detection quality, which is")
        print("a sanity check on the code and must never be quoted as a result.")
        return
    if diagnostic:
        pairs = labelled(sorted(windows, key=lambda w: w.start), labels)
        print(f"\nDIAGNOSTIC over all {len(pairs)} labelled windows, NOT held out.")
        print("The detector was fitted on 70% of these. It measures whether the")
        print("code works, not whether the model generalises. Do not quote it.")

    test_windows = [w for w, _ in pairs]
    truth = [y for _, y in pairs]

    baseline = RollingSigmaBaseline()
    # scored over the full ordered series so the trailing history is real history
    all_ordered = sorted(windows, key=lambda w: w.start)
    b_scores_all = dict(zip([w.start for w in all_ordered], baseline.score(all_ordered)))
    b_preds_all = dict(zip([w.start for w in all_ordered], baseline.predict(all_ordered)))
    b_scores = [b_scores_all[w.start] for w in test_windows]
    b_preds = [b_preds_all[w.start] for w in test_windows]

    detector = Detector().fit(train)
    m_scores = detector.score(test_windows)
    m_preds = detector.predict(test_windows)

    b = score(truth, b_preds)
    m = score(truth, m_preds)
    print()
    print(b.line("rolling 3-sigma"))
    print(m.line("isolation forest"))

    b_rank, m_rank = ranking(truth, b_scores), ranking(truth, m_scores)
    print()
    for name, r in (("rolling 3-sigma", b_rank), ("isolation forest", m_rank)):
        if r["roc_auc"] is None:
            print(f"{name:24} ranking undefined, one class only")
        else:
            print(f"{name:24} ROC AUC={r['roc_auc']:.3f}  AP={r['average_precision']:.3f}")

    positives = sum(truth)
    results = {
        "mode": "diagnostic-in-sample" if diagnostic else "held-out",
        "windows": len(windows), "train": len(train), "test": len(test),
        "labelled_test": len(pairs), "positive_test": positives,
        "positive_rate_test": positives / len(pairs),
        "baseline": {**b.as_dict(), **b_rank},
        "model": {**m.as_dict(), **m_rank},
        "top_features": [[n, round(v, 3)] for n, v in detector.feature_ranking(test_windows)[:6]],
    }
    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {REPO / 'docs' / 'results.json'}")


if __name__ == "__main__":
    main()
