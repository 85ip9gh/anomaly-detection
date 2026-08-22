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

Two reference rows sit alongside them: a detector that flags everything and one
that flags nothing. On a period that is mostly anomalous, flagging everything
scores a high F1 while knowing nothing at all, so a table without that row
invites the reader to credit the model for the prevalence.

Usage:
    python tools/report.py
    python tools/report.py --diagnostic            in-sample, never quotable
    python tools/report.py --without coverage      sensitivity, never the headline
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
    exclude: tuple[str, ...] = ()
    if "--without" in sys.argv:
        exclude = tuple(sys.argv[sys.argv.index("--without") + 1].split(","))
        print(f"SENSITIVITY CHECK, not the headline result: fitting without "
              f"{', '.join(exclude)}.\nThis exists to show whether a result "
              f"survives losing a feature, and choosing\nwhich feature to lose "
              f"after seeing the result is how a model gets tuned on its\nown "
              f"test set. Read it next to the default run, never instead of it.\n")
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

    detector = Detector(exclude=exclude).fit(train)
    m_scores = detector.score(test_windows)
    m_preds = detector.predict(test_windows)

    b = score(truth, b_preds)
    m = score(truth, m_preds)

    # Two detectors that do no work at all. They are here because F1 on an
    # unbalanced set is very easy to misread: on a period that is mostly
    # anomalous, flagging everything scores a high F1 while knowing nothing, and
    # a comparison that omits it invites the reader to credit the model for the
    # prevalence. Whether the model beats these is a separate question from
    # whether it beats the baseline, and both belong on the page.
    always = score(truth, [1] * len(truth))
    never = score(truth, [0] * len(truth))

    prevalence = sum(truth) / len(truth)
    print()
    print(b.line("rolling 3-sigma"))
    print(m.line("isolation forest"))
    print(always.line("always flag (trivial)"))
    print(never.line("never flag (trivial)"))
    print(f"\npositive rate in the evaluated set: {prevalence:.3f}. That is the floor "
          f"for average precision\nand the precision an always-flag rule gets for free.")

    b_rank, m_rank = ranking(truth, b_scores), ranking(truth, m_scores)
    print()
    for name, r in (("rolling 3-sigma", b_rank), ("isolation forest", m_rank)):
        if r["roc_auc"] is None:
            print(f"{name:24} ranking undefined, one class only")
        else:
            print(f"{name:24} ROC AUC={r['roc_auc']:.3f}  AP={r['average_precision']:.3f}")

    shift = detector.shift_report(test_windows)
    print("\ncovariate shift, test mean in training standard deviations:")
    for name, value in shift[:5]:
        print(f"  {name:28} {value:+8.1f}")
    print("  A large offset moves every test score together, so it leaves ROC AUC")
    print("  and AP alone and undermines the thresholded precision, recall and F1.")
    if detector.dropped:
        print(f"\ndropped {len(detector.dropped)} zero-variance features: "
              f"{', '.join(detector.dropped)}")

    positives = sum(truth)
    results = {
        "mode": "diagnostic-in-sample" if diagnostic else "held-out",
        "windows": len(windows), "train": len(train), "test": len(test),
        "labelled_test": len(pairs), "positive_test": positives,
        "positive_rate_test": positives / len(pairs),
        "split": "snapped to a gap in the archive, so train and test hold "
                 "different stretches of machine time",
        "baseline": {**b.as_dict(), **b_rank},
        "model": {**m.as_dict(), **m_rank},
        "trivial_always_flag": always.as_dict(),
        "trivial_never_flag": never.as_dict(),
        "top_features": [[n, round(v, 3)] for n, v in detector.feature_ranking(test_windows)[:6]],
        "covariate_shift_train_sigmas": [[n, round(v, 2)] for n, v in shift[:8]],
        "dropped_features": detector.dropped,
    }
    if exclude:
        print("\nSensitivity run: results.json not overwritten, because the file "
              "is\nthe record of the headline result and this is not it.")
        return
    (REPO / "docs").mkdir(exist_ok=True)
    (REPO / "docs" / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {REPO / 'docs' / 'results.json'}")


if __name__ == "__main__":
    main()
