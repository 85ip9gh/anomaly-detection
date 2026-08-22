"""Score a detector against labelled windows, and report all three numbers.

Precision, recall and F1 together, plus the confusion matrix, because reporting
only the flattering one is the tell that no real evaluation happened. A detector
that flags everything has perfect recall; a detector that flags one window it is
sure about has excellent precision. Neither is useful and both look good if you
choose the metric afterwards.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .features import Window


@dataclass(frozen=True)
class Scores:
    support: int
    positives: int
    flagged: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return asdict(self)

    def line(self, name: str) -> str:
        return (f"{name:24} P={self.precision:.3f} R={self.recall:.3f} "
                f"F1={self.f1:.3f}  (tp={self.true_positive} fp={self.false_positive} "
                f"fn={self.false_negative} tn={self.true_negative})")


def score(truth: list[int], predicted: list[int]) -> Scores:
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted differ in length")
    tp = sum(1 for t, p in zip(truth, predicted) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, predicted) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, predicted) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(truth, predicted) if t == 0 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Scores(
        support=len(truth), positives=tp + fn, flagged=tp + fp,
        true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
        precision=precision, recall=recall, f1=f1,
    )


def usable(pairs: list[tuple[Window, int]], minimum_positives: int = 5) -> tuple[bool, str]:
    """Whether a labelled set can support a metric anybody should quote.

    Exists because the failure this project is most likely to have is not a bad
    model, it is a good-looking number computed over four labelled windows. The
    check is deliberately part of the library rather than a note in the README,
    so a run that cannot be evaluated says so itself.
    """
    positives = sum(1 for _, y in pairs if y == 1)
    if not pairs:
        return False, "no labelled windows"
    if positives < minimum_positives:
        return False, (f"{positives} positive windows, fewer than the {minimum_positives} "
                       "needed before precision and recall mean anything")
    return True, f"{len(pairs)} labelled windows, {positives} positive"
