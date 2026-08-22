"""The obvious alternative: a rolling mean plus three standard deviations.

This exists to be beaten, or not beaten. Every claim the model makes is only
meaningful against a number that a person could have produced in an afternoon
without any machine learning at all, and this is that number. It is written
first and deliberately kept simple.

The rule: for each monitored signal, take the mean and standard deviation of the
previous `history` windows, and flag the current window if it sits more than
`sigma` standard deviations above that mean. A window is anomalous if any single
signal fires, which is what somebody writing threshold alerts would do.

Only a handful of signals are monitored rather than all 26 features, because a
three-sigma rule over 26 correlated features fires on almost everything. Keeping
the set small is what makes this a fair opponent instead of a straw man.
"""
from __future__ import annotations

from statistics import fmean, pstdev

from .features import Window

MONITORED = (
    "cpu_percent_mean",
    "memory_percent_mean",
    "process_count_mean",
    "net_bytes_recv_rate_mean",
    "net_bytes_sent_rate_mean",
)


class RollingSigmaBaseline:
    def __init__(self, history: int = 12, sigma: float = 3.0,
                 monitored: tuple[str, ...] = MONITORED) -> None:
        self.history = history
        self.sigma = sigma
        self.monitored = monitored

    def score(self, windows: list[Window]) -> list[float]:
        """Largest z-score across the monitored signals, per window.

        Windows without enough history to judge score 0.0, which counts them as
        normal. That is the honest treatment: the rule genuinely cannot say
        anything about the first few windows, and pretending otherwise by
        seeding from the whole series would leak the future into the past.
        """
        out: list[float] = []
        for i, w in enumerate(windows):
            if i < self.history:
                out.append(0.0)
                continue
            past = windows[i - self.history:i]
            worst = 0.0
            for name in self.monitored:
                values = [p.features[name] for p in past]
                mean = fmean(values)
                sd = pstdev(values)
                if sd <= 0:
                    continue
                z = (w.features[name] - mean) / sd
                worst = max(worst, z)
            out.append(worst)
        return out

    def predict(self, windows: list[Window]) -> list[int]:
        return [1 if s > self.sigma else 0 for s in self.score(windows)]
