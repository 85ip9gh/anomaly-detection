"""The inference service: one scoring endpoint, a health check, and a small view.

What it will not do is as much of the design as what it will.

It does not decide for you. `/score` returns the score, the threshold it was
compared against, and the decision that follows, so a caller can disagree with
the threshold without re-deriving the score. A service that returns a bare
boolean is a service whose calibration cannot be audited from the outside.

It does not compute features a second way. The request is turned into a
`Window` by the same `to_windows` the training set went through. A serving path
with its own feature code is the most common way a model that evaluated well
behaves differently in production, and the metrics then describe a model that
was never deployed.

It does not claim a history it has. The view holds the last few scored windows
in memory and says so; there is no database, and a restart empties it. Calling
that a history would be the first untrue sentence in the project.

Run it:
    python tools/train_model.py
    uvicorn anomaly.service:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import artifact as artifact_module
from .features import CADENCE_SECONDS, FEATURE_NAMES, WINDOW_READINGS, to_windows

RECENT_LIMIT = int(os.environ.get("ANOMALY_RECENT_LIMIT", "50"))
ARTIFACT_PATH = Path(os.environ.get("ANOMALY_ARTIFACT", str(artifact_module.DEFAULT_PATH)))


class Reading(BaseModel):
    """One system reading, in the shape `tools/export_slice.py` writes.

    Every numeric field is optional because the collector's own payload is
    optional in places (a host with no swap, an interface with no error
    counters), and the feature extractor already treats a missing value as
    absent rather than as zero. Requiring them here would reject readings the
    training set contained.
    """

    ts: str
    host: str = "unknown"
    cpu_percent: float | None = None
    memory_percent: float | None = None
    memory_available_bytes: float | None = None
    swap_percent: float | None = None
    process_count: float | None = None
    uptime_seconds: float | None = None
    disk_percent: float | None = None
    net_bytes_recv: float | None = None
    net_bytes_sent: float | None = None
    net_dropin: float | None = None
    net_errin: float | None = None


class ScoreRequest(BaseModel):
    readings: list[Reading] = Field(
        ...,
        description=(
            f"One window of consecutive readings from a single host. "
            f"{WINDOW_READINGS} at a {CADENCE_SECONDS:.0f} second cadence is one "
            f"window; more than that is scored as several."
        ),
    )


class WindowScore(BaseModel):
    window_start: str
    window_end: str
    host: str
    readings: int
    score: float = Field(..., description="Higher is more anomalous.")
    threshold: float
    anomaly: bool
    # The three features furthest from their training mean, in training standard
    # deviations. Not an explanation of the forest's decision and not claimed to
    # be one: it answers what was unusual about this window, which is the
    # question somebody looking at an alert actually asks.
    drivers: list[dict[str, float | str]]


class ScoreResponse(BaseModel):
    scored: list[WindowScore]
    model: dict[str, Any]


class Health(BaseModel):
    status: str
    model_loaded: bool
    trained_at: str | None = None
    dataset_sha256: str | None = None
    detail: str | None = None


app = FastAPI(
    title="Anomaly detection over machine telemetry",
    description=__doc__,
    version="0.2.0",
)

_recent: Deque[WindowScore] = deque(maxlen=RECENT_LIMIT)
_artifact = None
_load_error: str | None = None


def get_artifact():
    """Load once, lazily, and remember the failure rather than retrying forever.

    Lazily because an import-time load makes the module unimportable without a
    trained artifact, which would take the test suite and every tool down with
    it. Remembered because a missing artifact does not fix itself, and retrying
    on every request turns one clear error into a slow one.
    """
    global _artifact, _load_error
    if _artifact is None and _load_error is None:
        try:
            _artifact = artifact_module.load(ARTIFACT_PATH)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            _load_error = f"{type(exc).__name__}: {exc}"
    return _artifact


def _drivers(window, art, top: int = 3) -> list[dict[str, float | str]]:
    import numpy as np

    scaled = art.detector._prepare([window])[0]
    pairs = sorted(zip(art.detector.kept_names(), scaled), key=lambda p: -abs(p[1]))
    return [
        {"feature": name, "train_sigmas": round(float(value), 2)}
        for name, value in pairs[:top]
        if np.isfinite(value)
    ]


@app.get("/healthz", response_model=Health)
def healthz() -> Health:
    art = get_artifact()
    if art is None:
        # 200 with `model_loaded: false`, not 503. The process is up and
        # answering; what is missing is its artifact, and saying so plainly is
        # more useful to whoever has to fix it than a status code that could
        # equally mean the container is dead.
        return Health(status="degraded", model_loaded=False, detail=_load_error)
    return Health(
        status="ok",
        model_loaded=True,
        trained_at=art.trained_at,
        dataset_sha256=art.dataset_sha256,
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    art = get_artifact()
    if art is None:
        raise HTTPException(503, f"no model loaded: {_load_error}")

    rows = [r.model_dump() for r in request.readings]
    hosts = {r["host"] for r in rows}
    if len(hosts) > 1:
        raise HTTPException(
            422,
            f"readings span {len(hosts)} hosts ({', '.join(sorted(hosts))}). "
            "A window mixes two machines' counters into one rate, so score each "
            "host separately.",
        )

    windows = to_windows(rows)
    if not windows:
        raise HTTPException(
            422,
            f"{len(rows)} readings produced no complete window. A window is "
            f"{WINDOW_READINGS} consecutive readings with no gap longer than "
            f"{CADENCE_SECONDS * 6:.0f} seconds.",
        )

    scores = art.detector.score(windows)
    out: list[WindowScore] = []
    for window, value in zip(windows, scores):
        result = WindowScore(
            window_start=window.start,
            window_end=window.end,
            host=window.host,
            readings=window.readings,
            score=round(float(value), 6),
            threshold=round(art.threshold, 6),
            anomaly=bool(value > art.threshold),
            drivers=_drivers(window, art),
        )
        out.append(result)
        _recent.append(result)

    return ScoreResponse(
        scored=out,
        model={
            "trained_at": art.trained_at,
            "train_windows": art.train_windows,
            "train_first_ts": art.train_first_ts,
            "train_last_ts": art.train_last_ts,
            "dataset_sha256": art.dataset_sha256,
            "threshold": round(art.threshold, 6),
            "features_used": len(art.kept_features),
            "features_dropped": list(art.dropped_features),
            "held_out_metrics": art.metrics,
        },
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    art = get_artifact()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if art is None:
        body = (
            "<p class='warn'>No model artifact is loaded. Run "
            "<code>python tools/train_model.py</code>.</p>"
            f"<pre>{_load_error}</pre>"
        )
        header = ""
    else:
        m = art.metrics.get("model") or {}
        b = art.metrics.get("baseline") or {}
        trivial = art.metrics.get("trivial_always_flag") or {}
        rate = art.metrics.get("positive_rate_test")
        header = f"""
        <table>
          <tr><th>detector</th><th>precision</th><th>recall</th><th>F1</th>
              <th>ROC AUC</th><th>average precision</th></tr>
          <tr><td>rolling mean + 3 sigma</td>{_cells(b)}</tr>
          <tr class='model'><td>isolation forest</td>{_cells(m)}</tr>
          <tr><td>always flag (trivial)</td>{_cells(trivial)}</tr>
        </table>
        <p class='note'>Held out by time, split at a gap in the archive.
        {art.metrics.get('labelled_test', '?')} labelled windows,
        positive rate {rate if rate is None else round(rate, 3)}. The trivial row
        is there because F1 on a mostly-anomalous period rewards flagging
        everything, and a table without it invites you to credit the model for
        the prevalence.</p>
        """
        body = _recent_table()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Anomaly detection over machine telemetry</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 system-ui, sans-serif; margin: 2.5rem auto; max-width: 60rem;
         padding: 0 1.25rem; }}
  h1 {{ font-size: 1.35rem; margin-bottom: .2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #8884; }}
  th {{ font-weight: 600; font-size: .82rem; text-transform: uppercase; letter-spacing: .04em; }}
  tr.model td {{ font-weight: 600; }}
  td.flag {{ color: #b3261e; font-weight: 600; }}
  .note, .sub {{ color: #7a7a7a; font-size: .87rem; }}
  .warn {{ color: #b3261e; }}
  code, pre {{ font-family: ui-monospace, monospace; font-size: .85rem; }}
</style></head><body>
<h1>Anomaly detection over machine telemetry</h1>
<p class="sub">An isolation forest over five minute windows of
<a href="https://sentinel.pesanth.com">Sentinel</a>'s system telemetry, scored
against a rolling three-sigma rule it has to beat. Working name.</p>
{header}
{body}
<p class="note">Generated {now}. Recent scores are held in this process's memory
and are lost on restart; there is no database and this is not a history.</p>
</body></html>"""


def _cells(metrics: dict) -> str:
    def cell(key: str) -> str:
        value = metrics.get(key)
        return f"<td>{'-' if value is None else round(float(value), 3)}</td>"
    return "".join(cell(k) for k in
                   ("precision", "recall", "f1", "roc_auc", "average_precision"))


def _recent_table() -> str:
    if not _recent:
        return ("<p class='note'>Nothing scored yet. POST a window of readings to "
                "<code>/score</code>.</p>")
    rows = "".join(
        f"<tr><td>{r.window_start}</td><td>{r.host}</td>"
        f"<td>{r.score:.4f}</td><td>{r.threshold:.4f}</td>"
        f"<td class='{'flag' if r.anomaly else ''}'>{'anomaly' if r.anomaly else 'normal'}</td>"
        f"<td class='note'>{', '.join(str(d['feature']) for d in r.drivers)}</td></tr>"
        for r in reversed(_recent)
    )
    return ("<h2 style='font-size:1rem'>Recently scored</h2><table>"
            "<tr><th>window start</th><th>host</th><th>score</th><th>threshold</th>"
            "<th>decision</th><th>furthest features</th></tr>"
            f"{rows}</table>")


__all__ = ["app", "FEATURE_NAMES"]
