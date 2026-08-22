"""Service tests that need no trained artifact, no network and no container.

The artifact is built in a temporary directory from a synthetic series, so the
suite still runs on a clean checkout where `artifacts/` does not exist. That
matters more than it sounds: `artifacts/` is gitignored, so a test that needed
the real one would pass only on the machine that trained it.
"""
from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from anomaly import artifact as artifact_module
from anomaly.dataset import load_rows
from anomaly.features import parse_ts, to_windows
from anomaly.model import Detector

BASE = datetime(2026, 8, 19, tzinfo=timezone.utc)


def reading(i: int, cpu: float, host: str = "cubebox") -> dict:
    ts = BASE + timedelta(seconds=10 * i)
    return {
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "host": host,
        "cpu_percent": cpu,
        "memory_percent": 40.0 + (i % 5),
        "memory_available_bytes": 2.0e10,
        "swap_percent": 0.0,
        "process_count": 300 + (i % 7),
        "uptime_seconds": 1000.0 + 10 * i,
        "disk_percent": 46.0,
        "net_bytes_recv": 1_000_000 + 50_000 * i,
        "net_bytes_sent": 500_000 + 10_000 * i,
        "net_dropin": 0,
        "net_errin": 0,
    }


def first_contiguous(rows: list[dict], size: int = 30, max_gap: float = 60.0) -> list[dict]:
    """The first `size` readings with no gap in them.

    The head of the slice is not contiguous: the collector's first readings
    after a cold start are minutes apart, so `rows[:30]` spans a 325 second hole
    and `to_windows` correctly refuses it. Slicing blindly here would have the
    test assert that the windowing rule does not work.
    """
    for start in range(len(rows) - size + 1):
        chunk = rows[start:start + size]
        gaps = [
            (parse_ts(b["ts"]) - parse_ts(a["ts"])).total_seconds()
            for a, b in zip(chunk, chunk[1:])
        ]
        if max(gaps) <= max_gap:
            return chunk
    raise AssertionError("the pinned slice has no contiguous window, which it must")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A service backed by an artifact trained here, not by the repository's."""
    rows = [reading(i, 5.0 + (i % 4)) for i in range(300)]
    detector = Detector().fit(to_windows(rows))
    path = tmp_path / "detector.joblib"
    artifact_module.save(
        detector, train_windows=10, train_first_ts=rows[0]["ts"],
        train_last_ts=rows[-1]["ts"], dataset_sha256="test", metrics={},
        path=path,
    )
    monkeypatch.setenv("ANOMALY_ARTIFACT", str(path))
    service = importlib.reload(importlib.import_module("anomaly.service"))
    return TestClient(service.app)


def test_healthz_reports_the_artifact_it_loaded(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["dataset_sha256"] == "test"


def test_score_returns_the_threshold_alongside_the_decision(client):
    rows = [reading(i, 5.0) for i in range(30)]
    body = client.post("/score", json={"readings": rows}).json()
    assert len(body["scored"]) == 1
    scored = body["scored"][0]
    # The point of returning both: a caller can disagree with the threshold
    # without having to re-derive the score.
    assert scored["anomaly"] == (scored["score"] > scored["threshold"])
    assert scored["readings"] == 30
    assert scored["drivers"]


def test_a_window_short_of_a_full_one_is_refused_not_padded(client):
    rows = [reading(i, 5.0) for i in range(12)]
    response = client.post("/score", json={"readings": rows})
    assert response.status_code == 422
    assert "complete window" in response.json()["detail"]


def test_two_hosts_in_one_request_are_refused(client):
    rows = [reading(i, 5.0) for i in range(15)]
    rows += [reading(i, 5.0, host="g7-server") for i in range(15, 30)]
    response = client.post("/score", json={"readings": rows})
    assert response.status_code == 422
    assert "hosts" in response.json()["detail"]


def test_the_index_page_says_the_recent_list_is_not_a_history(client):
    client.post("/score", json={"readings": [reading(i, 5.0) for i in range(30)]})
    page = client.get("/").text
    assert "not a history" in page
    assert "cubebox" in page


def test_a_missing_artifact_degrades_rather_than_crashes(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_ARTIFACT", str(tmp_path / "absent.joblib"))
    service = importlib.reload(importlib.import_module("anomaly.service"))
    client = TestClient(service.app)
    health = client.get("/healthz").json()
    assert health["status"] == "degraded" and health["model_loaded"] is False
    # 200 on /healthz because the process is up; 503 on /score because it cannot
    # do the one thing it exists for.
    assert client.post("/score", json={"readings": []}).status_code == 503


def test_the_service_scores_real_readings_from_the_pinned_slice(client):
    """The serving path must accept the exact rows the model was trained from.

    A serving path that needs its own input shape is a serving path that will
    drift from the training one, and the published metrics would then describe a
    model nobody deployed.
    """
    rows = first_contiguous(load_rows())
    body = client.post("/score", json={"readings": rows})
    assert body.status_code == 200
    assert len(body.json()["scored"]) == 1
