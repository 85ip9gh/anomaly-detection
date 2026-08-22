"""Export a reproducible slice of Sentinel telemetry to a compact NDJSON file.

Why this exists rather than reading the archive from the training loop: the PRD
requires training to be reproducible, and Sentinel's archive lives in HDFS behind
a Docker stack on a workstation that is not always on. Exporting once pins the
dataset to a file, so a rerun months later trains on the same rows.

Two sources, in preference order:

  hdfs    the real archive at /sentinel/raw/system/dt=<date>/, the PRD's intent
  spool   the collector's local NDJSON spool, which holds readings that failed
          delivery and replayed later. Same schema, same collector, same host.

The spool is the fallback and it is not a lesser dataset in content, only in
coverage: it holds whatever accumulated while the broker was unreachable.
Whichever source ran is recorded in the manifest beside the slice, because a
metric is only reproducible if the dataset behind it is identified.

Usage:
    python tools/export_slice.py --source spool --kind system
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# Only the fields the feature extractor reads. Dropping the rest takes the
# system slice from 5.4 MB to well under one, which is the difference between a
# dataset that belongs in Git and one that does not.
def compact(reading: dict) -> dict | None:
    d = reading.get("data") or {}
    cpu = d.get("cpu") or {}
    mem = d.get("memory") or {}
    swap = d.get("swap") or {}
    net = d.get("network") or {}
    disks = d.get("disks") or []
    ts = reading.get("ts")
    if not ts or not isinstance(cpu.get("percent"), (int, float)):
        return None
    return {
        "ts": ts,
        "host": reading.get("host"),
        "cpu_percent": cpu.get("percent"),
        "memory_percent": mem.get("percent"),
        "memory_available_bytes": mem.get("available_bytes"),
        "swap_percent": swap.get("percent"),
        "process_count": d.get("process_count"),
        "uptime_seconds": d.get("uptime_seconds"),
        "disk_percent": max((x.get("percent") or 0) for x in disks) if disks else None,
        "net_bytes_recv": net.get("bytes_recv"),
        "net_bytes_sent": net.get("bytes_sent"),
        "net_dropin": net.get("dropin"),
        "net_errin": net.get("errin"),
    }


def read_spool(kind: str) -> list[dict]:
    path = Path.home() / "Projects" / "sentinel" / "var" / "spool" / f"sentinel.{kind}.ndjson"
    if not path.exists():
        raise SystemExit(f"no spool at {path}")
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_hdfs(kind: str) -> list[dict]:
    from hdfs import InsecureClient  # optional dependency, only for this path

    client = InsecureClient("http://127.0.0.1:9870", user="pesanth")
    root = f"/sentinel/raw/{kind}"
    out = []
    for part in client.list(root):
        for name in client.list(f"{root}/{part}"):
            with client.read(f"{root}/{part}/{name}", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("spool", "hdfs"), default="spool")
    ap.add_argument("--kind", default="system")
    ap.add_argument("--host", default=None, help="keep one host only")
    args = ap.parse_args()

    raw = read_spool(args.kind) if args.source == "spool" else read_hdfs(args.kind)
    rows = [c for c in (compact(r) for r in raw) if c]
    if args.host:
        rows = [r for r in rows if r["host"] == args.host]
    rows.sort(key=lambda r: r["ts"])
    if not rows:
        raise SystemExit("no rows after filtering")

    DATA.mkdir(exist_ok=True)
    out = DATA / f"{args.kind}-slice.ndjson"
    body = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in rows)
    out.write_text(body, encoding="utf-8")

    manifest = {
        "kind": args.kind,
        "source": args.source,
        "rows": len(rows),
        "hosts": sorted({r["host"] for r in rows}),
        "first_ts": rows[0]["ts"],
        "last_ts": rows[-1]["ts"],
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA / f"{args.kind}-slice.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
