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

**The spool is a one-shot source.** Starting the broker makes the collector
replay and delete it, so a spool export cannot be reproduced afterwards. That is
why the slice and its manifest are committed rather than regenerated on demand.

Reaching HDFS from Windows needs `--via docker`, which is the default:

  WebHDFS answers a LIST from the NameNode but answers an OPEN with a 307 to the
  DataNode's *container* hostname, `http://datanode:9864/...`. That name does not
  resolve outside the compose network, so the read fails after the listing has
  already succeeded, which makes it look like a permissions problem rather than a
  routing one. It is the same redirect that forced Sentinel's own dashboard to
  run inside the compose network. `--via docker` sidesteps it by running
  `hdfs dfs -getmerge` inside the NameNode container and copying one file out.

Usage:
    python tools/export_slice.py --source hdfs --host cubebox
    python tools/export_slice.py --source spool --kind system
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

NAMENODE_CONTAINER = "sentinel-namenode"
HDFS_ROOT = "/sentinel/raw"


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
        raise SystemExit(
            f"no spool at {path}. The collector deletes it once the broker "
            f"accepts the backlog, so this source is gone after the stack starts. "
            f"Use --source hdfs."
        )
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


def read_hdfs_via_docker(kind: str, container: str = NAMENODE_CONTAINER) -> list[dict]:
    """Merge every part file inside the container, then copy one file out.

    `-getmerge` rather than 5,000 `-cat` calls: each invocation pays a JVM start,
    and the archive is one immutable file per sink batch, so the file count grows
    with uptime and the per-file cost is the whole cost.
    """
    remote = f"/tmp/sentinel-{kind}-merged.ndjson"
    merge = subprocess.run(
        [
            "docker", "exec", container, "bash", "-lc",
            f"hdfs dfs -getmerge '{HDFS_ROOT}/{kind}/dt=*/*.ndjson' {remote} "
            f"&& wc -l {remote}",
        ],
        capture_output=True, text=True,
    )
    if merge.returncode != 0:
        raise SystemExit(
            f"getmerge failed in {container}:\n{merge.stderr.strip()[-2000:]}"
        )
    print(f"merged inside {container}: {merge.stdout.strip().splitlines()[-1]}")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "merged.ndjson"
        copy = subprocess.run(
            ["docker", "cp", f"{container}:{remote}", str(local)],
            capture_output=True, text=True,
        )
        if copy.returncode != 0:
            raise SystemExit(f"docker cp failed:\n{copy.stderr.strip()}")
        out = []
        with local.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    subprocess.run(["docker", "exec", container, "rm", "-f", remote],
                   capture_output=True, text=True)
    return out


def read_hdfs_via_webhdfs(kind: str) -> list[dict]:
    """Only works from inside the compose network. See the module docstring."""
    from hdfs import InsecureClient  # optional dependency, only for this path

    client = InsecureClient("http://127.0.0.1:9870", user="pesanth")
    root = f"{HDFS_ROOT}/{kind}"
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
    ap.add_argument("--source", choices=("spool", "hdfs"), default="hdfs")
    ap.add_argument("--via", choices=("docker", "webhdfs"), default="docker",
                    help="how to reach HDFS; webhdfs only works inside the compose network")
    ap.add_argument("--kind", default="system")
    ap.add_argument("--host", default=None, help="keep one host only")
    ap.add_argument("--plain", action="store_true",
                    help="write .ndjson instead of .ndjson.gz")
    args = ap.parse_args()

    if args.source == "spool":
        raw = read_spool(args.kind)
    elif args.via == "docker":
        raw = read_hdfs_via_docker(args.kind)
    else:
        raw = read_hdfs_via_webhdfs(args.kind)

    rows = [c for c in (compact(r) for r in raw) if c]
    hosts_before = sorted({r["host"] for r in rows})
    if args.host:
        rows = [r for r in rows if r["host"] == args.host]

    # The sink commits Kafka offsets only after its files land, which trades
    # duplicates for never losing a reading. That was the right trade there and
    # it leaves the duplicates here: a duplicated reading inflates a window's
    # count and shrinks its apparent gap, so both coverage features would lie.
    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in rows:
        key = (row["host"], row["ts"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    duplicates = len(rows) - len(deduped)
    rows = deduped

    rows.sort(key=lambda r: r["ts"])
    if not rows:
        raise SystemExit("no rows after filtering")

    DATA.mkdir(exist_ok=True)
    body = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n" for r in rows)
    raw_bytes = body.encode("utf-8")

    # The checksum is over the uncompressed body, so the pin identifies the data
    # rather than the compression settings that happened to be in use.
    digest = hashlib.sha256(raw_bytes).hexdigest()

    if args.plain:
        out = DATA / f"{args.kind}-slice.ndjson"
        out.write_bytes(raw_bytes)
        stale = DATA / f"{args.kind}-slice.ndjson.gz"
    else:
        out = DATA / f"{args.kind}-slice.ndjson.gz"
        # mtime=0 so the same rows produce the same bytes on any run, which is
        # what keeps a re-export from showing up as a diff when nothing changed.
        with gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"),
                           compresslevel=9, mtime=0) as gz:
            gz.write(raw_bytes)
        stale = DATA / f"{args.kind}-slice.ndjson"
    if stale.exists():
        stale.unlink()

    manifest = {
        "kind": args.kind,
        "source": args.source,
        "via": args.via if args.source == "hdfs" else None,
        "file": out.name,
        "rows": len(rows),
        "duplicates_dropped": duplicates,
        "hosts": sorted({r["host"] for r in rows}),
        "hosts_available": hosts_before,
        "host_filter": args.host,
        "first_ts": rows[0]["ts"],
        "last_ts": rows[-1]["ts"],
        "bytes_uncompressed": len(raw_bytes),
        "bytes_on_disk": out.stat().st_size,
        "sha256": digest,
        "sha256_covers": "uncompressed NDJSON body",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (DATA / f"{args.kind}-slice.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
