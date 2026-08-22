# Anomaly Detection over machine telemetry

Working name. An anomaly detector over the telemetry that [Sentinel](https://sentinel.pesanth.com)
already collects: a pinned dataset, labels that came from somewhere other than
the signal, a split that lands on a gap in the archive, a rolling three-sigma
baseline to beat, and two trivial detectors to keep the comparison honest.

## The result, stated plainly

**The isolation forest beats the rolling three-sigma rule on held-out data, and
neither beats a rule that flags every window.** Both halves of that sentence are
the result.

| detector | precision | recall | F1 | ROC AUC | average precision |
|---|---|---|---|---|---|
| rolling mean + 3 sigma | 0.692 | 0.208 | 0.320 | 0.586 | 0.666 |
| **isolation forest** | **0.719** | **0.531** | **0.611** | **0.622** | **0.725** |
| always flag (trivial) | 0.616 | 1.000 | 0.762 | 0.500* | 0.616* |
| never flag (trivial) | 0.000 | 0.000 | 0.000 | 0.500* | 0.616* |

\* definitional, not measured. A detector whose score never varies has ROC AUC
exactly 0.5 and average precision exactly the positive rate, so `tools/report.py`
does not compute those two cells.

211 labelled windows held out by time, 130 positive and 81 negative, from an
archive of 591 windows over four disconnected stretches. The detector was fitted
on 366 windows ending 2026-08-16 and never saw the test period, which begins
2026-08-19. Reproduce with `python tools/report.py`; `docs/results.json` is the
machine-readable form and CI fails if it stops reproducing.

**The trivial rows are there because F1 on a mostly-anomalous period is easy to
misread.** The held-out set is 61.6% positive, so flagging everything scores
F1 0.762 while knowing nothing at all. A table without that row invites you to
credit the model for the prevalence. The honest reading is the ranking metrics:
ROC AUC 0.622 against the baseline's 0.586 and a coin flip's 0.500, and average
precision 0.725 against a floor of 0.616. The model carries real signal. It is
not a large amount of signal.

**Where the model wins is worth naming, because it explains the shape of the
table.** The baseline's recall is 0.208 not because three sigma is a bad idea but
because the largest labelled event runs for 13.7 hours: after the first hour the
rolling mean has absorbed the anomaly and the rule goes quiet. A sustained shift
is exactly the regime a trailing-window rule is structurally blind to, and it is
most of what separates 0.531 from 0.208.

### How far to trust the thresholded numbers

Less than the ranking ones, and the report says so itself:

```
covariate shift, test mean in training standard deviations:
  coverage                          -40.1
  process_count_mean                 -4.3
  process_count_max                  -4.1
  net_dropin_rate_mean               +3.9
  swap_percent_mean                  -2.8
```

Train and test are different weeks of the same machine, so a feature can sit at
a different level in both without any window being anomalous. `coverage` is the
extreme case: it averages 0.96 in the test period against 0.999 in training, and
the difference has nothing to do with the label, since both classes average 0.96.
A uniform offset like that moves every test score together, which leaves ROC AUC
and average precision untouched (they only read the ordering) and quietly
undermines precision, recall and F1, which read a fixed threshold.

`python tools/report.py --without coverage` is the sensitivity check. Removing
it *raises* ROC AUC to 0.645 and AP to 0.732, so the result does not depend on
the artefact. That run is labelled a sensitivity check everywhere it appears and
never overwrites `docs/results.json`, because choosing which feature to drop
after seeing a result is how a model gets tuned on its own test set.

### What changed from the first attempt

The first version of this repository could not produce a held-out number at all.
A time split left 3 positive windows in the test set and `anomaly/evaluate.py`
refused to score them, which was the correct behaviour. The in-sample diagnostic
it ran instead put the isolation forest at ROC AUC 0.246, worse than chance,
because the dataset was 19.6 hours from the collector's local spool and the one
long labelled event occupied 59% of it: an unsupervised detector fitted on that
period correctly concluded the job running was the machine's normal state.

Four things fixed it, and none of them was a better model.

1. **The real archive.** 17,813 readings across six days from HDFS instead of
   19.6 hours from the spool.
2. **All eight runs in the ledger**, not the two that happened to fall inside
   the spool's coverage.
3. **A split that lands on a gap**, so training and testing hold stretches of
   machine time separated by two days with the machine switched off.
4. **A baseline that resets its history across those gaps**, so it is not losing
   to a handicap.

## The data

`data/system-slice.ndjson.gz`, 17,813 readings from `cubebox` at a 10 second
cadence, 2026-08-15 to 2026-08-22, pinned by sha256 in
`data/system-slice.manifest.json`. The checksum covers the uncompressed body, so
it identifies the data rather than the compression settings.

Six days of span is not six days of readings. The collector publishes only while
its host is awake, so the archive is four disconnected stretches totalling about
49 hours. That is a property of a workstation, and it is why the split snaps to a
gap instead of to a percentage.

`g7-server` is in the same archive and deliberately not in this slice. The run
ledger that produces the labels describes jobs on cubebox only, so a g7 window
could never be labelled from it, and training on a second machine's profile
would teach the model that g7's normal is cubebox's normal.

**Getting it out of HDFS needs `--via docker`.** WebHDFS answers a LIST from the
NameNode and answers an OPEN with a 307 to the DataNode's container hostname,
which does not resolve outside the compose network: the listing succeeds and the
read fails, so it reads as a permissions problem rather than a routing one.
`tools/export_slice.py` runs `hdfs dfs -getmerge` inside the NameNode container
and copies one file out.

Readings become 5 minute windows of 26 features. Windows do not overlap, because
overlapping windows share readings and a later time split then leaks the tail of
training into the head of testing.

## The labels, and why they are not from the signal

`labels/labels.csv`, 565 of 591 windows: 156 positive, 409 negative, 26 left
deliberately unlabelled.

The label means one narrow thing:

> **1** a jobspy scan pipeline was running on cubebox
> **0** no jobspy scan pipeline was running on cubebox

Every interval is generated by `tools/seed_labels.py` from evidence outside the
telemetry entirely:

- `nightly-history.tsv` from the job search pipeline, which records when each
  scan started
- the modification times of the artefacts each run produced, which bound its
  scraping and model-extraction stages

Neither knows anything about CPU or memory. **If the labels came from looking at
the telemetry and calling the spiky parts anomalous, the model would be scored on
its agreement with a spike detector**, and the F1 would be arithmetic performed
on a tautology.

The narrowness has a cost and it is stated rather than hidden: a 0 window may
still have had a Docker build, an agent session or a video call on it. That is
unlabelled load sitting inside the negative class, and it can only make the task
harder, never easier. A detector that scores well here has separated the scan
pipeline from everything else the machine does, which is the stronger claim.

Two things are left out of the file rather than guessed at. A 30 minute buffer
either side of every run, because start times come from a scheduler and end
times from file modification times and a window that straddles that boundary is
genuinely part-run. And every window on 2026-08-22, the night this exporter was
written: that date's telemetry includes a Docker Desktop restart and a full Kafka
replay performed *by the session building this dataset*, and labelling it normal
would be recording the measurement as if it were the thing measured.

Two ledger runs have no interval, 08-17 02:00 and 08-18 02:00, because cubebox
published no telemetry on those dates. Their scans ran and their shortlists
exist; the collector was not up. That is a fact about the archive, not about the
runs.

## The service

```bash
python tools/train_model.py
uvicorn anomaly.service:app --host 127.0.0.1 --port 8090
```

`POST /score` takes a window of readings and returns the score, the threshold it
was compared against, the decision that follows, and the three features furthest
from their training mean.

```json
{
  "window_start": "2026-08-16T04:59:40.864660Z",
  "host": "cubebox",
  "readings": 30,
  "score": 0.5517,
  "threshold": 0.560933,
  "anomaly": false,
  "drivers": [
    {"feature": "cpu_percent_std", "train_sigmas": 4.73},
    {"feature": "memory_percent_std", "train_sigmas": 3.51},
    {"feature": "cpu_percent_max", "train_sigmas": 2.95}
  ]
}
```

Three decisions in that response.

**It returns the threshold, not just the boolean.** A caller can disagree with
the calibration without re-deriving the score, and a service that returns a bare
verdict is one whose calibration cannot be audited from outside.

**It computes features with the same code the training set went through.** A
serving path with its own feature implementation is the most common way a model
that evaluated well behaves differently in production, and the published metrics
then describe a model nobody deployed.

**It refuses rather than guesses.** A request spanning two hosts is a 422,
because a window mixes their counters into one rate. A partial window is a 422
rather than being padded. With no artifact loaded, `/healthz` answers 200 with
`model_loaded: false` and `/score` answers 503, because the process being up and
the model being loaded are different facts and one status code cannot carry both.

`GET /` is a small page with the held-out table and the last few scored windows.
Those live in the process's memory and are lost on restart; there is no database
and the page says so.

## The container

```bash
docker build -t anomaly-detection:0.2.0 .
docker run -d -p 127.0.0.1:8090:8090 anomaly-detection:0.2.0
```

Two stages. The model is **built** in the builder stage from the pinned slice
rather than committed: `artifacts/` is gitignored, and a CI job refuses a tracked
`.joblib` in case that stops being enough. joblib and pickle both execute
whatever the file tells them to on load, so a model file in Git turns `git pull`
into code execution nobody reviewed.

Runs as uid 100, no home, no login shell. The `HEALTHCHECK` asks whether the
model is loaded rather than whether the port is open, because this service
starts and answers happily with no artifact by design.

## The gates

The same three as [cube-store] and [car-sale], on a third repository, and every
one of them was proved to block rather than asserted to work.

| gate | proof it blocks |
|---|---|
| gitleaks, full history | clean here; the control that caught a credential manual review had passed over |
| trivy fs | blocks on a planted `requests==2.19.1` |
| trivy image | blocked on 54 real findings before any allowlist |
| checkov | fails a Dockerfile with no `USER` and no `HEALTHCHECK` |

Two findings from building them here were not obvious.

**The dependency gate found nothing at all on its first run, and reported that
as a clean scan.** requirements.txt held version ranges, and Trivy resolves a
`==` and skips a `>=`. Worse, its pip analyzer matches on the *filename*, so
moving the pins into a `requirements.lock` beside it would have been silently
unread too. requirements.txt is now the pinned, compiled form and
`requirements.in` is the readable spec it compiles from.

**The image scan blocked on 54 findings, 20 of which had a fix waiting.** The
util-linux family CVE-2026-53612 through 53615 were all fixed in
2.41.5-0+deb13u1 and present only because `python:3.12-slim` had not been rebuilt
since the advisory. An `apt-get upgrade` in the Dockerfile cleared all 20. Taking
a fix always beats recording a reason not to.

The 12 CVEs that remain have no fixed version in Debian 13 and are allowlisted in
`.trivyignore` with a dated reason each. Four are CRITICAL in `perl-base`: perl
arrives with the base image, nothing here ever executes it, and it cannot be
purged without removing dpkg. **A perl-free base is an open item, not something
the allowlist pretends away.**

[cube-store]: https://cubestore.pesanth.com
[car-sale]: https://carsale.pesanth.com

## Layout

```
anomaly/features.py     readings to windows: counter resets, coverage, reboots
anomaly/baseline.py     rolling mean + 3 sigma, with history that resets on gaps
anomaly/model.py        isolation forest, plus the covariate-shift report
anomaly/evaluate.py     precision, recall, F1, and a refusal to score thin labels
anomaly/dataset.py      loading, and the split that snaps to a gap
anomaly/artifact.py     save and load, refusing a feature-set mismatch
anomaly/label_cli.py    the labelling tool, written before the model on purpose
anomaly/service.py      FastAPI: /score, /healthz, and a small view
tools/export_slice.py   pin a dataset from HDFS, or from the spool
tools/seed_labels.py    labels from the external run ledger
tools/train_model.py    fit the shipped model on the reported split
tools/report.py         every detector, side by side
```

```bash
pip install -r requirements-dev.txt
python tools/export_slice.py --source hdfs --host cubebox   # needs the stack up
python tools/seed_labels.py
python tools/report.py
python tools/train_model.py
python -m pytest
```

33 tests, none of which need Kafka, HDFS, a network or a Docker daemon. CI
enforces that: a live-service import in the library or the suite fails the build.

## What is not done

- **Not deployed.** PRD steps 8 and 9, k3s on the g7 server and a public URL
  through the Cloudflare Tunnel, have not been attempted.
- **No resume rows.** `scikit-learn`, `FastAPI` and `Model evaluation` stay where
  they are until the numbers above have survived more than one week of data.
- **`coverage` is a broken feature and should be redesigned.** It was added to
  catch a collector outage, but `to_windows` refuses to span a gap and every
  window holds exactly 30 readings, so a real outage never appears inside one.
  What is left measures the collector's timer drift, which differs by period
  rather than by label.
- **A perl-free base image**, to retire eight allowlist entries including four
  CRITICAL.

## What this does not claim

- Not production ML, and it serves nobody.
- No availability or uptime language: the collector runs on one workstation.
- Not Sentinel's AI agent. That does not exist and no bullet may say it does.
- The numbers above are held out by time on 211 labelled windows from one
  machine over six days. That is enough to say the model beats the baseline
  here. It is not enough to say it would beat it anywhere else.
