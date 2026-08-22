# Two stages, because the model is built from the repository rather than shipped
# in it. `artifacts/` is gitignored: a fitted estimator is a build output, and a
# pickle in Git is a thing that gets loaded without anyone reading the diff.
# Building it here means the image's model provably comes from the pinned slice
# and the committed code, and `tools/train_model.py` refuses to stamp metrics on
# it that did not come from a held-out run.

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# requirements.txt is pinned, and pinned in that file rather than a separate
# lock. Trivy's pip analyzer matches on the filename, so a `requirements.lock`
# beside a range-based requirements.txt is silently not scanned and the gate
# reports "no findings" because it found nothing to look at. requirements.in is
# the readable spec it compiles from.
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY anomaly/ ./anomaly/
COPY tools/ ./tools/
COPY data/ ./data/
COPY labels/ ./labels/
COPY docs/ ./docs/

# PYTHONPATH rather than an install, because the training step imports the same
# package the runtime stage will and there is no reason for those to differ.
RUN PYTHONPATH=/install/lib/python3.12/site-packages:/build \
    python tools/train_model.py


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ANOMALY_ARTIFACT=/app/artifacts/detector.joblib

WORKDIR /app

# Patch the base ahead of its own rebuild cadence, then create the user, in one
# layer so the apt lists never reach the image.
#
# This is not decoration: the first image scan blocked on 20 HIGH findings in
# the util-linux family (CVE-2026-53612 through 53615), all of them fixed in
# 2.41.5-0+deb13u1 and all of them present only because python:3.12-slim had not
# been rebuilt since the advisory. Waiting for upstream is a choice to ship a
# known-vulnerable image for however long that takes.
#
# System user, no home, no shell login, and it owns nothing it does not need to.
RUN apt-get update \
 && apt-get upgrade --yes --no-install-recommends \
 && adduser --system --group --no-create-home anomaly \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY anomaly/ ./anomaly/
COPY --from=builder /build/artifacts/ ./artifacts/

USER anomaly

EXPOSE 8090

# The health check asks whether the model is loaded, not whether the port is
# open. This service starts and answers happily with no artifact, by design, so
# a check that only proved the process was alive would report healthy for a
# container that cannot score anything.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8090/healthz',timeout=4)); \
sys.exit(0 if r.get('model_loaded') else 1)"

# 0.0.0.0 is the container's own namespace, not the host's LAN. What decides
# exposure is the port publish or the Kubernetes Service in front of it, and
# binding loopback here would make the published port unreachable rather than
# make anything safer.
CMD ["uvicorn", "anomaly.service:app", "--host", "0.0.0.0", "--port", "8090"]
