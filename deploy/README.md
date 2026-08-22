# Deploying to k3s on g7

PRD steps 8 and 9. This is the first real workload on that cluster, so the
procedure below is what was actually run on 2026-08-22, not a plan.

The service is public at **https://anomaly.pesanth.com**.

## Shape

```
internet -> Cloudflare edge (TLS terminates here)
         -> cloudflared on the host  (systemd, /etc/cloudflared/config.yml)
         -> Traefik  10.43.0.100:80  (ClusterIP, pinned, `web` entrypoint only)
         -> Service  anomaly-detection:80
         -> Pod      :8090
```

Nothing in that chain is reachable from the LAN or the tailnet. Traefik's
ClusterIP is programmed by kube-proxy into the host network namespace, which is
all cloudflared needs and is strictly less exposed than publishing a port.

## There is no registry

The image is built on the node and imported straight into containerd. That is a
deliberate choice for a single-node cluster with no second consumer, and it has
two consequences that are suppressed in `k8s/deployment.yaml` with dated
reasons rather than left silently failing:

- `imagePullPolicy` is `IfNotPresent`, not `Always`. With nothing to pull from,
  `Always` does not make the pod more current, it makes it unschedulable.
- The image is referenced by tag, not by digest. A digest pin defends against a
  mutable tag in a registry someone else can write to; there is no publication
  step here for anyone to race.

Both suppressions name the condition that retires them: a registry.

## Procedure

Everything runs on the node. From a workstation, `ssh g7`.

```bash
# 1. Source
git clone https://github.com/85ip9gh/anomaly-detection.git ~/anomaly-detection
cd ~/anomaly-detection

# 2. Build. The model is trained in the builder stage from the pinned slice,
#    so the image's estimator provably comes from committed data and code.
docker build --tag anomaly-detection:g7 .

# 3. Import into the containerd namespace k3s actually reads. `k3s ctr`
#    defaults to k8s.io; plain `ctr` does not, and an image imported into the
#    wrong namespace is invisible to the kubelet while `docker images` still
#    lists it.
docker save anomaly-detection:g7 | sudo k3s ctr images import --digests=true -

# 4. Apply
sudo k3s kubectl apply -f deploy/k8s/
sudo k3s kubectl -n apps rollout status deploy/anomaly-detection --timeout=180s
```

Redeploying after a code change is steps 2 to 4 again. The Deployment uses
`strategy: Recreate`, so the old pod is gone before the new one starts; with one
replica and a shared tag there is nothing a rolling update could roll back to.

## Tunnel and DNS

Done once, and recorded here because the flags are easy to get wrong.

```bash
# The --config flag goes BEFORE the subcommand. After it, cloudflared silently
# prints help instead of validating, and exits 0 while having checked nothing.
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress validate
sudo cloudflared --config /etc/cloudflared/config.yml tunnel ingress rule https://anomaly.pesanth.com/

sudo systemctl restart cloudflared
cloudflared tunnel route dns <tunnel-id> anomaly.pesanth.com
```

Back up `/etc/cloudflared/config.yml` before editing it, and recheck every other
hostname after the restart. One tunnel carries all six.

Order matters: pod, then tunnel route, then DNS. The reverse publishes a
hostname that resolves to a 404 for however long the gap lasts.

## Probes

Liveness is an HTTP GET on `/healthz`. Readiness is an `exec` that reads the
same endpoint and checks `model_loaded`.

That asymmetry is the point. The service answers `/healthz` with 200 and
`model_loaded: false` when no artifact is present, by design, so an httpGet
readiness probe would mark a container Ready that returns 503 on every `/score`.
A check that passes because it never looked at the thing that matters is the
single most repeated failure in this project's notes.

## NetworkPolicy

Ingress is restricted to the Traefik pod and to the node, egress to DNS only.
k3s runs the kube-router policy controller unless `--disable-network-policy` is
passed, and it is not passed here, so this is enforced. That was checked on the
node rather than assumed from the distribution.

**The node has to be in the ingress rule.** The kubelet's httpGet liveness probe
arrives from the host over the flannel bridge and is not a pod, so a policy
naming only Traefik passes a dry run, applies cleanly, and then kills the
container every 90 seconds with what looks like an application fault.

## Verification, 2026-08-22

Each of these was run, not assumed.

| Check | Result |
|---|---|
| Rollout | `1/1 Running`, 0 restarts |
| `/healthz` through Traefik | 200, `model_loaded: true` |
| `dataset_sha256` served by the live pod | matches `data/system-slice.manifest.json` |
| Unknown Host through Traefik | 404, so no accidental catch-all |
| NetworkPolicy | a busybox pod in `default` cannot open :8090 |
| Checkov, `kubernetes` framework | 92 passed, 0 failed, 2 dated skips |
| Other five public hostnames after the cloudflared restart | all 200 |

The NetworkPolicy row is the one that matters most. The rest confirm something
works; that one confirms something is prevented, which is the only kind of
evidence a control can offer.
