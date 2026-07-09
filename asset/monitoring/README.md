# Fleet monitoring — Prometheus scrape + Grafana dashboard

Visibility into the cortex `#137` capacity metrics (per-neuron:model load,
saturation, tok/s, load-shedding, and per-device GPU health) beyond what the
bench UI shows.

## Topology

- **cortex** runs on `hanzalova.internal` (10.6.0.46) and is the **only**
  Prometheus target — it exposes every `cortex_*` metric on `:31314`, already
  labelled by `{node,model}` / `{node,device}` from its neuron poller. neuron
  has no `/metrics` endpoint.
- **Prometheus + Grafana** run on `golgafrinchans.kosherinata.internal`
  (Prometheus is a podman quadlet, config bind-mounted at
  `/etc/prometheus/prometheus.yml`, `--web.enable-lifecycle` on). Its mesh IP
  toward hanzalova is `10.3.101.4`.

## Apply (three steps, in order)

### 1. Open the metrics port on the cortex host

cortex binds `0.0.0.0:31314` but firewalld has no rule for it, so a
cross-host scrape times out. Open it to the monitoring host **only**:

```sh
# on hanzalova.internal, as root
./firewalld-cortex-metrics.sh
```

### 2. Add the scrape job on the monitoring host

Append `prometheus-cortex.scrape.yml` into the `scrape_configs:` list of
`/etc/prometheus/prometheus.yml` on `golgafrinchans.kosherinata.internal`,
then hot-reload (no restart, lifecycle API is enabled):

```sh
curl -X POST http://localhost:9090/-/reload
# verify the target is UP:
curl -s 'http://localhost:9090/api/v1/targets' | jq '.data.activeTargets[]|select(.labels.job=="cortex")|{health,lastError}'
```

### 3. Import the dashboard

`grafana-helexa-fleet.json` is a raw dashboard model with a `datasource`
template variable, so it binds to whichever Prometheus data source you pick
at import. Import via the API (creds in `/etc/grafana/grafana.env`):

```sh
# on golgafrinchans, GF_SECURITY_ADMIN_USER/PASSWORD live in the env file
set -a; . /etc/grafana/grafana.env; set +a
jq -n --slurpfile d grafana-helexa-fleet.json \
  '{dashboard: $d[0], overwrite: true, folderUid: null}' \
| curl -s -u "$GF_SECURITY_ADMIN_USER:$GF_SECURITY_ADMIN_PASSWORD" \
    -H 'Content-Type: application/json' \
    -d @- http://localhost:3000/api/dashboards/db | jq '{status,uid,version}'
```

The dashboard lands at uid `helexa-fleet`. Re-importing with `overwrite:true`
updates it in place.

## What the dashboard shows

| Row | Panels |
|---|---|
| Capacity & saturation | saturation % (in_flight ÷ max_in_flight), in-flight vs ceiling, queue depth |
| Throughput | decode tok/s (revenue capacity), prefill tok/s |
| Backpressure & traffic | rejection rate by reason, request & error rate, TTFT p95/p50 |
| GPU health | VRAM used per device, GPU utilization %, temperature |

Templating: pick the Prometheus data source, then filter by `neuron` (node)
and `model`. Values are live at the cortex poll cadence (~10 s) scraped every
15 s.

## Note on the empirical knee

The live gauges above answer "how loaded is it right now". The
**sustainable-concurrency knee** (max N before latency/shedding breaks) is a
load-test result, not a live gauge — see `helexa-bench report --concurrency`
and `GET /api/concurrency` (#137 T3).
