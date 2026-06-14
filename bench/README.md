# helexa bench UI

A Vite + React (SWC, TypeScript) app that visualises the fleet benchmark
data collected by `helexa-bench`. It reads the read-only JSON API the
bench daemon serves (`crates/helexa-bench/src/api.rs`, default
`:13132` on bob).

Stack: React Router, react-bootstrap, Recharts.

## Pages

- **Overview** — latest median results per (host, model, scenario) cell.
- **Trends** — decode-tok/s and TTFT plotted across neuron build SHAs as
  releases roll out (the headline view). Pick host / model / scenario.
- **Runs** — filterable raw-run explorer.

## Develop

```sh
cd bench
npm install
npm run dev      # http://localhost:5173
```

`vite.config.ts` proxies `/api` → `http://bob.hanzalova.internal:13132`,
so the dev server talks to the live bench API with no CORS fuss. Point
the proxy elsewhere (or run a local `helexa-bench serve`) to develop
against other data.

## Production hosting

Public at **https://bench.helexa.ai** — nginx on the gateway
(`hanzalova.internal`) serves the static `dist/` and reverse-proxies
`/api` to the bench API on bob over WireGuard, so the SPA is same-origin
(no CORS) and the internal API stays off the public internet.

- `npm run build` is run with **no** `VITE_API_BASE` (the app calls
  `/api/...` on its own origin; nginx proxies it to bob).
- `.gitea/workflows/deploy.yml` (`deploy-bench-ui`) builds and rsyncs
  `dist/` to `/var/www/bench.helexa.ai` on every deploy.
- The nginx vhost (`asset/nginx/bench.helexa.ai.conf`) and the
  Let's Encrypt cert are one-time host setup in `script/infra-setup.sh`.

To host elsewhere instead, build with
`VITE_API_BASE=<bob-api-origin>` and serve the static `dist/`.
