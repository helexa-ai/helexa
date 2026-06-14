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

## Build (for separate hosting)

```sh
VITE_API_BASE=http://bob.hanzalova.internal:13132 npm run build   # → dist/
```

The UI is hosted separately from the API (per design): serve the static
`dist/` from any web host and set `VITE_API_BASE` to the bob API origin.
If `VITE_API_BASE` is unset, the app calls `/api/...` on its own origin
(useful behind a reverse proxy that fronts both).
