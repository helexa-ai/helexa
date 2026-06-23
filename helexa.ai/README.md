# helexa.ai

The public-beta frontend for the helexa mesh: a chat-first landing experience
(anonymous + authenticated, with all chat history kept client-side in
IndexedDB — no server-side history), a `/mission` page on European digital
sovereignty, and full account self-service (register, recover, manage API
keys, set per-key limits, redeem top-up codes) against `helexa-upstream`.

Vite + React (SWC) + TypeScript + react-bootstrap + react-router + react-i18next.
Lives as a top-level folder in the cortex monorepo; it is **not** a Cargo crate.

## Develop

```sh
cd helexa.ai
npm install
cp .env.example .env.local   # adjust backend URLs
npm run dev                  # vite dev server, proxies /v1+/health → router, /api → upstream
```

Other scripts: `npm run build` (`tsc -b && vite build` → `dist/`), `npm run
preview`, `npm run lint`, `npm run typecheck`.

In dev, `vite.config.ts` proxies the mesh data-plane (helexa-router) and the
account control-plane (helexa-upstream) same-origin. Run a local router
(`cargo run -p helexa-router`) for the chat path and a local helexa-upstream
for the account path.

## Status

F0 scaffold. Theming + i18n (33 languages, usage-ordered selector), the
`/mission` page, the chat workspace (Dexie + streaming), and the account
dashboard land in subsequent phases — see
`~/.claude/plans/we-need-to-plan-modular-graham.md`.

## Deploy (public beta)

Build the SPA and serve it from edge nginx on the **same origin** as the
two backends — so the browser makes no cross-origin request (no CORS) and
the user's API key rides as a first-party bearer.

```sh
npm ci && npm run build          # → dist/
sudo cp -r dist/* /var/www/helexa.ai/
sudo cp deploy/nginx.conf /etc/nginx/conf.d/helexa.ai.conf   # adjust upstreams + TLS
sudo nginx -t && sudo systemctl reload nginx
```

`deploy/nginx.conf` routes `/` → SPA (history fallback), `/v1` + `/health`
→ helexa-router, and `/api/` → helexa-upstream `/web/v1/`. Set
`VITE_PUBLIC_BETA=true` at build time for the beta banner. There is **no
server-side chat history**: conversations live only in the browser
(IndexedDB).
