#!/usr/bin/env node

/**
 * Write static HTML for every documentation page.
 *
 * Runs after `vite build` (which produces dist/index.html and the client
 * bundle) and after the SSR build of src/prerender/entry-server.tsx.
 * For each doc slug it takes the built index.html, substitutes the
 * rendered markup into #root and the real title and description into the
 * head, and writes dist/docs/<slug>/index.html.
 *
 * nginx serves these without configuration: `try_files $uri $uri/
 * /index.html` finds the directory before falling through to the SPA
 * shell. A visitor with JavaScript gets the shell replaced by the app on
 * mount; a crawler, or a reader without JavaScript, gets the page.
 *
 * The generated files are build output. They are not committed, and a
 * page deleted from content/ disappears from dist/ on the next build
 * because the whole directory is rewritten.
 */

import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const ROOT = path.resolve(
  path.dirname(url.fileURLToPath(import.meta.url)),
  "..",
);
const DIST = path.join(ROOT, "dist");
const SSR_ENTRY = path.join(ROOT, "dist-ssr", "entry-server.js");
const SITE = process.env.SITE_ORIGIN || "https://helexa.ai";

function fail(message) {
  console.error(`prerender: ${message}`);
  process.exit(1);
}

if (!fs.existsSync(path.join(DIST, "index.html"))) {
  fail("dist/index.html is missing — run `vite build` first");
}
if (!fs.existsSync(SSR_ENTRY)) {
  fail(`${path.relative(ROOT, SSR_ENTRY)} is missing — run the SSR build first`);
}

const { renderDoc, slugs } = await import(url.pathToFileURL(SSR_ENTRY).href);
const template = fs.readFileSync(path.join(DIST, "index.html"), "utf8");

// The shell must contain an empty #root for the markup to go into. If
// the template ever changes shape, say so rather than silently emitting
// pages with no content in them.
const ROOT_DIV = /<div id="root"><\/div>/;
if (!ROOT_DIV.test(template)) {
  fail('dist/index.html has no empty <div id="root"></div> to render into');
}

/** Replace <title> and description, or insert them if absent. */
function withHead(html, title, description, canonical) {
  let out = html.replace(
    /<title>.*?<\/title>/,
    `<title>${escape(title)} · helexa</title>`,
  );
  const meta = [
    description
      ? `<meta name="description" content="${escape(description)}">`
      : "",
    `<link rel="canonical" href="${escape(canonical)}">`,
  ]
    .filter(Boolean)
    .join("");
  return out.replace("</head>", `${meta}</head>`);
}

function escape(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

const written = [];
for (const slug of slugs()) {
  const doc = renderDoc(slug);
  if (!doc) {
    console.warn(`prerender: no content for ${slug}, skipping`);
    continue;
  }

  const canonical = `${SITE}/docs/${slug}`;
  const html = withHead(
    template.replace(ROOT_DIV, `<div id="root">${doc.html}</div>`),
    doc.title,
    doc.description,
    canonical,
  );

  const dir = path.join(DIST, "docs", ...slug.split("/"));
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, "index.html"), html);
  written.push({ slug, canonical });
}

// A sitemap so the pages are discoverable without relying on a crawler
// walking the sidebar.
const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${written.map((w) => `  <url><loc>${escape(w.canonical)}</loc></url>`).join("\n")}
</urlset>
`;
fs.mkdirSync(path.join(DIST, "docs"), { recursive: true });
fs.writeFileSync(path.join(DIST, "docs", "sitemap.xml"), sitemap);

console.log(`prerender: wrote ${written.length} pages + sitemap`);
