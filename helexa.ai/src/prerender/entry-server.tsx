// Static rendering of documentation pages, for crawlers and for readers
// without JavaScript.
//
// helexa.ai is a client-rendered app, which is the right shape for a
// chat workspace and the wrong shape for documentation: a crawler asking
// for /docs/using/api gets an empty shell, so the pages cannot be found
// by the people most likely to want them. This module renders each doc
// to real HTML at build time; scripts/prerender-docs.mjs writes the
// files.
//
// There is no hydration. The SPA mounts into #root and replaces whatever
// it finds, exactly as it does today, so the prerendered markup only has
// to be correct — it does not have to match what React would produce on
// the client. That is what makes this safe to bolt onto a client-only
// app.
//
// Interactive chrome is deliberately absent from the static output. The
// shared Markdown component carries copy buttons, lazy syntax
// highlighting and translated labels, all of which need a browser and a
// live i18n instance and none of which mean anything to a crawler. The
// parser and plugins are the same; only the interactive overrides are
// dropped.

import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { allDocSlugs, docTree, getDoc, headingId } from "../lib/docs";

export interface PrerenderedDoc {
  slug: string;
  title: string;
  description?: string;
  html: string;
}

/** Escape text for interpolation into an HTML attribute or body. */
function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * The sidebar, rendered as plain links.
 *
 * Included because crawlers discover pages by following links: without
 * it every doc is an island reachable only from the sitemap, and the
 * pages deepest in the tree are the ones that get missed.
 */
function navHtml(currentSlug: string): string {
  const sections = docTree()
    .map((section) => {
      const items = section.pages
        .map((page) => {
          const current =
            page.slug === currentSlug ? ' aria-current="page"' : "";
          return `<li><a href="/docs/${page.slug}"${current}>${escapeHtml(
            page.sidebarLabel,
          )}</a></li>`;
        })
        .join("");
      return `<div><h2>${escapeHtml(section.label)}</h2><ul>${items}</ul></div>`;
    })
    .join("");
  return `<nav aria-label="Documentation">${sections}</nav>`;
}

/** Render one documentation page to static HTML. */
export function renderDoc(slug: string): PrerenderedDoc | undefined {
  const doc = getDoc(slug, "en");
  if (!doc) return undefined;

  const body = renderToStaticMarkup(
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Anchors match the client rail, so a link copied from the
        // rendered page resolves the same way in the SPA.
        h2: ({ children }) => <h2 id={idOf(children)}>{children}</h2>,
        h3: ({ children }) => <h3 id={idOf(children)}>{children}</h3>,
      }}
    >
      {doc.body}
    </ReactMarkdown>,
  );

  return {
    slug,
    title: doc.title,
    description: doc.description,
    html: `${navHtml(slug)}<article>${body}</article>`,
  };
}

/** Heading id from rendered children, mirroring the client. */
function idOf(children: React.ReactNode): string {
  const text = flatten(children);
  return headingId(text);
}

function flatten(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(flatten).join("");
  if (typeof node === "object" && "props" in node) {
    return flatten(
      (node as { props: { children?: React.ReactNode } }).props.children,
    );
  }
  return "";
}

/** Every page the prerenderer should emit. */
export function slugs(): string[] {
  return allDocSlugs();
}
