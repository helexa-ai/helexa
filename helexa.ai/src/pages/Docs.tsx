import { useMemo, type ComponentProps } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";
import type { Components } from "react-markdown";
import Markdown from "../components/Markdown";
import {
  docHeadings,
  docNeighbours,
  docTree,
  firstDocSlug,
  getDoc,
  headingId,
} from "../lib/docs";
import "../App.css";

/**
 * Documentation.
 *
 * The sidebar is built from the content tree, which is built from the
 * filesystem — see lib/docs.ts. Nothing here enumerates pages, so adding
 * a markdown file is the whole of adding a page.
 *
 * The route is `/docs/*`, and `/docs` alone redirects to the first page
 * rather than rendering an index nobody maintains.
 */

/** Heading that carries an id, so the page rail and deep links work. */
function heading(level: 2 | 3) {
  const Tag = `h${level}` as const;
  return function Heading({ children }: { children?: React.ReactNode }) {
    const text = textOf(children);
    return (
      <Tag id={headingId(text)} className="hx-doc-heading">
        <a href={`#${headingId(text)}`} className="hx-doc-anchor" aria-hidden>
          #
        </a>
        {children}
      </Tag>
    );
  };
}

/** Recover the plain text of a rendered heading, for its anchor id. */
function textOf(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (typeof node === "object" && "props" in node) {
    return textOf((node as { props: { children?: React.ReactNode } }).props.children);
  }
  return "";
}

/**
 * Links inside documentation are mostly internal. Routing those through
 * react-router keeps navigation client-side and, more importantly, stops
 * them opening a new tab each — the shared renderer forces `_blank`
 * because it is built for untrusted model output, which docs are not.
 */
const LINK: Components["a"] = ({ href, children, ...props }) => {
  if (href && href.startsWith("/") && !href.startsWith("//")) {
    // `to` last: the spread carries markdown's own `href`, and letting
    // that win would leave the Link pointing nowhere.
    const rest = props as Omit<ComponentProps<typeof Link>, "to">;
    return (
      <Link {...rest} to={href}>
        {children}
      </Link>
    );
  }
  if (href?.startsWith("#")) {
    return (
      <a href={href} {...props}>
        {children}
      </a>
    );
  }
  return (
    <a href={href} {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
};

const DOC_COMPONENTS: Components = {
  a: LINK,
  h2: heading(2),
  h3: heading(3),
};

export default function Docs() {
  const { t, i18n } = useTranslation("docs");
  const location = useLocation();

  const slug = location.pathname.replace(/^\/docs\/?/, "").replace(/\/$/, "");
  const tree = useMemo(() => docTree(), []);
  const doc = useMemo(
    () => (slug ? getDoc(slug, i18n.language) : undefined),
    [slug, i18n.language],
  );
  const headings = useMemo(() => (doc ? docHeadings(doc.body) : []), [doc]);
  const { previous, next } = useMemo(
    () => (slug ? docNeighbours(slug) : {}),
    [slug],
  );

  // `/docs` has no index page of its own; send it to the first page so
  // there is one less document to keep in step with the tree.
  if (!slug) {
    const first = firstDocSlug();
    return first ? <Navigate to={`/docs/${first}`} replace /> : null;
  }

  // A section on its own — `/docs/operating` — is a directory rather than
  // a page. It is reachable: the prerenderer writes an index there, and
  // trimming a URL back to its parent is a thing people do. Send it to
  // the section's first page rather than claiming it does not exist.
  const section = tree.find((s) => s.id === slug);
  if (section?.pages.length) {
    return <Navigate to={`/docs/${section.pages[0].slug}`} replace />;
  }

  if (!doc) {
    return (
      <main className="container py-5">
        <h1>{t("notFound.title")}</h1>
        <p className="text-secondary">{t("notFound.body")}</p>
        <Link to="/docs">{t("notFound.back")}</Link>
      </main>
    );
  }

  return (
    <main className="hx-docs container-fluid">
      <div className="hx-docs-grid">
        <nav className="hx-docs-nav" aria-label={t("nav.label")}>
          {tree.map((section) => (
            <div key={section.id} className="hx-docs-section">
              <div className="hx-docs-section-label">
                {t(`sections.${section.id}`, { defaultValue: section.label })}
              </div>
              <ul>
                {section.pages.map((page) => (
                  <li key={page.slug}>
                    <Link
                      to={`/docs/${page.slug}`}
                      className={
                        page.slug === slug
                          ? "hx-docs-link active"
                          : "hx-docs-link"
                      }
                      aria-current={page.slug === slug ? "page" : undefined}
                    >
                      {page.sidebarLabel}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <article className="hx-docs-body">
          {doc.fallback && i18n.language !== "en" && (
            <div className="hx-docs-untranslated" role="note">
              {t("untranslated")}
            </div>
          )}
          <Markdown content={doc.body} components={DOC_COMPONENTS} />

          <hr className="hx-docs-rule" />
          <nav className="hx-docs-pager" aria-label={t("pager.label")}>
            {previous ? (
              <Link to={`/docs/${previous.slug}`} className="hx-docs-prev">
                <span>{t("pager.previous")}</span>
                {previous.sidebarLabel}
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link to={`/docs/${next.slug}`} className="hx-docs-next">
                <span>{t("pager.next")}</span>
                {next.sidebarLabel}
              </Link>
            )}
          </nav>
        </article>

        {headings.length > 0 && (
          <aside className="hx-docs-toc" aria-label={t("toc.label")}>
            <div className="hx-docs-toc-label">{t("toc.label")}</div>
            <ul>
              {headings.map((h) => (
                <li key={h.id} className={`depth-${h.depth}`}>
                  <a href={`#${h.id}`}>{h.text}</a>
                </li>
              ))}
            </ul>
          </aside>
        )}
      </div>
    </main>
  );
}
