import { memo, useEffect, useState, type ComponentProps, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { LuCheck, LuCopy } from "react-icons/lu";
import type { Element, ElementContent } from "hast";

/**
 * Markdown renderer for assistant turns (#194).
 *
 * `react-markdown` builds a React element tree — it never assigns to
 * `innerHTML`, and raw HTML in the source stays inert because we do NOT
 * add `rehype-raw`. That is a deliberate security property, not a style
 * preference: with grounding (#177) the model's context contains
 * arbitrary third-party page text fetched by `read_page`, and it comes
 * back out through this component.
 *
 * User turns are NOT rendered through here. People type `*` and `_` as
 * punctuation and paste code without fences; transforming someone's own
 * prompt back at them is the wrong default.
 */

type RehypePlugins = ComponentProps<typeof ReactMarkdown>["rehypePlugins"];

/**
 * Syntax highlighting is loaded on demand: `rehype-highlight` pulls in
 * lowlight's common-language pack, which is larger than the markdown
 * parser itself. Deferring it keeps the initial chat payload small and
 * costs one re-render on the first code block anyone sees. The promise
 * is a module singleton so every bubble shares the one fetch.
 */
let highlightPlugins: RehypePlugins;
let highlightLoad: Promise<RehypePlugins> | undefined;
function loadHighlight(): Promise<RehypePlugins> {
  highlightLoad ??= import("rehype-highlight").then((m) => {
    highlightPlugins = [[m.default, { detect: true, ignoreMissing: true }]];
    return highlightPlugins;
  });
  return highlightLoad;
}

/** Flatten a hast subtree back to its source text (for copy-to-clipboard). */
function textOf(node: ElementContent | Element): string {
  if (node.type === "text") return node.value;
  if (node.type === "element") return node.children.map(textOf).join("");
  return "";
}

/** `language-rust` on the inner <code> → "rust"; undefined when unlabelled. */
function languageOf(code: Element | undefined): string | undefined {
  const classes = code?.properties?.className;
  if (!Array.isArray(classes)) return undefined;
  for (const c of classes) {
    const m = /^language-(.+)$/.exec(String(c));
    if (m) return m[1];
  }
  return undefined;
}

function CopyButton({ text }: { text: string }) {
  const { t } = useTranslation("chat");
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1600);
    return () => clearTimeout(id);
  }, [copied]);
  return (
    <button
      type="button"
      className="hx-icon-btn hx-code-copy"
      title={copied ? t("copiedCode") : t("copyCode")}
      aria-label={copied ? t("copiedCode") : t("copyCode")}
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => setCopied(true));
      }}
    >
      {copied ? <LuCheck size={13} /> : <LuCopy size={13} />}
    </button>
  );
}

/** Fenced code block: language chip + copy button above the <pre>. */
function Pre({ node, children }: { node?: Element; children?: ReactNode }) {
  const code = node?.children.find(
    (c): c is Element => c.type === "element" && c.tagName === "code",
  );
  return (
    <div className="hx-code">
      <div className="hx-code-bar">
        <span className="hx-code-lang">{languageOf(code)}</span>
        <CopyButton text={code ? textOf(code) : ""} />
      </div>
      <pre>{children}</pre>
    </div>
  );
}

const COMPONENTS: Components = {
  pre: Pre,
  // Model output routinely cites the open web; treat every link as
  // untrusted and never let it reach back into this tab.
  a: ({ children, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  // Wide tables scroll inside the bubble rather than widening it.
  table: ({ children, ...props }) => (
    <div className="hx-table-wrap">
      <table {...props}>{children}</table>
    </div>
  ),
};

/**
 * The streaming caret is appended to the markdown *source*, not drawn
 * with CSS. A `::after` on the last rendered block puts it on a line of
 * its own whenever that block is a list, a quote or a table — which is
 * most of the time, for these models. Appended to the source it lands
 * wherever the text actually stops, at every nesting depth, for the
 * price of not being able to tint it.
 */
const CARET = "▋";

function Markdown({
  content,
  caret = false,
  highlight = true,
  className,
}: {
  content: string;
  /** Draw a caret at the end of the text (the message is still arriving). */
  caret?: boolean;
  /**
   * Highlight code blocks. Pass `false` while the message is streaming:
   * every delta re-parses the whole document, and re-tokenising each
   * code block per token is the one part of that which actually hurts.
   * The finished message re-renders once with colour.
   */
  highlight?: boolean;
  className?: string;
}) {
  const [plugins, setPlugins] = useState<RehypePlugins>(highlightPlugins);
  useEffect(() => {
    if (!highlight || plugins) return;
    let cancelled = false;
    void loadHighlight().then((p) => {
      if (!cancelled) setPlugins(p);
    });
    return () => {
      cancelled = true;
    };
  }, [highlight, plugins]);

  return (
    <div className={className ? `hx-md ${className}` : "hx-md"}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={highlight ? plugins : undefined}
        components={COMPONENTS}
      >
        {/* Trim first: a trailing newline would make the caret its own
          * paragraph, which flickers a blank line in on every delta. */}
        {caret ? content.replace(/\s+$/, "") + CARET : content}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Memoised on the props: streaming writes an absolute content snapshot to
 * Dexie per delta, so `useLiveQuery` re-renders every bubble in the thread
 * on every token. Without this, each of those re-parses its own markdown.
 */
export default memo(Markdown);
