// Documentation content layer.
//
// The navigation is *derived from the filesystem*, never registered
// anywhere. Adding a file or a folder under content/docs/ puts it in the
// sidebar; renaming it moves it. There is no index to keep in sync,
// because an index that can drift is an index that eventually lies.
//
// Ordering and labels come from the path itself, with front matter as
// the override:
//
//   content/docs/1-using/02-chat.md   → section "using", slug "using/chat"
//
// A leading `NN-` on a directory or file sets sort order and is stripped
// from the slug, so pages can be reordered without changing their URLs
// as long as the name after the prefix is unchanged.
//
// Translations live in a parallel tree and fall back per *file*:
//
//   content/i18n/<lang>/1-using/02-chat.md
//
// A locale that has translated three pages gets those three; everything
// else stays English. This is deliberately outside the JSON namespace
// parity that scripts/check-i18n-keys.mjs enforces — docs are prose, not
// interface strings, and requiring all 42 locales to move together would
// mean no page could ever be added without breaking the other 41.

/** Front matter recognised on a doc page. All fields optional. */
export interface DocFrontMatter {
  title?: string;
  sidebar_label?: string;
  sidebar_position?: number;
  description?: string;
}

export interface DocMeta {
  /** URL path below /docs, e.g. "using/chat". */
  slug: string;
  /** Section id, e.g. "using". */
  section: string;
  title: string;
  sidebarLabel: string;
  description?: string;
  position: number;
}

export interface DocPage extends DocMeta {
  /** Markdown body, front matter removed. */
  body: string;
  /** True when the requested locale had no translation and English was used. */
  fallback: boolean;
}

export interface DocSection {
  id: string;
  label: string;
  position: number;
  pages: DocMeta[];
}

// Eager, so the whole tree is in the bundle. That is a deliberate
// trade at this size — the sidebar needs every page's front matter to
// exist before anything renders, and fetching each page separately
// would put a network round trip in front of every navigation.
//
// The cost is that visitors who never open the docs still download
// them. At ~16 kB gzipped for the current tree that is well below the
// noise floor of the app bundle. Past roughly 150 kB gzipped it stops
// being free: generate a front-matter manifest at build time and make
// the bodies dynamic imports.
const EN = import.meta.glob("/content/docs/**/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const TRANSLATED = import.meta.glob("/content/i18n/**/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

/**
 * Minimal front-matter reader.
 *
 * Handles `key: value` pairs between `---` fences, with optional quotes.
 * That is the whole of what doc front matter uses, and a real YAML
 * parser is a dependency plus a parsing surface for something that only
 * ever carries four scalar fields. Anything it does not understand is
 * ignored rather than throwing, so a malformed header costs a title, not
 * the page.
 */
export function parseFrontMatter(raw: string): {
  data: DocFrontMatter;
  body: string;
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/.exec(raw);
  if (!match) return { data: {}, body: raw };

  const data: DocFrontMatter = {};
  for (const line of match[1].split(/\r?\n/)) {
    const kv = /^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$/.exec(line.trim());
    if (!kv) continue;
    const key = kv[1];
    const value = kv[2].replace(/^["']|["']$/g, "").trim();
    if (key === "sidebar_position") {
      const n = Number(value);
      if (Number.isFinite(n)) data.sidebar_position = n;
    } else if (
      key === "title" ||
      key === "sidebar_label" ||
      key === "description"
    ) {
      data[key] = value;
    }
  }
  return { data, body: raw.slice(match[0].length) };
}

/** Split a `NN-name` segment into its sort key and its slug part. */
function splitOrdered(segment: string): { position: number; name: string } {
  const m = /^(\d+)[-_](.*)$/.exec(segment);
  if (!m) return { position: Number.MAX_SAFE_INTEGER, name: segment };
  return { position: Number(m[1]), name: m[2] };
}

/** "getting-started" → "Getting started". Fallback when no title is given. */
function humanise(name: string): string {
  const spaced = name.replace(/[-_]/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** First markdown heading, used when front matter carries no title. */
function firstHeading(body: string): string | undefined {
  const m = /^\s*#\s+(.+?)\s*$/m.exec(body);
  return m?.[1];
}

interface ParsedPath {
  slug: string;
  section: string;
  sectionLabel: string;
  sectionPosition: number;
  position: number;
  name: string;
}

/**
 * Turn a content path into routing and ordering information.
 * Returns null for paths that are not a section page (a stray file at
 * the tree root, for instance), so the caller can skip them rather than
 * inventing a section for them.
 */
function parsePath(fullPath: string, root: string): ParsedPath | null {
  const rel = fullPath.slice(root.length).replace(/^\/+/, "");
  const parts = rel.replace(/\.md$/, "").split("/");
  if (parts.length < 2) return null;

  const [rawSection, ...rest] = parts;
  const section = splitOrdered(rawSection);
  const file = splitOrdered(rest[rest.length - 1]);
  const middle = rest.slice(0, -1).map((p) => splitOrdered(p).name);

  return {
    slug: [section.name, ...middle, file.name].join("/"),
    section: section.name,
    sectionLabel: humanise(section.name),
    sectionPosition: section.position,
    position: file.position,
    name: file.name,
  };
}

function metaFor(path: string, raw: string, root: string): DocMeta | null {
  const parsed = parsePath(path, root);
  if (!parsed) return null;
  const { data, body } = parseFrontMatter(raw);
  const title = data.title ?? firstHeading(body) ?? humanise(parsed.name);
  return {
    slug: parsed.slug,
    section: parsed.section,
    title,
    sidebarLabel: data.sidebar_label ?? title,
    description: data.description,
    position: data.sidebar_position ?? parsed.position,
  };
}

/**
 * The full navigation tree, ordered by directory prefix then file
 * prefix. Always built from English, so the sidebar has the same shape
 * in every locale and a partially-translated language does not appear to
 * be missing pages.
 */
export function docTree(): DocSection[] {
  const sections = new Map<string, DocSection>();

  for (const [path, raw] of Object.entries(EN)) {
    const parsed = parsePath(path, "/content/docs");
    const meta = metaFor(path, raw, "/content/docs");
    if (!parsed || !meta) continue;

    let section = sections.get(parsed.section);
    if (!section) {
      section = {
        id: parsed.section,
        label: parsed.sectionLabel,
        position: parsed.sectionPosition,
        pages: [],
      };
      sections.set(parsed.section, section);
    }
    section.pages.push(meta);
  }

  const ordered = [...sections.values()].sort(
    (a, b) => a.position - b.position || a.id.localeCompare(b.id),
  );
  for (const section of ordered) {
    section.pages.sort(
      (a, b) => a.position - b.position || a.title.localeCompare(b.title),
    );
  }
  return ordered;
}

/** Every slug, in sidebar order. The prerenderer walks this. */
export function allDocSlugs(): string[] {
  return docTree().flatMap((s) => s.pages.map((p) => p.slug));
}

/** The first page of the first section — where /docs lands. */
export function firstDocSlug(): string | undefined {
  return docTree()[0]?.pages[0]?.slug;
}

function findBySlug(
  index: Record<string, string>,
  root: string,
  slug: string,
): { path: string; raw: string } | undefined {
  for (const [path, raw] of Object.entries(index)) {
    const parsed = parsePath(path, root);
    if (parsed?.slug === slug) return { path, raw };
  }
  return undefined;
}

/**
 * Resolve a slug to a page, preferring `lang` and falling back to
 * English for that individual file.
 */
export function getDoc(slug: string, lang?: string): DocPage | undefined {
  const english = findBySlug(EN, "/content/docs", slug);
  if (!english) return undefined;

  let source = english;
  let fallback = true;
  if (lang && lang !== "en") {
    const translated = findBySlug(TRANSLATED, `/content/i18n/${lang}`, slug);
    if (translated) {
      source = translated;
      fallback = false;
    }
  } else if (lang === "en" || !lang) {
    fallback = false;
  }

  // Metadata comes from the English source so ordering and sidebar
  // labels cannot drift between locales; only prose is localised.
  const meta = metaFor(english.path, english.raw, "/content/docs");
  if (!meta) return undefined;

  const { data, body } = parseFrontMatter(source.raw);
  return {
    ...meta,
    title: fallback ? meta.title : (data.title ?? meta.title),
    body,
    fallback,
  };
}

/** Sidebar neighbours, for previous/next links at the foot of a page. */
export function docNeighbours(slug: string): {
  previous?: DocMeta;
  next?: DocMeta;
} {
  const flat = docTree().flatMap((s) => s.pages);
  const i = flat.findIndex((p) => p.slug === slug);
  if (i < 0) return {};
  return { previous: flat[i - 1], next: flat[i + 1] };
}

/** Headings for the "on this page" rail. */
export interface DocHeading {
  depth: 2 | 3;
  text: string;
  id: string;
}

/** Slugify a heading into a stable anchor id. */
export function headingId(text: string): string {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

/**
 * Extract h2/h3 headings for the page rail.
 *
 * Fenced code blocks are skipped: a `#` at the start of a line inside a
 * shell example is a comment, not a heading, and every install page is
 * full of them.
 */
export function docHeadings(body: string): DocHeading[] {
  const out: DocHeading[] = [];
  let inFence = false;
  for (const line of body.split(/\r?\n/)) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const m = /^(#{2,3})\s+(.+?)\s*$/.exec(line);
    if (!m) continue;
    const text = m[2].replace(/[*_`]/g, "").trim();
    out.push({ depth: m[1].length as 2 | 3, text, id: headingId(text) });
  }
  return out;
}
