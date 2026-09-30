import { describe, expect, it } from "vitest";
import { SUPPORTED_LANGUAGES } from "./languages";

type Tree = { [k: string]: string | Tree };

const decisions = import.meta.glob<Tree>("./resources/*/decisions.json", {
  eager: true,
  import: "default",
});
const common = import.meta.glob<Tree>("./resources/*/common.json", {
  eager: true,
  import: "default",
});

const langOf = (path: string) => path.split("/")[2];
const byLang = (files: Record<string, Tree>) =>
  Object.fromEntries(Object.entries(files).map(([p, v]) => [langOf(p), v]));

const DECISIONS = byLang(decisions);
const COMMON = byLang(common);

/** Every leaf, as dot-path → string. */
function leaves(tree: Tree, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  for (const [k, v] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "string") out.set(path, v);
    else for (const [p, s] of leaves(v, path)) out.set(p, s);
  }
  return out;
}

const placeholders = (s: string) => [...s.matchAll(/\{\{\s*(\w+)\s*\}\}/g)].map((m) => m[1]).sort();

const en = leaves(DECISIONS.en);
// Languages whose resources exist; the key-check script covers only a subset,
// so this test is what holds the other locales to the English shape.
const translated = Object.keys(DECISIONS).filter((l) => l !== "en");

describe("decisions translations", () => {
  it("exist for every locale that has a common namespace", () => {
    expect(Object.keys(DECISIONS).sort()).toEqual(Object.keys(COMMON).sort());
  });

  it("cover every wired supported language", () => {
    const wired = SUPPORTED_LANGUAGES.filter((l) => l in COMMON);
    for (const lang of wired) expect(DECISIONS, lang).toHaveProperty(lang);
  });

  it.each(translated)("%s has exactly the English keys", (lang) => {
    expect([...leaves(DECISIONS[lang]).keys()].sort()).toEqual([...en.keys()].sort());
  });

  it.each(translated)("%s keeps every {{placeholder}} of each string", (lang) => {
    const mine = leaves(DECISIONS[lang]);
    for (const [path, text] of en) {
      expect(placeholders(mine.get(path) ?? ""), `${lang}: ${path}`).toEqual(placeholders(text));
    }
  });

  it.each(translated)("%s leaves no string empty", (lang) => {
    for (const [path, text] of leaves(DECISIONS[lang])) {
      expect(text.trim(), `${lang}: ${path}`).not.toBe("");
    }
  });

  it.each(Object.keys(COMMON))("%s names the playground in the nav", (lang) => {
    const nav = COMMON[lang].nav as Tree;
    expect(typeof nav.playground).toBe("string");
    expect((nav.playground as string).trim()).not.toBe("");
  });
});
