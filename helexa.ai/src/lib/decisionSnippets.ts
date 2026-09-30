// The playground's current request as code a visitor can run.
//
// Built from the exact body the playground would send (key order
// included), for the public endpoint. No key is ever embedded: snippets
// read `HELEXA_API_KEY` from the environment, and the service also answers
// anonymous requests within its rate limit.

import { JsonNumber, isMap, parseOrdered, stringifyOrdered, type OJ } from "./orderedJson";

export type SnippetLang = "curl" | "python" | "laya";
export const SNIPPET_LANGS: SnippetLang[] = ["curl", "python", "laya"];

/** Quote for a POSIX shell single-quoted string. */
function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

/** A Python literal, keeping key order (Python dicts preserve it). */
export function pyLiteral(v: OJ, indent = 0): string {
  const pad = " ".repeat(indent);
  const inner = " ".repeat(indent + 4);
  if (v === null) return "None";
  if (v === true) return "True";
  if (v === false) return "False";
  if (v instanceof JsonNumber) return v.raw;
  if (typeof v === "string") return JSON.stringify(v);
  if (Array.isArray(v)) {
    if (!v.length) return "[]";
    return `[\n${v.map((x) => inner + pyLiteral(x, indent + 4)).join(",\n")},\n${pad}]`;
  }
  if (!v.size) return "{}";
  return `{\n${[...v]
    .map(([k, x]) => `${inner}${JSON.stringify(k)}: ${pyLiteral(x, indent + 4)}`)
    .join(",\n")},\n${pad}}`;
}

/**
 * `body` is the request body as the playground sends it; `baseUrl` the
 * origin that serves `/v1/systemone`.
 */
export function snippet(lang: SnippetLang, body: string, baseUrl: string): string {
  const base = baseUrl.replace(/\/$/, "");
  const url = `${base}/v1/systemone`;
  const root = parseOrdered(body);
  const get = (k: string): OJ => (isMap(root) ? (root.get(k) ?? null) : null);
  switch (lang) {
    case "curl":
      return [
        `curl ${url} \\`,
        `  -H 'Content-Type: application/json' \\`,
        `  -H "Authorization: Bearer $HELEXA_API_KEY" \\`,
        `  -d ${shellQuote(stringifyOrdered(root))}`,
      ].join("\n");
    case "python":
      return [
        "import json",
        "import os",
        "import urllib.request",
        "",
        `body = ${pyLiteral(root)}`,
        "",
        "req = urllib.request.Request(",
        `    ${JSON.stringify(url)},`,
        '    data=json.dumps(body).encode("utf-8"),',
        "    headers={",
        '        "Content-Type": "application/json",',
        '        "Authorization": "Bearer " + os.environ.get("HELEXA_API_KEY", ""),',
        "    },",
        ")",
        "with urllib.request.urlopen(req, timeout=30) as resp:",
        "    result = json.load(resp)",
        'for qid, answer in result["answers"].items():',
        "    print(qid, answer)",
      ].join("\n");
    case "laya":
      return [
        "# pip install laya — its remote client speaks the same protocol.",
        "import os",
        "from laya.integrations.llamaindex import _call_remote",
        "",
        `state = ${pyLiteral(get("state"))}`,
        `questions = ${pyLiteral(get("questions"))}`,
        "",
        "result = _call_remote(",
        `    ${JSON.stringify(base)},`,
        "    state,",
        "    questions,",
        `    model=${pyLiteral(get("model"))},`,
        '    api_key=os.environ.get("HELEXA_API_KEY"),',
        ")",
        'print(result["answers"])',
      ].join("\n");
  }
}
