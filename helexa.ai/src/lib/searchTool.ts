// The web_search tool (#177): definition sent with chat requests and
// the executor that calls the edge's /tools/web_search (SearXNG JSON
// API, GET-only, per-IP rate-limited at nginx). The agentic loop lives
// in useChat; this module is just the tool's two halves.
//
// Beyond plain snippets, SearXNG gives us three higher-quality signals
// that the executor forwards to the model:
// - `answers`: live/structured data — Open-Meteo weather (category
//   "weather"), ECB currency conversion. Authoritative where present.
// - `infoboxes`: Wikipedia/Wikidata fact summaries.
// - category "news" + time_range: recency-filtered current events.
// Snippets are SEO crawl text of unknown age (a weather site's snippet
// is a forecast fragment, not a reading — the "61°F in Prague" bug),
// so the tool result carries a note telling the model to prefer
// answers/infoboxes and to treat snippets as possibly stale.

import type { MessageSource } from "../data/db";

export const WEB_SEARCH_TOOL = {
  type: "function",
  function: {
    name: "web_search",
    description:
      "Search the web for current information. Returns structured live answers " +
      "(weather, currency conversion), encyclopedia infoboxes, and web results with " +
      "titles, URLs and text snippets. Use for recent events, facts that may postdate " +
      "your training data, or anything you are unsure about.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query, in the language most likely to find good results.",
        },
        category: {
          type: "string",
          enum: ["general", "news", "weather", "science", "it"],
          description:
            "Search category. Use 'weather' for current weather or forecasts (returns live " +
            "measurements), 'news' for current events, 'science'/'it' for scholarly or " +
            "technical sources. Default 'general'.",
        },
        time_range: {
          type: "string",
          enum: ["day", "week", "month", "year"],
          description:
            "Restrict results to this recency window. Recommended with 'news' for current events.",
        },
      },
      required: ["query"],
    },
  },
} as const;

export interface SearchResult extends MessageSource {
  snippet: string;
}

export interface SearchArgs {
  query: string;
  category?: string;
  time_range?: string;
}

const MAX_RESULTS = 5;
const MAX_ANSWERS = 3;
const MAX_INFOBOXES = 2;
const SNIPPET_CHARS = 300;
const INFOBOX_CHARS = 400;
const SEARCH_TIMEOUT_MS = 15_000;
const CATEGORIES = new Set(["general", "news", "weather", "science", "it"]);
const TIME_RANGES = new Set(["day", "week", "month", "year"]);

const SNIPPET_NOTE =
  "Snippets are search-index text of unknown age and may quote forecasts, lows/highs or " +
  "old data rather than current conditions. Prefer `answers` (live, structured) and " +
  "`infoboxes` (encyclopedic) over snippets when they are present.";

/** Compact a SearXNG answer for the model. Weather answers carry a huge
 * hourly `forecasts` array — keep the live reading plus a short outlook;
 * other answers (currency, plugins) are already a single string. */
function compactAnswer(a: Record<string, unknown>): unknown {
  if (typeof a.answer === "string") return a.answer;
  const current = a.current as { summary?: string } | undefined;
  if (current) {
    const forecasts = (a.forecasts as { summary?: string }[] | undefined) ?? [];
    return {
      current,
      outlook: forecasts.slice(0, 4).map((f) => f.summary).filter(Boolean),
    };
  }
  return JSON.stringify(a).slice(0, 500);
}

/** Execute a web_search tool call. Returns the trimmed results plus the
 * JSON string to feed back to the model as the tool message content.
 * Failures return an error payload the model can react to (e.g. by
 * answering from its own knowledge) instead of throwing — a broken
 * search should degrade the answer, not kill the turn. */
export async function executeWebSearch(
  args: SearchArgs,
  signal?: AbortSignal,
): Promise<{ results: SearchResult[]; content: string }> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), SEARCH_TIMEOUT_MS);
  signal?.addEventListener("abort", () => ctl.abort(), { once: true });
  try {
    const params = new URLSearchParams({ q: args.query, format: "json" });
    if (args.category && CATEGORIES.has(args.category) && args.category !== "general") {
      params.set("categories", args.category);
    }
    if (args.time_range && TIME_RANGES.has(args.time_range)) {
      params.set("time_range", args.time_range);
    }
    const resp = await fetch(`/tools/web_search?${params}`, { signal: ctl.signal });
    if (!resp.ok) {
      return {
        results: [],
        content: JSON.stringify({ error: `search failed (${resp.status})` }),
      };
    }
    const body = await resp.json();
    const results: SearchResult[] = (body?.results ?? [])
      .slice(0, MAX_RESULTS)
      .map((r: { title?: string; url?: string; content?: string }) => ({
        title: r.title ?? "",
        url: r.url ?? "",
        snippet: (r.content ?? "").slice(0, SNIPPET_CHARS),
      }))
      .filter((r: SearchResult) => r.url);
    const answers = ((body?.answers ?? []) as Record<string, unknown>[])
      .slice(0, MAX_ANSWERS)
      .map(compactAnswer);
    const infoboxes = ((body?.infoboxes ?? []) as { infobox?: string; content?: string }[])
      .slice(0, MAX_INFOBOXES)
      .map((b) => ({ title: b.infobox ?? "", summary: (b.content ?? "").slice(0, INFOBOX_CHARS) }))
      .filter((b) => b.summary);
    const payload: Record<string, unknown> = { results };
    if (answers.length) payload.answers = answers;
    if (infoboxes.length) payload.infoboxes = infoboxes;
    if (results.length) payload.note = SNIPPET_NOTE;
    return { results, content: JSON.stringify(payload) };
  } catch {
    return {
      results: [],
      content: JSON.stringify({ error: "search unavailable" }),
    };
  } finally {
    clearTimeout(timer);
  }
}
