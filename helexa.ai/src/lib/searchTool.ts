// The web_search tool (#177): definition sent with chat requests and
// the executor that calls the edge's /tools/web_search (SearXNG JSON
// API, GET-only, per-IP rate-limited at nginx). The agentic loop lives
// in useChat; this module is just the tool's two halves.

import type { MessageSource } from "../data/db";

export const WEB_SEARCH_TOOL = {
  type: "function",
  function: {
    name: "web_search",
    description:
      "Search the web for current information. Returns result titles, URLs and text snippets. " +
      "Use for recent events, facts that may postdate your training data, or anything you are unsure about.",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "The search query, in the language most likely to find good results.",
        },
      },
      required: ["query"],
    },
  },
} as const;

export interface SearchResult extends MessageSource {
  snippet: string;
}

const MAX_RESULTS = 5;
const SNIPPET_CHARS = 300;
const SEARCH_TIMEOUT_MS = 15_000;

/** Execute a web_search tool call. Returns the trimmed results plus the
 * JSON string to feed back to the model as the tool message content.
 * Failures return an error payload the model can react to (e.g. by
 * answering from its own knowledge) instead of throwing — a broken
 * search should degrade the answer, not kill the turn. */
export async function executeWebSearch(
  query: string,
  signal?: AbortSignal,
): Promise<{ results: SearchResult[]; content: string }> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), SEARCH_TIMEOUT_MS);
  signal?.addEventListener("abort", () => ctl.abort(), { once: true });
  try {
    const resp = await fetch(
      `/tools/web_search?q=${encodeURIComponent(query)}&format=json`,
      { signal: ctl.signal },
    );
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
    return { results, content: JSON.stringify({ results }) };
  } catch {
    return {
      results: [],
      content: JSON.stringify({ error: "search unavailable" }),
    };
  } finally {
    clearTimeout(timer);
  }
}
