// @vitest-environment jsdom
import "../../test/i18n";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ResultsPane, { ErrorPanel, type RunState } from "./ResultsPane";
import { issueFromError } from "../../lib/decisionRequest";
import { DecisionError, parseDecisionResponse, type ErrorKind } from "../../lib/decisionClient";
import { isMap, parseOrdered, stringifyOrdered } from "../../lib/orderedJson";

const reference = JSON.parse(
  // jsdom gives modules a non-file import.meta.url; tests run from helexa.ai/.
  readFileSync(resolve(process.cwd(), "../crates/neuron/src/harness/testdata/laya/reference.json"), "utf8"),
) as { cases: { name: string; request: unknown; response: unknown }[] };

/** A finished run built from a recorded reference case. */
function doneFrom(name: string): RunState {
  const c = reference.cases.find((x) => x.name === name)!;
  const body = JSON.stringify(c.request);
  const raw = JSON.stringify(c.response);
  const req = parseOrdered(body);
  const questions = isMap(req) && isMap(req.get("questions")) ? (req.get("questions") as Map<string, never>) : new Map();
  return {
    status: "done",
    body,
    questions,
    result: { response: parseDecisionResponse(raw), raw, timing: { clientMs: 42, serverMs: 7.5 } },
  };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("ResultsPane answers", () => {
  it("shows a card per question, in request order, with every option's bar", () => {
    render(<ResultsPane run={doneFrom("multi_question")} />);
    const cards = screen.getAllByRole("article");
    expect(cards.map((c) => c.getAttribute("aria-label"))).toEqual(["queue", "urgency", "refund", "tone", "spam"]);
    // A choice card: every option, the chosen one marked and headlined.
    const queue = within(cards[0]);
    expect(queue.getAllByRole("listitem")).toHaveLength(3);
    const chosen = cards[0].querySelector('li[aria-current="true"]')!;
    expect(chosen.textContent).toContain("billing");
    expect(cards[0].querySelector(".hx-pg-headline")!.textContent).toBe("billing");
    // Described choice options show their description.
    expect(queue.getByText("billing and refunds")).toBeTruthy();
  });

  it("shows a score's expected level and its legend", () => {
    render(<ResultsPane run={doneFrom("score_basic")} />);
    const card = screen.getByRole("article");
    expect(card.querySelector(".hx-pg-headline")!.textContent).toMatch(/^Expected level \d\.\d\d$/);
    expect(within(card).getByText(/^0 · calm$/)).toBeTruthy();
    expect(within(card).getAllByRole("listitem")).toHaveLength(4);
  });

  it("shows a noul's P(true) and uses the question's own labels", () => {
    render(<ResultsPane run={doneFrom("noul_custom_criteria_and_labels")} />);
    const card = screen.getByRole("article");
    expect(card.querySelector(".hx-pg-headline")!.textContent).toMatch(/likely true$/);
    const labels = [...card.querySelectorAll(".hx-pg-bar-label span:first-child")].map((s) => s.textContent);
    expect(labels).toEqual(["locked", "not locked"]);
    expect(within(card).getByText("the user cannot sign in")).toBeTruthy();
  });

  it("explains confidence once, not per card", () => {
    render(<ResultsPane run={doneFrom("multi_question")} />);
    expect(screen.getAllByText(/calibrated number to set a threshold on/)).toHaveLength(1);
    expect(screen.getAllByText("Answer confidence")).toHaveLength(5);
  });
});

describe("ResultsPane routing and usage", () => {
  it("makes the switch to the multilingual checkpoint obvious", () => {
    render(<ResultsPane run={doneFrom("ml_khmer")} />);
    const routing = screen.getByRole("region", { name: "Routing" });
    const pill = routing.querySelector(".hx-pg-pill")!;
    expect(pill.textContent).toBe("multilingual");
    expect(pill.classList.contains("hx-pg-pill-ml")).toBe(true);
    expect(routing.textContent).toContain("non-Latin script (khmer");
    expect(within(routing).getByRole("list", { name: /per script/ }).textContent).toContain("khmer");
  });

  it("says when a pinned checkpoint skipped detection", () => {
    render(<ResultsPane run={doneFrom("model_pinned_typed_decisions")} />);
    const routing = screen.getByRole("region", { name: "Routing" });
    expect(routing.querySelector(".hx-pg-pill")!.textContent).toBe("typed-decisions");
    expect(routing.querySelector(".hx-pg-pill-ml")).toBeNull();
    expect(routing.textContent).toContain("language detection was skipped");
  });

  it("shows input tokens as the metered unit, and both timings", () => {
    const run = doneFrom("choice_basic");
    render(<ResultsPane run={run} />);
    const usage = screen.getByRole("region", { name: "Usage and timing" });
    expect(usage.textContent).toContain("41");
    expect(usage.textContent).toContain("input tokens (metered)");
    expect(usage.textContent).toContain("7.5 ms");
    expect(usage.textContent).toContain("42 ms");
  });

  it("offers the raw request in the order it was sent", () => {
    // "10" before "2": the order JSON.parse would silently swap.
    const body = '{"10":1,"2":2}';
    render(<ResultsPane run={{ ...(doneFrom("choice_basic") as Extract<RunState, { status: "done" }>), body }} />);
    const pre = screen.getByText("Request JSON").closest("details")!.querySelector("pre")!;
    expect(pre.textContent).toBe(stringifyOrdered(parseOrdered(body)));
    expect(pre.textContent!.indexOf('"10"')).toBeLessThan(pre.textContent!.indexOf('"2"'));
  });
});

describe("ErrorPanel", () => {
  const kinds: ErrorKind[] = [
    "validation",
    "too_large",
    "rate_limited",
    "unavailable",
    "not_found",
    "server",
    "network",
    "timeout",
    "bad_response",
  ];

  it.each(kinds)("renders a %s error with its own title", (kind) => {
    render(<ErrorPanel error={new DecisionError(kind, "boom", { status: 400 })} />);
    const title = screen.getByRole("alert").querySelector("strong")!.textContent!;
    expect(title).not.toMatch(/^errors\./);
    expect(title.length).toBeGreaterThan(3);
  });

  it("tells a network failure from a server rejection", () => {
    render(<ErrorPanel error={new DecisionError("network", "Failed to fetch")} />);
    expect(screen.getByRole("alert").textContent).toContain("Couldn't reach the service");
    cleanup();
    render(<ErrorPanel error={new DecisionError("server", "x", { status: 500 })} />);
    expect(screen.getByRole("alert").textContent).toContain("The service failed");
  });

  it("shows a validation detail with the question it names", () => {
    render(
      <ErrorPanel
        error={new DecisionError("validation", "question 'urgency': a score question needs at least one level", {
          status: 422,
          questionId: "urgency",
        })}
      />,
    );
    const alert = screen.getByRole("alert");
    expect(alert.querySelector("code")!.textContent).toBe("urgency");
    expect(alert.textContent).toContain("needs at least one level");
  });

  it("counts down a Retry-After", () => {
    vi.useFakeTimers();
    render(<ErrorPanel error={new DecisionError("rate_limited", "x", { status: 429, retryAfter: 3 })} />);
    expect(screen.getByText("You can try again in 3 s.")).toBeTruthy();
    act(() => void vi.advanceTimersByTime(1100));
    expect(screen.getByText("You can try again in 2 s.")).toBeTruthy();
    act(() => void vi.advanceTimersByTime(3000));
    expect(screen.getByText("You can try again now.")).toBeTruthy();
  });

  it("explains the public rate limit when no Retry-After came back", () => {
    render(<ErrorPanel error={new DecisionError("rate_limited", "x", { status: 429 })} />);
    expect(screen.getByRole("alert").textContent).toContain("10 requests a minute");
  });
});

describe("issueFromError", () => {
  it("turns a 422 naming a question into an issue under that question", () => {
    const issue = issueFromError(
      new DecisionError("validation", "question 'q1': a choice question needs at least one criterion", {
        questionId: "q1",
      }),
    );
    expect(issue).toMatchObject({
      field: "questions",
      questionId: "q1",
      params: { id: "q1", detail: "a choice question needs at least one criterion" },
    });
    expect(issueFromError(new DecisionError("rate_limited", "x"))).toBeNull();
    expect(issueFromError(new DecisionError("validation", "request body must be valid JSON"))).toBeNull();
  });
});
