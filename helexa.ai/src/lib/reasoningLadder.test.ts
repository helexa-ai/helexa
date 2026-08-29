import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchReasoningLadder,
  reasoningOptions,
  resolveSelection,
  type ReasoningRung,
} from "./reasoningLadder";

const LADDER: ReasoningRung[] = [
  { effort: "low" },
  { effort: "medium" },
  { effort: "xhigh", default: true },
];

const BUSY: ReasoningRung[] = [
  { effort: "low" },
  { effort: "medium" },
  { effort: "xhigh", default: true, unavailable_reason: "max_in_flight=8" },
];

describe("reasoningOptions", () => {
  it("offers every rung on a quiet node", () => {
    const o = reasoningOptions(LADDER);
    expect(o.available.map((r) => r.effort)).toEqual(["low", "medium", "xhigh"]);
    expect(o.withheld).toEqual([]);
  });

  it("separates a rung the deployment is withholding", () => {
    const o = reasoningOptions(BUSY);
    expect(o.available.map((r) => r.effort)).toEqual(["low", "medium"]);
    expect(o.withheld.map((r) => r.effort)).toEqual(["xhigh"]);
  });

  it("yields nothing for a model with no ladder", () => {
    // A non-reasoning model, or a cortex too old to publish one. The
    // caller should render no control rather than an empty one.
    expect(reasoningOptions(undefined).available).toEqual([]);
    expect(reasoningOptions([]).available).toEqual([]);
  });
});

describe("resolveSelection", () => {
  it("keeps a remembered rung that is still offered", () => {
    expect(resolveSelection("medium", reasoningOptions(LADDER))).toBe("medium");
  });

  it("falls back to off when the remembered rung is withheld", () => {
    // Deliberately not "promote to the next best rung": that chooses on
    // the user's behalf without telling them, and the model's own
    // default is `xhigh` — the very rung most likely to be withheld.
    expect(resolveSelection("xhigh", reasoningOptions(BUSY))).toBeNull();
  });

  it("stays off when nothing was remembered", () => {
    expect(resolveSelection(null, reasoningOptions(LADDER))).toBeNull();
  });

  it("stays off when the model has no ladder at all", () => {
    expect(resolveSelection("medium", reasoningOptions([]))).toBeNull();
  });
});

describe("fetchReasoningLadder", () => {
  afterEach(() => vi.unstubAllGlobals());

  function respond(body: unknown, ok = true) {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok, json: () => Promise.resolve(body) }),
    );
  }

  it("returns the ladder for the requested model", async () => {
    respond({
      data: [
        { id: "other/model", reasoning_budget: [{ effort: "wrong" }] },
        { id: "Qwen/Qwen3.8-27B", reasoning_budget: BUSY },
      ],
    });
    const rungs = await fetchReasoningLadder("https://x.invalid", "Qwen/Qwen3.8-27B");
    expect(rungs.map((r) => r.effort)).toEqual(["low", "medium", "xhigh"]);
    expect(rungs[2].unavailable_reason).toBe("max_in_flight=8");
  });

  it("returns nothing for a model that is not listed", async () => {
    respond({ data: [{ id: "other/model", reasoning_budget: LADDER }] });
    expect(await fetchReasoningLadder("https://x.invalid", "absent")).toEqual([]);
  });

  it("returns nothing for a model with no ladder published", async () => {
    respond({ data: [{ id: "m" }] });
    expect(await fetchReasoningLadder("https://x.invalid", "m")).toEqual([]);
  });

  describe("degrades quietly rather than surfacing an error", () => {
    // The reasoning control is an enhancement. A chat that works without
    // it beats an error banner about a capability nobody asked for.
    it("on a non-2xx response", async () => {
      respond({}, false);
      expect(await fetchReasoningLadder("https://x.invalid", "m")).toEqual([]);
    });

    it("on a network failure", async () => {
      vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
      expect(await fetchReasoningLadder("https://x.invalid", "m")).toEqual([]);
    });

    it("on a body that is not the expected shape", async () => {
      respond({ unexpected: true });
      expect(await fetchReasoningLadder("https://x.invalid", "m")).toEqual([]);
    });
  });
});
