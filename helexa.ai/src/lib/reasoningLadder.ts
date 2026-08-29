/**
 * The reasoning rungs a model currently offers, read from `/v1/models`.
 *
 * Deliberately not a hardcoded `["low","medium","xhigh"]`. The rungs come
 * from the model's own chat template (#290) — `Qwen/Qwen3.8-27B` accepts
 * exactly those three and raises on anything else, but a different model
 * spells its ladder differently, and a list frozen into the SPA rots the
 * first time the catalogue changes. The server already publishes the
 * truth; this reads it.
 *
 * Availability is also the server's to decide. A node running many
 * concurrent slots withholds its deepest rung, and it reports that rather
 * than the SPA inferring it from a `max_in_flight` it would have to be
 * told separately.
 */

/** One rung as `/v1/models` publishes it. */
export interface ReasoningRung {
  effort: string;
  default?: boolean;
  /** Present when the deployment is not currently offering this rung. */
  unavailable_reason?: string;
}

/** What the picker should show for a model, in ladder order. */
export interface ReasoningOptions {
  /** Rungs that can be selected right now. */
  available: ReasoningRung[];
  /** Rungs the deployment is withholding, with its stated reason. */
  withheld: ReasoningRung[];
}

/**
 * Split a model's advertised ladder into offerable and withheld.
 *
 * A model with no ladder — a non-reasoning model, or a cortex too old to
 * publish one — yields nothing, and the caller should offer no reasoning
 * control at all rather than a control that cannot work.
 */
export function reasoningOptions(rungs: readonly ReasoningRung[] | undefined): ReasoningOptions {
  const all = rungs ?? [];
  return {
    available: all.filter((r) => !r.unavailable_reason),
    withheld: all.filter((r) => !!r.unavailable_reason),
  };
}

/**
 * The rung to preselect: the caller's remembered choice when it is still
 * offered, otherwise nothing.
 *
 * Falling back to the model's own `default` would be wrong here. That
 * default is what the model applies when a caller names no effort at all,
 * and on Qwen3.8 it is `xhigh` — the rung most likely to be withheld. A
 * picker that silently promoted a remembered `xhigh` to whatever remains
 * would be choosing on the user's behalf without saying so; better to
 * show the reasoning control as off and let them pick again.
 */
export function resolveSelection(
  remembered: string | null,
  options: ReasoningOptions,
): string | null {
  if (!remembered) return null;
  return options.available.some((r) => r.effort === remembered) ? remembered : null;
}

/**
 * Fetch the ladder for one model.
 *
 * Failure is not an error worth surfacing: the reasoning control is an
 * enhancement, and a chat that works without it is better than an error
 * banner about a capability nobody asked for yet. Returns an empty
 * ladder, which renders as no control.
 */
export async function fetchReasoningLadder(
  baseUrl: string,
  model: string,
  signal?: AbortSignal,
): Promise<ReasoningRung[]> {
  try {
    const resp = await fetch(`${baseUrl.replace(/\/$/, "")}/v1/models`, { signal });
    if (!resp.ok) return [];
    const body: unknown = await resp.json();
    const data = (body as { data?: unknown })?.data;
    if (!Array.isArray(data)) return [];
    const entry = data.find(
      (m): m is { id: string; reasoning_budget?: ReasoningRung[] } =>
        typeof (m as { id?: unknown })?.id === "string" && (m as { id: string }).id === model,
    );
    const rungs = entry?.reasoning_budget;
    return Array.isArray(rungs) ? rungs.filter((r) => typeof r?.effort === "string") : [];
  } catch {
    return [];
  }
}
