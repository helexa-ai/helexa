// Records every playground example's live answers (#360).
//
// Skipped unless RECORD_DECISION_EXAMPLES names a service that answers
// /v1/systemone, e.g. a cortex on the mesh:
//
//   RECORD_DECISION_EXAMPLES=http://hanzalova.internal:31313 \
//     npx vitest run src/data/decisionExamples.record.test.ts
//
// It rewrites decisionExamples.recorded.json; commit the result, and read
// the diff — a changed answer is a changed example.
import { writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { EXAMPLES } from "./decisionExamples";
import { RECORDED_PATH, recordingOf, requestFor } from "./decisionExamplesRecording";
import { parseDecisionResponse } from "../lib/decisionClient";

const endpoint = process.env.RECORD_DECISION_EXAMPLES;

describe.skipIf(!endpoint)("record decision examples", () => {
  it("runs every example against the live service", async () => {
    const examples: Record<string, unknown> = {};
    for (const ex of EXAMPLES) {
      const { body, hash } = requestFor(ex);
      const resp = await fetch(`${endpoint!.replace(/\/$/, "")}/v1/systemone`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body,
      });
      expect(resp.status, `${ex.id}: HTTP ${resp.status}`).toBe(200);
      examples[ex.id] = recordingOf(parseDecisionResponse(await resp.text()), hash);
    }
    writeFileSync(
      RECORDED_PATH,
      JSON.stringify({ recordedAt: new Date().toISOString(), examples }, null, 2) + "\n",
    );
  });
});
