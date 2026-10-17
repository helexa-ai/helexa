import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MusicError,
  estimateTokens,
  isTerminal,
  listMusicModels,
  submitJob,
} from "./musicClient";

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("estimateTokens", () => {
  it("multiplies seconds by the rate, rounding up", () => {
    expect(estimateTokens(12.5, 300)).toBe(3750);
    expect(estimateTokens(10.1, 3)).toBe(31);
  });
  it("is undefined when unpriced or the duration is not usable", () => {
    expect(estimateTokens(30, undefined)).toBeUndefined();
    expect(estimateTokens(0, 100)).toBeUndefined();
    expect(estimateTokens(Number.NaN, 100)).toBeUndefined();
  });
});

describe("isTerminal", () => {
  it("knows which states are final", () => {
    expect(isTerminal("succeeded")).toBe(true);
    expect(isTerminal("failed")).toBe(true);
    expect(isTerminal("cancelled")).toBe(true);
    expect(isTerminal("queued")).toBe(false);
    expect(isTerminal("running")).toBe(false);
  });
});

describe("listMusicModels", () => {
  it("keeps only models that advertise audio limits, with their rate", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(200, {
          data: [
            { id: "text-model" },
            {
              id: "music-b",
              audio: { min_duration: 10, max_duration: 600, default_duration: 30, vocals: true, sample_rate: 48000, channels: 2 },
              audio_tokens_per_second: 40,
              notices: [{ text: "Example — Apache-2.0", url: "https://example.org/l" }],
            },
            {
              id: "music-a",
              audio: { min_duration: 1, max_duration: 10, default_duration: 5, vocals: false, sample_rate: 24000, channels: 1 },
            },
          ],
        }),
      ),
    );
    const models = await listMusicModels({ baseUrl: "http://x" });
    expect(models.map((m) => m.id)).toEqual(["music-a", "music-b"]);
    expect(models[1].tokensPerSecond).toBe(40);
    expect(models[0].tokensPerSecond).toBeUndefined();
    expect(models[1].notices).toEqual([{ text: "Example — Apache-2.0", url: "https://example.org/l" }]);
    expect(models[0].notices).toEqual([]);
  });
});

describe("submitJob", () => {
  it("sends only the fields that are set, with the bearer", async () => {
    const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>(async () =>
      jsonResponse(202, { id: "audiogen_1", status: "queued", model: "m", created_at: 1 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const job = await submitJob(
      { model: "m", prompt: "lofi", duration: 30 },
      { baseUrl: "http://x/", apiKey: "k" },
    );
    expect(job.id).toBe("audiogen_1");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://x/v1/audio/generations");
    expect((init!.headers as Record<string, string>).authorization).toBe("Bearer k");
    expect(JSON.parse(init!.body as string)).toEqual({ model: "m", prompt: "lofi", duration: 30 });
  });

  it("surfaces the #63 envelope as a MusicError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse(400, { error: { code: "invalid_audio_params", message: "duration out of range" } }),
      ),
    );
    await expect(submitJob({ model: "m", prompt: "p" }, { baseUrl: "http://x" })).rejects.toMatchObject({
      code: "invalid_audio_params",
      message: "duration out of range",
    });
    await expect(submitJob({ model: "m", prompt: "p" }, { baseUrl: "http://x" })).rejects.toBeInstanceOf(
      MusicError,
    );
  });
});
