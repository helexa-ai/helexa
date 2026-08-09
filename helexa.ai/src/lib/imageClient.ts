// Image generation client → the mesh router's OpenAI-compatible
// /v1/images/generations. Unlike chat there is no stream: the response
// arrives whole, so the only progress signal the UI has is elapsed time.

export interface ImageRequest {
  baseUrl?: string;
  apiKey?: string;
  model: string;
  prompt: string;
  /** Square edge in pixels; sent as OpenAI's `size` ("1024x1024"). */
  size: number;
  /** Fixed seed makes a prompt change comparable. Omitted = server picks. */
  seed?: number;
  steps?: number;
  /** Enabling this turns on CFG, which doubles time and cost per step. */
  negativePrompt?: string;
  guidanceScale?: number;
  signal: AbortSignal;
}

export interface ImageTiming {
  encode_ms?: number;
  denoise_ms?: number;
  decode_ms?: number;
  steps?: number;
  cfg?: boolean;
}

export interface ImageResult {
  /** Raw base64 PNG, without a data: prefix. */
  b64: string;
  /** Megapixel-steps billed for this generation (#202). */
  units?: number;
  timing?: ImageTiming;
}

export class ImageError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.name = "ImageError";
    this.code = code;
  }
}

const DEFAULT_BASE = import.meta.env.VITE_ROUTER_BASE_URL || "";

/**
 * A generation can legitimately take minutes: a cold model load is ~10s
 * before denoising even starts, and a large size with CFG multiplies the
 * work. This bound exists only to stop a dead origin hanging the UI
 * forever, so it is far above any real generation.
 */
const TIMEOUT_MS = 300_000;

export async function generateImage(req: ImageRequest): Promise<ImageResult> {
  const base = (req.baseUrl ?? DEFAULT_BASE).replace(/\/$/, "");
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (req.apiKey) headers.authorization = `Bearer ${req.apiKey}`;

  // Chain the caller's cancel signal with a timeout, and remember which
  // one fired so a user-initiated cancel is not reported as a fault.
  let timedOut = false;
  const ctl = new AbortController();
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, TIMEOUT_MS);
  const onAbort = (): void => ctl.abort();
  req.signal.addEventListener("abort", onAbort);

  try {
    const resp = await fetch(`${base}/v1/images/generations`, {
      method: "POST",
      headers,
      signal: ctl.signal,
      body: JSON.stringify({
        model: req.model,
        prompt: req.prompt,
        n: 1,
        size: `${req.size}x${req.size}`,
        response_format: "b64_json",
        ...(req.seed !== undefined ? { seed: req.seed } : {}),
        ...(req.steps !== undefined ? { num_steps: req.steps } : {}),
        ...(req.negativePrompt ? { negative_prompt: req.negativePrompt } : {}),
        ...(req.guidanceScale !== undefined
          ? { guidance_scale: req.guidanceScale }
          : {}),
      }),
    });

    if (!resp.ok) {
      // #63 error envelope: { error: { code, message, ... } }. Fall back to
      // the status when a proxy returns something that isn't ours.
      let code = `http_${resp.status}`;
      let message = `Request failed (${resp.status})`;
      try {
        const body = (await resp.json()) as {
          error?: { code?: string; message?: string };
        };
        if (body.error?.code) code = body.error.code;
        if (body.error?.message) message = body.error.message;
      } catch {
        /* keep the status-derived fallback */
      }
      throw new ImageError(code, message);
    }

    const body = (await resp.json()) as {
      data?: { b64_json?: string }[];
      usage?: { helexa_image_units?: number; helexa_timing?: ImageTiming };
    };
    const b64 = body.data?.[0]?.b64_json;
    if (!b64) throw new ImageError("empty_response", "No image was returned");

    return {
      b64,
      units: body.usage?.helexa_image_units,
      timing: body.usage?.helexa_timing,
    };
  } catch (e) {
    if (e instanceof ImageError) throw e;
    if ((e as Error)?.name === "AbortError") {
      throw new ImageError(
        timedOut ? "timeout" : "cancelled",
        timedOut ? "The request timed out" : "Cancelled",
      );
    }
    throw new ImageError("network_error", (e as Error)?.message ?? "Network error");
  } finally {
    clearTimeout(timer);
    req.signal.removeEventListener("abort", onAbort);
  }
}
