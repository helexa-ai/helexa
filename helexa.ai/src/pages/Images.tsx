import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useLiveQuery } from "dexie-react-hooks";
import { FaArrowUp, FaStop, FaDownload, FaTrash } from "react-icons/fa6";
import { useAuth } from "../auth/context";
import { CHAT_API_KEY, db } from "../data/db";
import { deleteImage, listImages, saveImage } from "../data/repositories";
import { ensureChatKey } from "../lib/ensureChatKey";
import { ImageError, generateImage } from "../lib/imageClient";

const IMAGE_MODEL = import.meta.env.VITE_IMAGE_MODEL || "helexa/image";

/**
 * Sizes offered, grouped by orientation.
 *
 * The engine validates width and height independently — both must be
 * multiples of 16 and within the host's ceiling — so non-square costs
 * nothing extra to support. Every option here stays within 1024, which
 * every image-capable host in the fleet can serve; the real ceiling is
 * per host and higher on the bigger cards, but `/v1/models` publishes
 * capabilities and not a resolution limit, so the UI cannot discover it.
 *
 * Ratios rather than arbitrary numbers: 1:1, 3:4 and 9:16 portrait, and
 * their landscape mirrors. Anything not a multiple of 16 is rejected
 * outright, so these are not free-form.
 */
const SIZE_GROUPS = [
  { key: "square" as const, sizes: [[512, 512], [768, 768], [1024, 1024]] },
  { key: "portrait" as const, sizes: [[768, 1024], [576, 1024]] },
  { key: "landscape" as const, sizes: [[1024, 768], [1024, 576]] },
];

/** Turbo models are tuned for very few steps; more is rarely better. */
const STEP_CHOICES = [4, 9, 16] as const;
const DEFAULT_STEPS = 9;

/**
 * `/images` — generation as its own surface, beside the chat rather than
 * inside it.
 *
 * Deliberately not a chat mode: the parameters are different, a
 * generation takes tens of seconds with no token stream to watch, and
 * the output is a file people want to keep rather than a turn in a
 * conversation.
 *
 * Everything stays in this browser, like chat history — but images are
 * kept more carefully, because one costs real GPU time and cannot be
 * reproduced without its seed.
 */
export default function Images() {
  const { t } = useTranslation(["images", "common"]);
  const { status, accountId, token } = useAuth();
  const authed = status === "authed" && !!accountId;
  const owner = authed ? accountId! : "anon";

  const apiKey = useLiveQuery<string | null, undefined>(
    async () => {
      const m = await db.meta.get(CHAT_API_KEY);
      return typeof m?.value === "string" ? m.value : null;
    },
    [],
    undefined,
  );

  // Same on-demand provisioning as the chat: a signed-in browser should
  // never be asked to go and mint a key by hand.
  useEffect(() => {
    if (!authed || !token || apiKey !== null) return;
    void ensureChatKey(token);
  }, [authed, token, apiKey]);

  const images = useLiveQuery(() => listImages(owner), [owner], []);

  const [prompt, setPrompt] = useState("");
  // Serialised as "WxH" so it can be a plain <select> value; the engine
  // takes the two axes independently.
  const [size, setSize] = useState("1024x1024");
  const [width, height] = size.split("x").map(Number);
  const [steps, setSteps] = useState<number>(DEFAULT_STEPS);
  const [seed, setSeed] = useState("");
  const [negative, setNegative] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // A generation gives no progress signal at all until it completes, and
  // a cold model load alone is ~10s. A ticking counter is the honest
  // substitute — a spinner that sits still for half a minute reads as a
  // hang, which is exactly when people reload and pay for it twice.
  // `startedAt` is stamped by run() rather than here: resetting the
  // counter inside the effect would be a synchronous setState on every
  // start, which cascades renders.
  const startedAt = useRef(0);
  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setElapsed(Date.now() - startedAt.current), 200);
    return () => clearInterval(id);
  }, [busy]);

  async function run(): Promise<void> {
    const text = prompt.trim();
    if (!text || busy) return;
    setError(null);
    startedAt.current = Date.now();
    setElapsed(0);
    setBusy(true);
    const ctl = new AbortController();
    abort.current = ctl;
    const parsedSeed = seed.trim() === "" ? undefined : Number(seed.trim());
    try {
      const result = await generateImage({
        apiKey: apiKey ?? undefined,
        model: IMAGE_MODEL,
        prompt: text,
        width,
        height,
        steps,
        seed: Number.isFinite(parsedSeed) ? parsedSeed : undefined,
        negativePrompt: negative.trim() || undefined,
        signal: ctl.signal,
      });
      // Decode once, here: the gallery renders from object URLs, and
      // keeping base64 in IndexedDB would cost a third more space.
      const bytes = Uint8Array.from(atob(result.b64), (c) => c.charCodeAt(0));
      await saveImage({
        owner,
        prompt: text,
        negativePrompt: negative.trim() || undefined,
        model: IMAGE_MODEL,
        width,
        height,
        steps,
        seed: Number.isFinite(parsedSeed) ? parsedSeed : undefined,
        png: new Blob([bytes], { type: "image/png" }),
        units: result.units,
      });
    } catch (e) {
      if (e instanceof ImageError && e.code === "cancelled") {
        /* the user asked; not an error */
      } else {
        setError(e instanceof ImageError ? e.message : t("images:errorGeneric"));
      }
    } finally {
      abort.current = null;
      setBusy(false);
    }
  }

  const canGenerate = !!prompt.trim() && !busy && authed && apiKey !== null;

  return (
    <main className="app-main container py-4 hx-images">
      <h1 className="h4 mb-1">{t("images:title")}</h1>
      <p className="text-muted small">{t("images:lead")}</p>

      {/* Visible to everyone, generated only by account holders. The
          anonymous tier is capped in messages, and one image costs
          orders of magnitude more GPU time than one message — but
          hiding the page entirely would leave image generation exactly
          as undiscoverable as it was, which is the problem this route
          exists to solve. So: show the offer, ask for an account. */}
      {!authed ? (
        <div className="hx-image-signin">
          <p>{t("images:signInPrompt")}</p>
          <Link to="/auth?tab=signup" className="hx-btn-primary">
            {t("common:nav.register")}
          </Link>
        </div>
      ) : (
      <form
        className="hx-image-form"
        onSubmit={(e) => {
          e.preventDefault();
          void run();
        }}
      >
        <textarea
          rows={3}
          value={prompt}
          disabled={busy}
          placeholder={t("images:promptPlaceholder")}
          aria-label={t("images:promptPlaceholder")}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void run();
            }
          }}
        />

        <div className="hx-image-controls">
          <label>
            <span>{t("images:size")}</span>
            <select
              value={size}
              disabled={busy}
              onChange={(e) => setSize(e.target.value)}
            >
              {/* optgroup rather than disabled separator options: it
                  is the element a screen reader announces as a group,
                  where a disabled option is announced as an option you
                  cannot pick. */}
              {SIZE_GROUPS.map((g) => (
                <optgroup key={g.key} label={t(`images:orientation.${g.key}`)}>
                  {g.sizes.map(([w, h]) => (
                    // Dimensions are numerals in every locale, and
                    // `dir=ltr` keeps "1024×576" from being reordered
                    // under RTL — which would silently swap the
                    // orientation the option is offering.
                    <option key={`${w}x${h}`} value={`${w}x${h}`} dir="ltr">
                      {`${w}×${h}`}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>

          <label>
            <span>{t("images:steps")}</span>
            <select
              value={steps}
              disabled={busy}
              onChange={(e) => setSteps(Number(e.target.value))}
            >
              {STEP_CHOICES.map((s) => (
                <option key={s} value={s} dir="ltr">
                  {s}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="hx-image-advanced"
            aria-expanded={advanced}
            onClick={() => setAdvanced((v) => !v)}
          >
            {t("images:advanced")}
          </button>

          <div className="hx-image-actions">
            {busy ? (
              <button
                type="button"
                className="hx-btn-ghost"
                onClick={() => abort.current?.abort()}
              >
                <FaStop size={13} /> {t("images:cancel")}
              </button>
            ) : null}
            <button type="submit" className="hx-btn-primary" disabled={!canGenerate}>
              <FaArrowUp size={13} /> {t("images:generate")}
            </button>
          </div>
        </div>

        {advanced ? (
          <div className="hx-image-advanced-panel">
            <label>
              <span>{t("images:seed")}</span>
              <input
                type="text"
                inputMode="numeric"
                dir="ltr"
                value={seed}
                disabled={busy}
                placeholder={t("images:seedRandom")}
                onChange={(e) => setSeed(e.target.value)}
              />
              <small>{t("images:seedHint")}</small>
            </label>
            <label>
              <span>{t("images:negativePrompt")}</span>
              <input
                type="text"
                value={negative}
                disabled={busy}
                onChange={(e) => setNegative(e.target.value)}
              />
              {/* Not a footnote: turning this on doubles the work per
                  step, so it has to be said where the choice is made. */}
              <small className="hx-warn">{t("images:negativePromptHint")}</small>
            </label>
          </div>
        ) : null}
      </form>
      )}

      {authed && apiKey === null ? (
        <p className="text-muted small mt-2">{t("images:preparing")}</p>
      ) : null}

      {busy ? (
        <p className="hx-image-progress" aria-live="polite">
          {/* Bare seconds rather than a phrase with a counted noun —
              plural and case agreement varies across the shipped
              locales, and a numeral needs no grammar. */}
          {t("images:generating")} <span dir="ltr">{(elapsed / 1000).toFixed(1)}s</span>
          {elapsed > 12_000 ? (
            <span className="d-block text-muted">{t("images:coldStart")}</span>
          ) : null}
        </p>
      ) : null}

      {error ? (
        <div className="alert alert-warning mt-3" role="alert">
          {error}
        </div>
      ) : null}

      <section className="hx-image-gallery">
        {(images ?? []).length === 0 ? (
          <p className="text-muted small hx-image-empty">{t("images:galleryEmpty")}</p>
        ) : (
          (images ?? []).map((img) => (
            <ImageCard key={img.id} id={img.id} />
          ))
        )}
      </section>
    </main>
  );
}

/**
 * One stored image. The object URL is created per card and revoked on
 * unmount — a gallery that leaks them holds every PNG it has ever shown
 * in memory for the life of the tab.
 */
function ImageCard({ id }: { id: string }): React.ReactElement | null {
  const { t } = useTranslation(["images", "common"]);
  const image = useLiveQuery(() => db.images.get(id), [id]);
  // Derived rather than state: creating the URL in an effect would mean a
  // synchronous setState, and the render that follows would show a blank
  // card for one frame. The cleanup still runs on unmount and whenever the
  // blob changes, so nothing leaks — a gallery that keeps its object URLs
  // pins every PNG it has ever displayed in memory for the life of the tab.
  // Bound first so the memo depends on the blob itself. Depending on
  // `image?.png` makes the compiler infer the whole `image` record and
  // bail out of optimising the component.
  const png = image?.png;
  const url = useMemo(() => (png ? URL.createObjectURL(png) : null), [png]);
  useEffect(() => {
    if (!url) return;
    return () => URL.revokeObjectURL(url);
  }, [url]);

  if (!image || !url) return null;

  return (
    <figure className="hx-image-card">
      <img src={url} alt={image.prompt} loading="lazy" />
      <figcaption>
        <p className="hx-image-prompt">{image.prompt}</p>
        <div className="hx-image-meta" dir="ltr">
          {image.width}×{image.height}
          {image.seed !== undefined ? ` · seed ${image.seed}` : ""}
        </div>
        <div className="hx-image-card-actions">
          <a
            className="hx-btn-ghost"
            href={url}
            download={`helexa-${image.id.slice(0, 8)}.png`}
          >
            <FaDownload size={12} /> {t("images:download")}
          </a>
          <button
            type="button"
            className="hx-btn-ghost"
            onClick={() => void deleteImage(image.id)}
            aria-label={t("images:delete")}
            title={t("images:delete")}
          >
            <FaTrash size={12} />
          </button>
        </div>
      </figcaption>
    </figure>
  );
}
