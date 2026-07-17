// The chat app's application-owned system prompt (#178).
//
// System prompts are an application responsibility, not an operator one —
// the serving chain passes them through untouched (#179). This one gives
// the model its identity, the current date (open-weight models otherwise
// present their training cutoff as "now"), and the UI language.
//
// Kept deliberately short: it rides on every request and eats context on
// 8B-class models. The prompt body stays in English (instruction-following
// is strongest there); only the response-language name is interpolated.

import { AUTONYM_MAP, type LanguageCode } from "../i18n/languages";

export function buildSystemPrompt(
  model: string,
  locale: string,
  withTools = false,
): string {
  const language = AUTONYM_MAP[locale as LanguageCode] ?? "English";
  const date = new Date().toISOString().slice(0, 10);
  const cutoff = withTools
    ? `Today's date is ${date}. Your training data has a cutoff and does not cover ` +
      `recent events (including helexa itself, which postdates it). For questions ` +
      `about current events, recent facts, or anything you are unsure of, use the ` +
      `web_search tool and ground your answer in the results rather than guessing.`
    : `Today's date is ${date}. Your training data has a cutoff and does not cover ` +
      `recent events (including helexa itself, which postdates it). When a question ` +
      `may concern something newer than your training data, say what you know and ` +
      `note that your information may be out of date, rather than guessing.`;
  return [
    `You are ${model}, an open-weight AI model served by helexa (helexa.ai) — ` +
      `sovereign AI infrastructure running near-frontier open models on small, ` +
      `operator-run GPU facilities. You are the assistant of the helexa.ai chat app.`,
    cutoff,
    `Respond in ${language} unless the user writes in, or asks for, another ` +
      `language. Be concise and direct.`,
  ].join("\n\n");
}
