import type { Resource } from "i18next";

/**
 * Supported language codes for the application.
 *
 * For the foreseeable future we deliberately stay at the language level
 * (e.g. "en", "ru") rather than full locales (e.g. "en-GB") to keep
 * translation overhead manageable.
 *
 * When you add a new language:
 * - Add its code to `SUPPORTED_LANGUAGES`
 * - Add its autonym to `AUTONYM_MAP`
 * - Add its resources to the i18n configuration
 */
export type LanguageCode =
  | "en"
  | "bg"
  | "ca"
  | "cs"
  | "da"
  | "de"
  | "el"
  | "es"
  | "he"
  | "et"
  | "ar"
  | "fa"
  | "fi"
  | "sw"
  | "ha"
  | "am"
  | "yo"
  | "zu"
  | "fr"
  | "ma"
  | "ga"
  | "hr"
  | "hu"
  | "is"
  | "it"
  | "ka"
  | "lt"
  | "lv"
  | "mt"
  | "nl"
  | "no"
  | "pl"
  | "pt"
  | "ro"
  | "ru"
  | "sk"
  | "sl"
  | "sr"
  | "sv"
  | "tr"
  | "uk"
  | "bs"
  | "mk"
  | "kk"
  | "uz"
  | "ig"
  | "om"
  | "so"
  | "ti"
  | "wo";

/**
 * Ordered list of languages enabled in the UI.
 *
 * For now you can keep `SUPPORTED_LANGUAGES` in sync with the
 * actually configured i18n resources (e.g. ["en", "ru"]) and grow
 * it as translations land.
 */
export const SUPPORTED_LANGUAGES: LanguageCode[] = [
  "bg",
  "da",
  "de",
  "el",
  "en",
  "es",
  "et",
  "fi",
  "fr",
  "he",
  "it",
  "ka",
  "kk",
  "nl",
  "no",
  "sv",
  "uz",
  "ar",
  "fa",
  "sw",
  "ha",
  "am",
  "yo",
  "zu",
  "ma",
  "pl",
  "pt",
  "ro",
  "ru",
  "sr",
  "tr",
  "uk",
  // EU-24 completion + Catalan (2026-07-16): every official EU language
  // is now supported.
  "ca",
  "cs",
  "ga",
  "hr",
  "hu",
  "lt",
  "lv",
  "mt",
  "sk",
  "sl",
  // Future Afro‑European / Eurasian candidates; keep out of SUPPORTED_LANGUAGES until translated:
  // "ig", // Igbo
  // "om", // Oromo
  // "so", // Somali
  // "ti", // Tigrinya
  // "wo", // Wolof
  //
  // Asian and Latin-American languages (hi, bn, ur, id, vi, th, fil, ja,
  // ko, ta, es-419, pt-BR, …) are DELIBERATELY not queued here: each
  // launches together with the market narrative (mission copy) for its
  // region, not ahead of it — a language arriving alongside a market
  // story reads as intent; arriving alone it reads as autotranslate.
  // See TRANSLATION_PRIORITY in translation-priority.ts for the policy.
];

/**
 * Autonym map.
 *
 * Each language is named in its own language so that a user only
 * needs to know their own language to find it in the selector.
 */
export const AUTONYM_MAP: Record<LanguageCode, string> = {
  en: "English",
  bg: "български",
  ca: "català",
  cs: "čeština",
  da: "dansk",
  de: "Deutsch",
  el: "Ελληνικά",
  es: "español",
  et: "eesti",
  he: "עברית",
  ar: "العربية",
  fa: "فارسی",
  sw: "Kiswahili",
  ha: "Hausa",
  am: "አማርኛ",
  yo: "Yorùbá",
  zu: "isiZulu",
  ma: "Darija",
  fi: "suomi",
  fr: "français",
  ga: "Gaeilge",
  hr: "hrvatski",
  hu: "magyar",
  is: "íslenska",
  it: "italiano",
  lt: "lietuvių",
  lv: "latviešu",
  mt: "Malti",
  nl: "Nederlands",
  no: "norsk",
  pl: "polski",
  pt: "português",
  ro: "română",
  ru: "русский",
  sk: "slovenčina",
  sl: "slovenščina",
  sr: "српски",
  sv: "svenska",
  tr: "Türkçe",
  uk: "українська",
  bs: "bosanski",
  mk: "македонски",
  ka: "ქართული", // Georgian
  kk: "қазақ тілі", // Kazakh
  uz: "oʻzbekcha", // Uzbek
  ig: "Igbo", // Igbo
  om: "Afaan Oromoo", // Oromo
  so: "Af-Soomaali", // Somali
  ti: "ትግርኛ", // Tigrinya
  wo: "Wolof", // Wolof
};

/**
 * Normalize a full locale (e.g. "en-GB") down to a `LanguageCode`.
 *
 * - Uses the first segment of the locale (before "-")
 * - Falls back to "en" if the language is unsupported or invalid
 */
export const normalizeLocaleToLanguage = (
  locale: string | null | undefined,
): LanguageCode => {
  if (!locale) return "en";
  const lang = locale.split("-")[0]?.toLowerCase() ?? "en";

  if (SUPPORTED_LANGUAGES.includes(lang as LanguageCode)) {
    return lang as LanguageCode;
  }

  return "en";
};

/**
 * Build a stable list of language options for UI components
 * such as dropdowns.
 */
export type LanguageOption = {
  code: LanguageCode;
  autonym: string;
};

export const getLanguageOptions = (): LanguageOption[] =>
  [...SUPPORTED_LANGUAGES]
    .map((code) => ({
      code,
      autonym: AUTONYM_MAP[code],
    }))
    .sort((a, b) => a.autonym.localeCompare(b.autonym));

/**
 * Utility to derive i18next `supportedLngs` from our language codes,
 * so configuration can import from this module instead of hardcoding
 * the list in multiple places.
 */
export const getSupportedLngsForI18Next = (): Resource["en"] extends never
  ? string[]
  : string[] => {
  // i18next accepts string[], while we keep a stricter LanguageCode[]
  return [...SUPPORTED_LANGUAGES];
};

/**
 * Languages whose natural writing direction is right-to-left.
 *
 * This is used by layout code (outside this module) to switch
 * document direction and RTL-aware styling when needed.
 */
export const RTL_LANGUAGES: LanguageCode[] = ["he", "ar", "fa", "ma"];

/**
 * Utility to check whether a given language code is RTL.
 */
export const isRtlLanguage = (code: LanguageCode): boolean =>
  RTL_LANGUAGES.includes(code);
