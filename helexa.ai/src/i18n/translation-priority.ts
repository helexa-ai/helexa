/**
 * Translation priority metadata for all languages in `LanguageCode`.
 *
 * This is a guidance list for contributors and maintainers when deciding
 * which language to prioritise for new or improved translations.
 *
 * It is intentionally approximate and based on:
 * - Rough, order‑of‑magnitude estimates of native speaker counts
 * - A bias toward potential audience size
 * - Some awareness of regional groupings (e.g. South Slavic languages)
 *
 * IMPORTANT:
 * - This is NOT an authoritative demographic source.
 * - Numbers are approximate and occasionally rounded for clarity.
 * - You SHOULD adjust this based on:
 *   - Real user/operator demand
 *   - Strategic goals (e.g. mesh operator regions)
 *   - Availability and reliability of translators
 */

import {
  AUTONYM_MAP,
  SUPPORTED_LANGUAGES,
  type LanguageCode,
  type LanguageOption,
} from "./languages";

/**
 * Languages that are defined in `LanguageCode` but are not yet
 * in `SUPPORTED_LANGUAGES`.
 *
 * This is a human‑maintained checklist of “not yet fully supported”
 * languages for the UI. Once a language is added to SUPPORTED_LANGUAGES
 * and wired into the i18n resources, remove it from this list.
 */
export const REMAINING_LANGUAGES: LanguageCode[] = [
  "hu",
  "cs",
  "hr",
  "sk",
  "sl",
  "lt",
  "lv",
  "bs",
  "mk",
  "ga",
  "is",
  "mt",
];

/**
 * Rough qualitative buckets for prioritisation.
 *
 * - "high":   large potential audience; strong candidate for early support
 * - "medium": mid‑sized audience; often follows the high tier
 * - "low":    smaller audience but potentially important for coverage,
 *             policy, or symbolic reasons
 */
export type TranslationPriorityBucket = "high" | "medium" | "low";

/**
 * A rough, order‑of‑magnitude estimate of native speakers.
 *
 * Values are intentionally coarse; they’re meant for orientation,
 * not for demographic precision.
 */
export type NativeSpeakerEstimate =
  | "70–90M"
  | "40–50M"
  | "30–40M"
  | "20–30M"
  | "10–20M"
  | "5–10M"
  | "2–5M"
  | "1–2M"
  | "<1M";

/**
 * Coarse-grained translation status for a language.
 *
 * - "complete":  common/home/chat resources are present and wired,
 *                and `lang.*` labels exist in English
 * - "partial":   some resources or labels are present, but work remains
 * - "missing":   planned language with no concrete resources yet
 */
export type TranslationStatus = "complete" | "partial" | "missing";

/**
 * Metadata for a single language in the translation roadmap.
 */
export interface TranslationPriorityEntry {
  code: LanguageCode;
  /**
   * Rough priority tier based on potential audience.
   */
  bucket: TranslationPriorityBucket;
  /**
   * Very rough estimate of native speakers, for human orientation only.
   */
  nativeSpeakers: NativeSpeakerEstimate;
  /**
   * Current translation status for this language in the codebase.
   */
  status: TranslationStatus;
  /**
   * Free‑form notes for maintainers (e.g. “strong regional cluster with hr/bs/sr”).
   */
  notes?: string;
}

/**
 * Ordered list of languages, roughly sorted by:
 *  1) estimated native speaker count
 *  2) regional grouping / ecosystem considerations
 *
 * This is a *starting point* — reorder based on real
 * product needs and contributor interest.
 *
 * Includes all `LanguageCode` values, regardless of whether
 * resources already exist in the repo.
 */
export const TRANSLATION_PRIORITY: TranslationPriorityEntry[] = [
  // High‑priority by estimated native speakers
  {
    code: "en",
    bucket: "high",
    nativeSpeakers: "40–50M",
    status: "complete",
    notes:
      "English (native speakers concentrated in a few countries; far larger as a second language). Already the primary UI language and global lingua franca.",
  },
  {
    code: "ru",
    bucket: "high",
    nativeSpeakers: "70–90M",
    status: "complete",
    notes:
      "Russian; large regional audience across Eastern Europe and Central Asia. Already supported in the UI.",
  },
  {
    code: "de",
    bucket: "high",
    nativeSpeakers: "70–90M",
    status: "complete",
    notes:
      "German; large European language with strong industry, research, and operator presence. Already supported.",
  },
  {
    code: "fr",
    bucket: "high",
    nativeSpeakers: "70–90M",
    status: "complete",
    notes:
      "French; major European and global language, important for broader reach and EU coverage. Already supported.",
  },
  {
    code: "tr",
    bucket: "high",
    nativeSpeakers: "70–90M",
    status: "complete",
    notes:
      "Turkish; large native base in Türkiye plus diaspora; strong early win for reach. Already supported.",
  },
  {
    code: "pl",
    bucket: "high",
    nativeSpeakers: "40–50M",
    status: "complete",
    notes:
      "Polish; substantial EU audience and strong developer/infra ecosystem. Already supported.",
  },
  {
    code: "it",
    bucket: "high",
    nativeSpeakers: "40–50M",
    status: "complete",
    notes:
      "Italian; sizable EU language with active developer and research communities. Already supported.",
  },
  {
    code: "es",
    bucket: "high",
    nativeSpeakers: "30–40M",
    status: "complete",
    notes:
      "Spanish (European focus here); global reach is much higher when including Latin America. Already supported.",
  },
  {
    code: "uk",
    bucket: "high",
    nativeSpeakers: "30–40M",
    status: "complete",
    notes:
      "Ukrainian; significant and currently very visible digital community. Already supported.",
  },
  {
    code: "pt",
    bucket: "high",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Portuguese (European focus here); global footprint expands strongly with Brazil and Lusophone Africa. Already supported.",
  },
  {
    code: "nl",
    bucket: "high",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Dutch; Netherlands, Flanders and beyond. High English proficiency but strategically relevant. Already supported.",
  },
  {
    code: "ro",
    bucket: "high",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "Romanian; ties into Eastern European and Balkan ecosystems. Already supported.",
  },
  {
    code: "hu",
    bucket: "high",
    nativeSpeakers: "10–20M",
    status: "missing",
    notes:
      "Hungarian; Central European hub language with distinct linguistic profile. Planned for future translation.",
  },
  {
    code: "ka",
    bucket: "medium",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Georgian; key language in the Caucasus region with an active tech and OSS community, relevant for operators and users near Eastern Europe and Western Asia. Planned for future translation.",
  },
  {
    code: "kk",
    bucket: "medium",
    nativeSpeakers: "10–20M",
    status: "missing",
    notes:
      "Kazakh; widely spoken in Kazakhstan and Central Asia, with growing digital infrastructure and proximity to Eastern European and Eurasian compute hubs. Planned for future translation.",
  },
  {
    code: "uz",
    bucket: "medium",
    nativeSpeakers: "20–30M",
    status: "missing",
    notes:
      "Uzbek; major Central Asian language with significant urban and tech-savvy populations, relevant for Eurasian and Afro‑European compute corridors. Planned for future translation.",
  },
  {
    code: "sw",
    bucket: "high",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "Swahili (Kiswahili); major lingua franca across East and Central Africa, with growing digital and developer ecosystems, including diasporas in Europe. Already supported.",
  },
  {
    code: "sw",
    bucket: "high",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "Swahili (Kiswahili); major lingua franca across East and Central Africa, with growing digital and developer ecosystems, including diasporas in Europe. Already supported.",
  },
  {
    code: "sr",
    bucket: "high",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "Serbian; part of South Slavic cluster (hr/bs/sr); improves coverage for the wider region. Already supported.",
  },

  // Caucasus / Central Asian languages relevant to Afro‑European infrastructure
  {
    code: "ka",
    bucket: "medium",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Georgian; key language in the Caucasus region with an active tech and OSS community, relevant for operators and users near Eastern Europe and Western Asia. Planned for future translation.",
  },
  {
    code: "kk",
    bucket: "medium",
    nativeSpeakers: "10–20M",
    status: "missing",
    notes:
      "Kazakh; widely spoken in Kazakhstan and Central Asia, with growing digital infrastructure and proximity to Eastern European and Eurasian compute hubs. Planned for future translation.",
  },
  {
    code: "uz",
    bucket: "medium",
    nativeSpeakers: "20–30M",
    status: "missing",
    notes:
      "Uzbek; major Central Asian language with significant urban and tech-savvy populations, relevant for Eurasian and Afro‑European compute corridors. Planned for future translation.",
  },

  // Additional African languages (planned)
  {
    code: "ig",
    bucket: "medium",
    nativeSpeakers: "20–30M",
    status: "missing",
    notes:
      "Igbo; major language in Nigeria with a large, increasingly digital and diaspora population, including in Europe. Planned for future translation.",
  },
  {
    code: "om",
    bucket: "medium",
    nativeSpeakers: "20–30M",
    status: "missing",
    notes:
      "Oromo; widely spoken in Ethiopia and Kenya, with growing online and tech presence. Planned for future translation.",
  },
  {
    code: "so",
    bucket: "medium",
    nativeSpeakers: "10–20M",
    status: "missing",
    notes:
      "Somali; spoken in the Horn of Africa with a notable diaspora in Europe and active digital communities. Planned for future translation.",
  },
  {
    code: "ti",
    bucket: "low",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Tigrinya; important language in Eritrea and northern Ethiopia, with diaspora communities using digital services. Planned for future translation.",
  },
  {
    code: "wo",
    bucket: "low",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Wolof; major language in Senegal and neighbouring countries, with visible online culture and some European diaspora. Planned for future translation.",
  },

  // Medium‑priority block (mid‑sized audiences and/or strong regional value)
  {
    code: "cs",
    bucket: "medium",
    nativeSpeakers: "10–20M",
    status: "missing",
    notes:
      "Czech; strong tech and OSS presence, mid‑sized audience. Planned future translation.",
  },
  {
    code: "bg",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "complete",
    notes:
      "Bulgarian; already supported, relevant for South‑East European operator communities.",
  },
  {
    code: "el",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "complete",
    notes:
      "Greek; already supported; important for Eastern Mediterranean operators and communities.",
  },
  {
    code: "da",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Danish; Denmark, Faroe Islands. Smallish audience but high digital readiness. Planned for future translation.",
  },
  {
    code: "fi",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Finnish; strong tech ecosystem, often prioritised despite high English proficiency. Planned for future translation.",
  },
  {
    code: "no",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Norwegian; Bokmål/Nynorsk split but typically Bokmål for UI. Similar profile to Danish/Swedish. Planned for future translation.",
  },
  {
    code: "sv",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Swedish; Sweden + parts of Finland. High English proficiency, but useful for regional completeness. Planned for future translation.",
  },
  {
    code: "hr",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Croatian; closely related to Serbian/Bosnian; may share some translations with sr/bs. Planned for future translation.",
  },
  {
    code: "sk",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "missing",
    notes:
      "Slovak; complements existing Czech/Polish coverage in the region. Planned for future translation.",
  },
  {
    code: "sl",
    bucket: "medium",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Slovene; completes a lot of the Central European mesh. Planned for future translation.",
  },
  {
    code: "et",
    bucket: "medium",
    nativeSpeakers: "1–2M",
    status: "complete",
    notes:
      "Estonian; already supported; important for Baltic / Northern European operator ecosystems.",
  },

  // Lower‑priority by raw native speaker count (coverage/policy still important)
  {
    code: "lt",
    bucket: "low",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Lithuanian; Baltic language, EU official; good for EU completeness.",
  },
  {
    code: "ha",
    bucket: "low",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Hausa; widely spoken in West Africa and within diaspora communities, including Europe. Already supported.",
  },
  {
    code: "yo",
    bucket: "low",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Yorùbá; major West African language with a significant diaspora, including in Europe. Already supported.",
  },
  {
    code: "et",
    bucket: "medium",
    nativeSpeakers: "1–2M",
    status: "complete",
    notes:
      "Estonian; already supported; important for Baltic / Northern European operator ecosystems.",
  },

  // Lower‑priority by raw native speaker count (coverage/policy still important)
  {
    code: "lt",
    bucket: "low",
    nativeSpeakers: "2–5M",
    status: "missing",
    notes:
      "Lithuanian; Baltic language, EU official; good for EU completeness.",
  },
  {
    code: "lv",
    bucket: "low",
    nativeSpeakers: "1–2M",
    status: "missing",
    notes:
      "Latvian; Baltic language, EU official; pairs naturally with Lithuanian.",
  },
  {
    code: "bs",
    bucket: "low",
    nativeSpeakers: "1–2M",
    status: "missing",
    notes:
      "Bosnian; part of the Serbo‑Croatian continuum; benefits from work on sr/hr. Planned for future translation.",
  },
  {
    code: "mk",
    bucket: "low",
    nativeSpeakers: "1–2M",
    status: "missing",
    notes:
      "Macedonian; South Slavic, close to Bulgarian; regional mesh coverage.",
  },
  {
    code: "ga",
    bucket: "low",
    nativeSpeakers: "<1M",
    status: "missing",
    notes:
      "Irish (Gaeilge); relatively small fully fluent base but high cultural and policy importance.",
  },
  {
    code: "is",
    bucket: "low",
    nativeSpeakers: "<1M",
    status: "missing",
    notes:
      "Icelandic; small absolute numbers but linguistically distinct; often prioritised for diversity.",
  },
  {
    code: "mt",
    bucket: "low",
    nativeSpeakers: "<1M",
    status: "missing",
    notes:
      "Maltese; EU official language with a compact but important speaker base.",
  },
  {
    code: "ar",
    bucket: "high",
    nativeSpeakers: "70–90M",
    status: "complete",
    notes:
      "Arabic; large native base across MENA and significant diaspora communities in Europe. Already supported and RTL.",
  },
  {
    code: "fa",
    bucket: "medium",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Persian (Farsi); widely spoken in Iran and neighbouring countries with a substantial diaspora in Europe. Already supported and RTL.",
  },
  {
    code: "he",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "complete",
    notes:
      "Hebrew; key language for Israel with an active tech and AI ecosystem, plus diaspora in Europe. Already supported and RTL.",
  },
  {
    code: "ha",
    bucket: "low",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Hausa; widely spoken in West Africa and within diaspora communities, including Europe. Already supported.",
  },
  {
    code: "yo",
    bucket: "low",
    nativeSpeakers: "20–30M",
    status: "complete",
    notes:
      "Yorùbá; major West African language with a significant diaspora, including in Europe. Already supported.",
  },
  {
    code: "zu",
    bucket: "low",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "isiZulu; one of South Africa’s largest languages, important for Southern African operator and user communities. Already supported.",
  },
  {
    code: "am",
    bucket: "low",
    nativeSpeakers: "10–20M",
    status: "complete",
    notes:
      "Amharic; widely spoken in Ethiopia with diaspora communities in Europe and elsewhere. Already supported.",
  },
  {
    code: "ma",
    bucket: "medium",
    nativeSpeakers: "5–10M",
    status: "complete",
    notes:
      "Darija (Moroccan Arabic); important for North African users and operators, with strong links to Arabic/French ecosystems and diaspora communities in Europe. Already supported and treated as RTL.",
  },
];

/**
 * Convenience helper: returns the priority entry for a given language code,
 * if one exists.
 */
export const getTranslationPriorityFor = (
  code: LanguageCode,
): TranslationPriorityEntry | undefined =>
  TRANSLATION_PRIORITY.find((entry) => entry.code === code);

/**
 * Convenience helper: returns remaining languages grouped by priority bucket.
 */
export const getLanguagesByPriorityBucket = (): Record<
  TranslationPriorityBucket,
  LanguageCode[]
> => {
  const result: Record<TranslationPriorityBucket, LanguageCode[]> = {
    high: [],
    medium: [],
    low: [],
  };

  for (const { code, bucket } of TRANSLATION_PRIORITY) {
    result[bucket].push(code);
  }

  return result;
};

/**
 * Language options ordered by estimated usage (the TRANSLATION_PRIORITY
 * ranking — roughly native-speaker count), NOT alphabetically. This is the
 * deliberate marketing choice for the language selector: it foregrounds
 * helexa's international grounding and weights real language usage over a
 * silicon-valley "everyone learns American" default.
 *
 * Each supported language is ranked by its first appearance in
 * TRANSLATION_PRIORITY (the list has a few duplicate entries — first wins).
 * Any supported language missing from the priority list is appended
 * (alphabetically by autonym) so none are ever dropped from the picker.
 */
export const getLanguageOptionsByUsage = (): LanguageOption[] => {
  const rank = new Map<LanguageCode, number>();
  TRANSLATION_PRIORITY.forEach((entry, i) => {
    if (!rank.has(entry.code)) rank.set(entry.code, i);
  });
  const ranked = [...SUPPORTED_LANGUAGES]
    .filter((c) => rank.has(c))
    .sort((a, b) => rank.get(a)! - rank.get(b)!);
  const unranked = [...SUPPORTED_LANGUAGES]
    .filter((c) => !rank.has(c))
    .sort((a, b) => AUTONYM_MAP[a].localeCompare(AUTONYM_MAP[b]));
  return [...ranked, ...unranked].map((code) => ({
    code,
    autonym: AUTONYM_MAP[code],
  }));
};
