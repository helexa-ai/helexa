import i18n, { type Resource } from "i18next";
import { initReactI18next } from "react-i18next";
import {
  SUPPORTED_LANGUAGES,
  normalizeLocaleToLanguage,
  isRtlLanguage,
} from "./languages";
import type { LanguageCode } from "./languages";

// Core languages
import enCommon from "./resources/en/common.json";
import ruCommon from "./resources/ru/common.json";
import enHome from "./resources/en/home.json";
import ruHome from "./resources/ru/home.json";
import enChat from "./resources/en/chat.json";
import ruChat from "./resources/ru/chat.json";

// Scandinavian & Nordic languages
import daCommon from "./resources/da/common.json";
import daHome from "./resources/da/home.json";
import daChat from "./resources/da/chat.json";

import fiCommon from "./resources/fi/common.json";
import fiHome from "./resources/fi/home.json";
import fiChat from "./resources/fi/chat.json";

import noCommon from "./resources/no/common.json";
import noHome from "./resources/no/home.json";
import noChat from "./resources/no/chat.json";

import svCommon from "./resources/sv/common.json";
import svHome from "./resources/sv/home.json";
import svChat from "./resources/sv/chat.json";

import bgCommon from "./resources/bg/common.json";
import bgHome from "./resources/bg/home.json";
import bgChat from "./resources/bg/chat.json";

import etCommon from "./resources/et/common.json";
import etHome from "./resources/et/home.json";
import etChat from "./resources/et/chat.json";

// African & MENA languages
import swCommon from "./resources/sw/common.json";
import swHome from "./resources/sw/home.json";
import swChat from "./resources/sw/chat.json";

import arCommon from "./resources/ar/common.json";
import arHome from "./resources/ar/home.json";
import arChat from "./resources/ar/chat.json";

import faCommon from "./resources/fa/common.json";
import faHome from "./resources/fa/home.json";
import faChat from "./resources/fa/chat.json";

import haCommon from "./resources/ha/common.json";
import haHome from "./resources/ha/home.json";
import haChat from "./resources/ha/chat.json";

import amCommon from "./resources/am/common.json";
import amHome from "./resources/am/home.json";
import amChat from "./resources/am/chat.json";

import yoCommon from "./resources/yo/common.json";
import yoHome from "./resources/yo/home.json";
import yoChat from "./resources/yo/chat.json";

import zuCommon from "./resources/zu/common.json";
import zuHome from "./resources/zu/home.json";
import zuChat from "./resources/zu/chat.json";

// Darija (Moroccan Arabic)
import maCommon from "./resources/ma/common.json";
import maHome from "./resources/ma/home.json";
import maChat from "./resources/ma/chat.json";

// European / other languages
import esCommon from "./resources/es/common.json";
import esHome from "./resources/es/home.json";
import esChat from "./resources/es/chat.json";

import frCommon from "./resources/fr/common.json";
import frHome from "./resources/fr/home.json";
import frChat from "./resources/fr/chat.json";

import deCommon from "./resources/de/common.json";
import deHome from "./resources/de/home.json";
import deChat from "./resources/de/chat.json";

import elCommon from "./resources/el/common.json";
import elHome from "./resources/el/home.json";
import elChat from "./resources/el/chat.json";

import itCommon from "./resources/it/common.json";
import itHome from "./resources/it/home.json";
import itChat from "./resources/it/chat.json";

import heCommon from "./resources/he/common.json";
import heHome from "./resources/he/home.json";
import heChat from "./resources/he/chat.json";

import ptCommon from "./resources/pt/common.json";
import ptHome from "./resources/pt/home.json";
import ptChat from "./resources/pt/chat.json";

import roCommon from "./resources/ro/common.json";
import roHome from "./resources/ro/home.json";
import roChat from "./resources/ro/chat.json";

import kaCommon from "./resources/ka/common.json";
import kaHome from "./resources/ka/home.json";
import kaChat from "./resources/ka/chat.json";

import trCommon from "./resources/tr/common.json";
import trHome from "./resources/tr/home.json";
import trChat from "./resources/tr/chat.json";

import plCommon from "./resources/pl/common.json";
import plHome from "./resources/pl/home.json";
import plChat from "./resources/pl/chat.json";

import ukCommon from "./resources/uk/common.json";
import ukHome from "./resources/uk/home.json";
import ukChat from "./resources/uk/chat.json";

import nlCommon from "./resources/nl/common.json";
import nlHome from "./resources/nl/home.json";
import nlChat from "./resources/nl/chat.json";

import srCommon from "./resources/sr/common.json";
import srHome from "./resources/sr/home.json";
import srChat from "./resources/sr/chat.json";

import kkCommon from "./resources/kk/common.json";
import kkHome from "./resources/kk/home.json";
import kkChat from "./resources/kk/chat.json";

import uzCommon from "./resources/uz/common.json";
import uzHome from "./resources/uz/home.json";
import uzChat from "./resources/uz/chat.json";

/**
 * Application translation resources, split by language and namespace.
 *
 * - `common`: shared UI elements (navigation, theme toggle, etc.)
 * - `home`:   marketing / narrative copy on the landing page
 * - `chat`:   copy for the chat workspace
 */
const resources: Resource = {
  en: {
    common: enCommon,
    home: enHome,
    chat: enChat,
  },
  ru: {
    common: ruCommon,
    home: ruHome,
    chat: ruChat,
  },
  bg: {
    common: bgCommon,
    home: bgHome,
    chat: bgChat,
  },
  da: {
    common: daCommon,
    home: daHome,
    chat: daChat,
  },
  et: {
    common: etCommon,
    home: etHome,
    chat: etChat,
  },
  fi: {
    common: fiCommon,
    home: fiHome,
    chat: fiChat,
  },
  kk: {
    common: kkCommon,
    home: kkHome,
    chat: kkChat,
  },
  uz: {
    common: uzCommon,
    home: uzHome,
    chat: uzChat,
  },

  // African & MENA languages (LTR unless marked RTL via isRtlLanguage)
  sw: {
    common: swCommon,
    home: swHome,
    chat: swChat,
  },
  ar: {
    common: arCommon,
    home: arHome,
    chat: arChat,
  },
  fa: {
    common: faCommon,
    home: faHome,
    chat: faChat,
  },
  ha: {
    common: haCommon,
    home: haHome,
    chat: haChat,
  },
  am: {
    common: amCommon,
    home: amHome,
    chat: amChat,
  },
  yo: {
    common: yoCommon,
    home: yoHome,
    chat: yoChat,
  },
  zu: {
    common: zuCommon,
    home: zuHome,
    chat: zuChat,
  },
  ma: {
    common: maCommon,
    home: maHome,
    chat: maChat,
  },

  // European & other languages
  es: {
    common: esCommon,
    home: esHome,
    chat: esChat,
  },
  fr: {
    common: frCommon,
    home: frHome,
    chat: frChat,
  },
  de: {
    common: deCommon,
    home: deHome,
    chat: deChat,
  },
  el: {
    common: elCommon,
    home: elHome,
    chat: elChat,
  },
  it: {
    common: itCommon,
    home: itHome,
    chat: itChat,
  },
  he: {
    common: heCommon,
    home: heHome,
    chat: heChat,
  },
  pt: {
    common: ptCommon,
    home: ptHome,
    chat: ptChat,
  },
  ro: {
    common: roCommon,
    home: roHome,
    chat: roChat,
  },
  ka: {
    common: kaCommon,
    home: kaHome,
    chat: kaChat,
  },
  tr: {
    common: trCommon,
    home: trHome,
    chat: trChat,
  },
  pl: {
    common: plCommon,
    home: plHome,
    chat: plChat,
  },
  uk: {
    common: ukCommon,
    home: ukHome,
    chat: ukChat,
  },
  nl: {
    common: nlCommon,
    home: nlHome,
    chat: nlChat,
  },
  sr: {
    common: srCommon,
    home: srHome,
    chat: srChat,
  },
  no: {
    common: noCommon,
    home: noHome,
    chat: noChat,
  },
  sv: {
    common: svCommon,
    home: svHome,
    chat: svChat,
  },
};

// Determine initial language from browser, normalised to language-only.
const browserLang: LanguageCode =
  typeof navigator !== "undefined"
    ? normalizeLocaleToLanguage(navigator.language)
    : "en";

// Keep document direction (ltr/rtl) in sync with the active language.
if (typeof document !== "undefined") {
  document.documentElement.dir = isRtlLanguage(browserLang) ? "rtl" : "ltr";
}

/**
 * Initialize i18next with React bindings.
 *
 * This module is imported once in src/main.tsx before any React
 * rendering so that `useTranslation` is ready everywhere.
 */
i18n.use(initReactI18next).init({
  resources,
  lng: browserLang,
  fallbackLng: "en",
  supportedLngs: SUPPORTED_LANGUAGES,
  ns: ["common", "home", "chat"],
  defaultNS: "common",
  // Because we control the keys and interpolate only simple values.
  interpolation: {
    escapeValue: false,
  },
  // For now we stay language-only; we already normalise the browser locale.
  load: "languageOnly",
  // Be explicit about react options for clarity.
  react: {
    useSuspense: false,
  },
});

// Ensure that when the language changes at runtime, document direction
// tracks the new language's natural writing direction.
i18n.on("languageChanged", (lng) => {
  if (typeof document === "undefined") return;
  const lang = normalizeLocaleToLanguage(lng);
  document.documentElement.dir = isRtlLanguage(lang) ? "rtl" : "ltr";
});

export default i18n;
