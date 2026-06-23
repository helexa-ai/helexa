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
import enMission from "./resources/en/mission.json";
import ruMission from "./resources/ru/mission.json";
import enChat from "./resources/en/chat.json";
import ruChat from "./resources/ru/chat.json";

// Scandinavian & Nordic languages
import daCommon from "./resources/da/common.json";
import daMission from "./resources/da/mission.json";
import daChat from "./resources/da/chat.json";

import fiCommon from "./resources/fi/common.json";
import fiMission from "./resources/fi/mission.json";
import fiChat from "./resources/fi/chat.json";

import noCommon from "./resources/no/common.json";
import noMission from "./resources/no/mission.json";
import noChat from "./resources/no/chat.json";

import svCommon from "./resources/sv/common.json";
import svMission from "./resources/sv/mission.json";
import svChat from "./resources/sv/chat.json";

import bgCommon from "./resources/bg/common.json";
import bgMission from "./resources/bg/mission.json";
import bgChat from "./resources/bg/chat.json";

import etCommon from "./resources/et/common.json";
import etMission from "./resources/et/mission.json";
import etChat from "./resources/et/chat.json";

// African & MENA languages
import swCommon from "./resources/sw/common.json";
import swMission from "./resources/sw/mission.json";
import swChat from "./resources/sw/chat.json";

import arCommon from "./resources/ar/common.json";
import arMission from "./resources/ar/mission.json";
import arChat from "./resources/ar/chat.json";

import faCommon from "./resources/fa/common.json";
import faMission from "./resources/fa/mission.json";
import faChat from "./resources/fa/chat.json";

import haCommon from "./resources/ha/common.json";
import haMission from "./resources/ha/mission.json";
import haChat from "./resources/ha/chat.json";

import amCommon from "./resources/am/common.json";
import amMission from "./resources/am/mission.json";
import amChat from "./resources/am/chat.json";

import yoCommon from "./resources/yo/common.json";
import yoMission from "./resources/yo/mission.json";
import yoChat from "./resources/yo/chat.json";

import zuCommon from "./resources/zu/common.json";
import zuMission from "./resources/zu/mission.json";
import zuChat from "./resources/zu/chat.json";

// Darija (Moroccan Arabic)
import maCommon from "./resources/ma/common.json";
import maMission from "./resources/ma/mission.json";
import maChat from "./resources/ma/chat.json";

// European / other languages
import esCommon from "./resources/es/common.json";
import esMission from "./resources/es/mission.json";
import esChat from "./resources/es/chat.json";

import frCommon from "./resources/fr/common.json";
import frMission from "./resources/fr/mission.json";
import frChat from "./resources/fr/chat.json";

import deCommon from "./resources/de/common.json";
import deMission from "./resources/de/mission.json";
import deChat from "./resources/de/chat.json";

import elCommon from "./resources/el/common.json";
import elMission from "./resources/el/mission.json";
import elChat from "./resources/el/chat.json";

import itCommon from "./resources/it/common.json";
import itMission from "./resources/it/mission.json";
import itChat from "./resources/it/chat.json";

import heCommon from "./resources/he/common.json";
import heMission from "./resources/he/mission.json";
import heChat from "./resources/he/chat.json";

import ptCommon from "./resources/pt/common.json";
import ptMission from "./resources/pt/mission.json";
import ptChat from "./resources/pt/chat.json";

import roCommon from "./resources/ro/common.json";
import roMission from "./resources/ro/mission.json";
import roChat from "./resources/ro/chat.json";

import kaCommon from "./resources/ka/common.json";
import kaMission from "./resources/ka/mission.json";
import kaChat from "./resources/ka/chat.json";

import trCommon from "./resources/tr/common.json";
import trMission from "./resources/tr/mission.json";
import trChat from "./resources/tr/chat.json";

import plCommon from "./resources/pl/common.json";
import plMission from "./resources/pl/mission.json";
import plChat from "./resources/pl/chat.json";

import ukCommon from "./resources/uk/common.json";
import ukMission from "./resources/uk/mission.json";
import ukChat from "./resources/uk/chat.json";

import nlCommon from "./resources/nl/common.json";
import nlMission from "./resources/nl/mission.json";
import nlChat from "./resources/nl/chat.json";

import srCommon from "./resources/sr/common.json";
import srMission from "./resources/sr/mission.json";
import srChat from "./resources/sr/chat.json";

import kkCommon from "./resources/kk/common.json";
import kkMission from "./resources/kk/mission.json";
import kkChat from "./resources/kk/chat.json";

import uzCommon from "./resources/uz/common.json";
import uzMission from "./resources/uz/mission.json";
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
    mission: enMission,
    chat: enChat,
  },
  ru: {
    common: ruCommon,
    mission: ruMission,
    chat: ruChat,
  },
  bg: {
    common: bgCommon,
    mission: bgMission,
    chat: bgChat,
  },
  da: {
    common: daCommon,
    mission: daMission,
    chat: daChat,
  },
  et: {
    common: etCommon,
    mission: etMission,
    chat: etChat,
  },
  fi: {
    common: fiCommon,
    mission: fiMission,
    chat: fiChat,
  },
  kk: {
    common: kkCommon,
    mission: kkMission,
    chat: kkChat,
  },
  uz: {
    common: uzCommon,
    mission: uzMission,
    chat: uzChat,
  },

  // African & MENA languages (LTR unless marked RTL via isRtlLanguage)
  sw: {
    common: swCommon,
    mission: swMission,
    chat: swChat,
  },
  ar: {
    common: arCommon,
    mission: arMission,
    chat: arChat,
  },
  fa: {
    common: faCommon,
    mission: faMission,
    chat: faChat,
  },
  ha: {
    common: haCommon,
    mission: haMission,
    chat: haChat,
  },
  am: {
    common: amCommon,
    mission: amMission,
    chat: amChat,
  },
  yo: {
    common: yoCommon,
    mission: yoMission,
    chat: yoChat,
  },
  zu: {
    common: zuCommon,
    mission: zuMission,
    chat: zuChat,
  },
  ma: {
    common: maCommon,
    mission: maMission,
    chat: maChat,
  },

  // European & other languages
  es: {
    common: esCommon,
    mission: esMission,
    chat: esChat,
  },
  fr: {
    common: frCommon,
    mission: frMission,
    chat: frChat,
  },
  de: {
    common: deCommon,
    mission: deMission,
    chat: deChat,
  },
  el: {
    common: elCommon,
    mission: elMission,
    chat: elChat,
  },
  it: {
    common: itCommon,
    mission: itMission,
    chat: itChat,
  },
  he: {
    common: heCommon,
    mission: heMission,
    chat: heChat,
  },
  pt: {
    common: ptCommon,
    mission: ptMission,
    chat: ptChat,
  },
  ro: {
    common: roCommon,
    mission: roMission,
    chat: roChat,
  },
  ka: {
    common: kaCommon,
    mission: kaMission,
    chat: kaChat,
  },
  tr: {
    common: trCommon,
    mission: trMission,
    chat: trChat,
  },
  pl: {
    common: plCommon,
    mission: plMission,
    chat: plChat,
  },
  uk: {
    common: ukCommon,
    mission: ukMission,
    chat: ukChat,
  },
  nl: {
    common: nlCommon,
    mission: nlMission,
    chat: nlChat,
  },
  sr: {
    common: srCommon,
    mission: srMission,
    chat: srChat,
  },
  no: {
    common: noCommon,
    mission: noMission,
    chat: noChat,
  },
  sv: {
    common: svCommon,
    mission: svMission,
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
  ns: ["common", "mission", "chat"],
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
