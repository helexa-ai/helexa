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
import enAccount from "./resources/en/account.json";
import enPrivacy from "./resources/en/privacy.json";
import ruChat from "./resources/ru/chat.json";
import ruAccount from "./resources/ru/account.json";
import ruPrivacy from "./resources/ru/privacy.json";

// Scandinavian & Nordic languages
import daCommon from "./resources/da/common.json";
import daMission from "./resources/da/mission.json";
import daChat from "./resources/da/chat.json";
import daAccount from "./resources/da/account.json";
import daPrivacy from "./resources/da/privacy.json";

import fiCommon from "./resources/fi/common.json";
import fiMission from "./resources/fi/mission.json";
import fiChat from "./resources/fi/chat.json";
import fiAccount from "./resources/fi/account.json";
import fiPrivacy from "./resources/fi/privacy.json";

import noCommon from "./resources/no/common.json";
import noMission from "./resources/no/mission.json";
import noChat from "./resources/no/chat.json";
import noAccount from "./resources/no/account.json";
import noPrivacy from "./resources/no/privacy.json";

import svCommon from "./resources/sv/common.json";
import svMission from "./resources/sv/mission.json";
import svChat from "./resources/sv/chat.json";
import svAccount from "./resources/sv/account.json";
import svPrivacy from "./resources/sv/privacy.json";

import bgCommon from "./resources/bg/common.json";
import bgMission from "./resources/bg/mission.json";
import bgChat from "./resources/bg/chat.json";
import bgAccount from "./resources/bg/account.json";
import bgPrivacy from "./resources/bg/privacy.json";

import etCommon from "./resources/et/common.json";
import etMission from "./resources/et/mission.json";
import etChat from "./resources/et/chat.json";
import etAccount from "./resources/et/account.json";
import etPrivacy from "./resources/et/privacy.json";

// African & MENA languages
import swCommon from "./resources/sw/common.json";
import swMission from "./resources/sw/mission.json";
import swChat from "./resources/sw/chat.json";
import swAccount from "./resources/sw/account.json";
import swPrivacy from "./resources/sw/privacy.json";

import arCommon from "./resources/ar/common.json";
import arMission from "./resources/ar/mission.json";
import arChat from "./resources/ar/chat.json";
import arAccount from "./resources/ar/account.json";
import arPrivacy from "./resources/ar/privacy.json";

import faCommon from "./resources/fa/common.json";
import faMission from "./resources/fa/mission.json";
import faChat from "./resources/fa/chat.json";
import faAccount from "./resources/fa/account.json";
import faPrivacy from "./resources/fa/privacy.json";

import haCommon from "./resources/ha/common.json";
import haMission from "./resources/ha/mission.json";
import haChat from "./resources/ha/chat.json";
import haAccount from "./resources/ha/account.json";
import haPrivacy from "./resources/ha/privacy.json";

import amCommon from "./resources/am/common.json";
import amMission from "./resources/am/mission.json";
import amChat from "./resources/am/chat.json";
import amAccount from "./resources/am/account.json";
import amPrivacy from "./resources/am/privacy.json";

import yoCommon from "./resources/yo/common.json";
import yoMission from "./resources/yo/mission.json";
import yoChat from "./resources/yo/chat.json";
import yoAccount from "./resources/yo/account.json";
import yoPrivacy from "./resources/yo/privacy.json";

import zuCommon from "./resources/zu/common.json";
import zuMission from "./resources/zu/mission.json";
import zuChat from "./resources/zu/chat.json";
import zuAccount from "./resources/zu/account.json";
import zuPrivacy from "./resources/zu/privacy.json";

// Darija (Moroccan Arabic)
import maCommon from "./resources/ma/common.json";
import maMission from "./resources/ma/mission.json";
import maChat from "./resources/ma/chat.json";
import maAccount from "./resources/ma/account.json";
import maPrivacy from "./resources/ma/privacy.json";

// European / other languages
import esCommon from "./resources/es/common.json";
import esMission from "./resources/es/mission.json";
import esChat from "./resources/es/chat.json";
import esAccount from "./resources/es/account.json";
import esPrivacy from "./resources/es/privacy.json";

import frCommon from "./resources/fr/common.json";
import frMission from "./resources/fr/mission.json";
import frChat from "./resources/fr/chat.json";
import frAccount from "./resources/fr/account.json";
import frPrivacy from "./resources/fr/privacy.json";

import deCommon from "./resources/de/common.json";
import deMission from "./resources/de/mission.json";
import deChat from "./resources/de/chat.json";
import deAccount from "./resources/de/account.json";
import dePrivacy from "./resources/de/privacy.json";

import elCommon from "./resources/el/common.json";
import elMission from "./resources/el/mission.json";
import elChat from "./resources/el/chat.json";
import elAccount from "./resources/el/account.json";
import elPrivacy from "./resources/el/privacy.json";

import itCommon from "./resources/it/common.json";
import itMission from "./resources/it/mission.json";
import itChat from "./resources/it/chat.json";
import itAccount from "./resources/it/account.json";
import itPrivacy from "./resources/it/privacy.json";

import heCommon from "./resources/he/common.json";
import heMission from "./resources/he/mission.json";
import heChat from "./resources/he/chat.json";
import heAccount from "./resources/he/account.json";
import hePrivacy from "./resources/he/privacy.json";

import ptCommon from "./resources/pt/common.json";
import ptMission from "./resources/pt/mission.json";
import ptChat from "./resources/pt/chat.json";
import ptAccount from "./resources/pt/account.json";
import ptPrivacy from "./resources/pt/privacy.json";

import roCommon from "./resources/ro/common.json";
import roMission from "./resources/ro/mission.json";
import roChat from "./resources/ro/chat.json";
import roAccount from "./resources/ro/account.json";
import roPrivacy from "./resources/ro/privacy.json";

import kaCommon from "./resources/ka/common.json";
import kaMission from "./resources/ka/mission.json";
import kaChat from "./resources/ka/chat.json";
import kaAccount from "./resources/ka/account.json";
import kaPrivacy from "./resources/ka/privacy.json";

import trCommon from "./resources/tr/common.json";
import trMission from "./resources/tr/mission.json";
import trChat from "./resources/tr/chat.json";
import trAccount from "./resources/tr/account.json";
import trPrivacy from "./resources/tr/privacy.json";

import plCommon from "./resources/pl/common.json";
import plMission from "./resources/pl/mission.json";
import plChat from "./resources/pl/chat.json";
import plAccount from "./resources/pl/account.json";
import plPrivacy from "./resources/pl/privacy.json";

import ukCommon from "./resources/uk/common.json";
import ukMission from "./resources/uk/mission.json";
import ukChat from "./resources/uk/chat.json";
import ukAccount from "./resources/uk/account.json";
import ukPrivacy from "./resources/uk/privacy.json";

import nlCommon from "./resources/nl/common.json";
import nlMission from "./resources/nl/mission.json";
import nlChat from "./resources/nl/chat.json";
import nlAccount from "./resources/nl/account.json";
import nlPrivacy from "./resources/nl/privacy.json";

import srCommon from "./resources/sr/common.json";
import srMission from "./resources/sr/mission.json";
import srChat from "./resources/sr/chat.json";
import srAccount from "./resources/sr/account.json";
import srPrivacy from "./resources/sr/privacy.json";

import kkCommon from "./resources/kk/common.json";
import kkMission from "./resources/kk/mission.json";
import kkChat from "./resources/kk/chat.json";
import kkAccount from "./resources/kk/account.json";
import kkPrivacy from "./resources/kk/privacy.json";

import uzCommon from "./resources/uz/common.json";
import uzMission from "./resources/uz/mission.json";
import uzChat from "./resources/uz/chat.json";
import uzAccount from "./resources/uz/account.json";
import uzPrivacy from "./resources/uz/privacy.json";

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
    account: enAccount,
    privacy: enPrivacy,
  },
  ru: {
    common: ruCommon,
    mission: ruMission,
    chat: ruChat,
    account: ruAccount,
    privacy: ruPrivacy,
  },
  bg: {
    common: bgCommon,
    mission: bgMission,
    chat: bgChat,
    account: bgAccount,
    privacy: bgPrivacy,
  },
  da: {
    common: daCommon,
    mission: daMission,
    chat: daChat,
    account: daAccount,
    privacy: daPrivacy,
  },
  et: {
    common: etCommon,
    mission: etMission,
    chat: etChat,
    account: etAccount,
    privacy: etPrivacy,
  },
  fi: {
    common: fiCommon,
    mission: fiMission,
    chat: fiChat,
    account: fiAccount,
    privacy: fiPrivacy,
  },
  kk: {
    common: kkCommon,
    mission: kkMission,
    chat: kkChat,
    account: kkAccount,
    privacy: kkPrivacy,
  },
  uz: {
    common: uzCommon,
    mission: uzMission,
    chat: uzChat,
    account: uzAccount,
    privacy: uzPrivacy,
  },

  // African & MENA languages (LTR unless marked RTL via isRtlLanguage)
  sw: {
    common: swCommon,
    mission: swMission,
    chat: swChat,
    account: swAccount,
    privacy: swPrivacy,
  },
  ar: {
    common: arCommon,
    mission: arMission,
    chat: arChat,
    account: arAccount,
    privacy: arPrivacy,
  },
  fa: {
    common: faCommon,
    mission: faMission,
    chat: faChat,
    account: faAccount,
    privacy: faPrivacy,
  },
  ha: {
    common: haCommon,
    mission: haMission,
    chat: haChat,
    account: haAccount,
    privacy: haPrivacy,
  },
  am: {
    common: amCommon,
    mission: amMission,
    chat: amChat,
    account: amAccount,
    privacy: amPrivacy,
  },
  yo: {
    common: yoCommon,
    mission: yoMission,
    chat: yoChat,
    account: yoAccount,
    privacy: yoPrivacy,
  },
  zu: {
    common: zuCommon,
    mission: zuMission,
    chat: zuChat,
    account: zuAccount,
    privacy: zuPrivacy,
  },
  ma: {
    common: maCommon,
    mission: maMission,
    chat: maChat,
    account: maAccount,
    privacy: maPrivacy,
  },

  // European & other languages
  es: {
    common: esCommon,
    mission: esMission,
    chat: esChat,
    account: esAccount,
    privacy: esPrivacy,
  },
  fr: {
    common: frCommon,
    mission: frMission,
    chat: frChat,
    account: frAccount,
    privacy: frPrivacy,
  },
  de: {
    common: deCommon,
    mission: deMission,
    chat: deChat,
    account: deAccount,
    privacy: dePrivacy,
  },
  el: {
    common: elCommon,
    mission: elMission,
    chat: elChat,
    account: elAccount,
    privacy: elPrivacy,
  },
  it: {
    common: itCommon,
    mission: itMission,
    chat: itChat,
    account: itAccount,
    privacy: itPrivacy,
  },
  he: {
    common: heCommon,
    mission: heMission,
    chat: heChat,
    account: heAccount,
    privacy: hePrivacy,
  },
  pt: {
    common: ptCommon,
    mission: ptMission,
    chat: ptChat,
    account: ptAccount,
    privacy: ptPrivacy,
  },
  ro: {
    common: roCommon,
    mission: roMission,
    chat: roChat,
    account: roAccount,
    privacy: roPrivacy,
  },
  ka: {
    common: kaCommon,
    mission: kaMission,
    chat: kaChat,
    account: kaAccount,
    privacy: kaPrivacy,
  },
  tr: {
    common: trCommon,
    mission: trMission,
    chat: trChat,
    account: trAccount,
    privacy: trPrivacy,
  },
  pl: {
    common: plCommon,
    mission: plMission,
    chat: plChat,
    account: plAccount,
    privacy: plPrivacy,
  },
  uk: {
    common: ukCommon,
    mission: ukMission,
    chat: ukChat,
    account: ukAccount,
    privacy: ukPrivacy,
  },
  nl: {
    common: nlCommon,
    mission: nlMission,
    chat: nlChat,
    account: nlAccount,
    privacy: nlPrivacy,
  },
  sr: {
    common: srCommon,
    mission: srMission,
    chat: srChat,
    account: srAccount,
    privacy: srPrivacy,
  },
  no: {
    common: noCommon,
    mission: noMission,
    chat: noChat,
    account: noAccount,
    privacy: noPrivacy,
  },
  sv: {
    common: svCommon,
    mission: svMission,
    chat: svChat,
    account: svAccount,
    privacy: svPrivacy,
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
  ns: ["common", "mission", "chat", "account", "privacy"],
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
