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
import enImages from "./resources/en/images.json";
import enDocs from "./resources/en/docs.json";
import enAccount from "./resources/en/account.json";
import enPrivacy from "./resources/en/privacy.json";
import ruChat from "./resources/ru/chat.json";
import ruImages from "./resources/ru/images.json";
import ruDocs from "./resources/ru/docs.json";
import ruAccount from "./resources/ru/account.json";
import ruPrivacy from "./resources/ru/privacy.json";

// Scandinavian & Nordic languages
import daCommon from "./resources/da/common.json";
import daMission from "./resources/da/mission.json";
import daChat from "./resources/da/chat.json";
import daImages from "./resources/da/images.json";
import daDocs from "./resources/da/docs.json";
import daAccount from "./resources/da/account.json";
import daPrivacy from "./resources/da/privacy.json";

import fiCommon from "./resources/fi/common.json";
import fiMission from "./resources/fi/mission.json";
import fiChat from "./resources/fi/chat.json";
import fiImages from "./resources/fi/images.json";
import fiDocs from "./resources/fi/docs.json";
import fiAccount from "./resources/fi/account.json";
import fiPrivacy from "./resources/fi/privacy.json";

import noCommon from "./resources/no/common.json";
import noMission from "./resources/no/mission.json";
import noChat from "./resources/no/chat.json";
import noImages from "./resources/no/images.json";
import noDocs from "./resources/no/docs.json";
import noAccount from "./resources/no/account.json";
import noPrivacy from "./resources/no/privacy.json";

import svCommon from "./resources/sv/common.json";
import svMission from "./resources/sv/mission.json";
import svChat from "./resources/sv/chat.json";
import svImages from "./resources/sv/images.json";
import svDocs from "./resources/sv/docs.json";
import svAccount from "./resources/sv/account.json";
import svPrivacy from "./resources/sv/privacy.json";

import bgCommon from "./resources/bg/common.json";
import bgMission from "./resources/bg/mission.json";
import bgChat from "./resources/bg/chat.json";
import bgImages from "./resources/bg/images.json";
import bgDocs from "./resources/bg/docs.json";
import bgAccount from "./resources/bg/account.json";
import bgPrivacy from "./resources/bg/privacy.json";

import etCommon from "./resources/et/common.json";
import etMission from "./resources/et/mission.json";
import etChat from "./resources/et/chat.json";
import etImages from "./resources/et/images.json";
import etDocs from "./resources/et/docs.json";
import etAccount from "./resources/et/account.json";
import etPrivacy from "./resources/et/privacy.json";

// African & MENA languages
import swCommon from "./resources/sw/common.json";
import swMission from "./resources/sw/mission.json";
import swChat from "./resources/sw/chat.json";
import swImages from "./resources/sw/images.json";
import swDocs from "./resources/sw/docs.json";
import swAccount from "./resources/sw/account.json";
import swPrivacy from "./resources/sw/privacy.json";

import arCommon from "./resources/ar/common.json";
import arMission from "./resources/ar/mission.json";
import arChat from "./resources/ar/chat.json";
import arImages from "./resources/ar/images.json";
import arDocs from "./resources/ar/docs.json";
import arAccount from "./resources/ar/account.json";
import arPrivacy from "./resources/ar/privacy.json";

import faCommon from "./resources/fa/common.json";
import faMission from "./resources/fa/mission.json";
import faChat from "./resources/fa/chat.json";
import faImages from "./resources/fa/images.json";
import faDocs from "./resources/fa/docs.json";
import faAccount from "./resources/fa/account.json";
import faPrivacy from "./resources/fa/privacy.json";

import haCommon from "./resources/ha/common.json";
import haMission from "./resources/ha/mission.json";
import haChat from "./resources/ha/chat.json";
import haImages from "./resources/ha/images.json";
import haDocs from "./resources/ha/docs.json";
import haAccount from "./resources/ha/account.json";
import haPrivacy from "./resources/ha/privacy.json";

import amCommon from "./resources/am/common.json";
import amMission from "./resources/am/mission.json";
import amChat from "./resources/am/chat.json";
import amImages from "./resources/am/images.json";
import amDocs from "./resources/am/docs.json";
import amAccount from "./resources/am/account.json";
import amPrivacy from "./resources/am/privacy.json";

import yoCommon from "./resources/yo/common.json";
import yoMission from "./resources/yo/mission.json";
import yoChat from "./resources/yo/chat.json";
import yoImages from "./resources/yo/images.json";
import yoDocs from "./resources/yo/docs.json";
import yoAccount from "./resources/yo/account.json";
import yoPrivacy from "./resources/yo/privacy.json";

import zuCommon from "./resources/zu/common.json";
import zuMission from "./resources/zu/mission.json";
import zuChat from "./resources/zu/chat.json";
import zuImages from "./resources/zu/images.json";
import zuDocs from "./resources/zu/docs.json";
import zuAccount from "./resources/zu/account.json";
import zuPrivacy from "./resources/zu/privacy.json";

// Darija (Moroccan Arabic)
import maCommon from "./resources/ma/common.json";
import maMission from "./resources/ma/mission.json";
import maChat from "./resources/ma/chat.json";
import maImages from "./resources/ma/images.json";
import maDocs from "./resources/ma/docs.json";
import maAccount from "./resources/ma/account.json";
import maPrivacy from "./resources/ma/privacy.json";

// European / other languages
import esCommon from "./resources/es/common.json";
import esMission from "./resources/es/mission.json";
import esChat from "./resources/es/chat.json";
import esImages from "./resources/es/images.json";
import esDocs from "./resources/es/docs.json";
import esAccount from "./resources/es/account.json";
import esPrivacy from "./resources/es/privacy.json";

import frCommon from "./resources/fr/common.json";
import frMission from "./resources/fr/mission.json";
import frChat from "./resources/fr/chat.json";
import frImages from "./resources/fr/images.json";
import frDocs from "./resources/fr/docs.json";
import frAccount from "./resources/fr/account.json";
import frPrivacy from "./resources/fr/privacy.json";

import deCommon from "./resources/de/common.json";
import deMission from "./resources/de/mission.json";
import deChat from "./resources/de/chat.json";
import deImages from "./resources/de/images.json";
import deDocs from "./resources/de/docs.json";
import deAccount from "./resources/de/account.json";
import dePrivacy from "./resources/de/privacy.json";

import elCommon from "./resources/el/common.json";
import elMission from "./resources/el/mission.json";
import elChat from "./resources/el/chat.json";
import elImages from "./resources/el/images.json";
import elDocs from "./resources/el/docs.json";
import elAccount from "./resources/el/account.json";
import elPrivacy from "./resources/el/privacy.json";

import itCommon from "./resources/it/common.json";
import itMission from "./resources/it/mission.json";
import itChat from "./resources/it/chat.json";
import itImages from "./resources/it/images.json";
import itDocs from "./resources/it/docs.json";
import itAccount from "./resources/it/account.json";
import itPrivacy from "./resources/it/privacy.json";

import heCommon from "./resources/he/common.json";
import heMission from "./resources/he/mission.json";
import heChat from "./resources/he/chat.json";
import heImages from "./resources/he/images.json";
import heDocs from "./resources/he/docs.json";
import heAccount from "./resources/he/account.json";
import hePrivacy from "./resources/he/privacy.json";

import ptCommon from "./resources/pt/common.json";
import ptMission from "./resources/pt/mission.json";
import ptChat from "./resources/pt/chat.json";
import ptImages from "./resources/pt/images.json";
import ptDocs from "./resources/pt/docs.json";
import ptAccount from "./resources/pt/account.json";
import ptPrivacy from "./resources/pt/privacy.json";

import roCommon from "./resources/ro/common.json";
import roMission from "./resources/ro/mission.json";
import roChat from "./resources/ro/chat.json";
import roImages from "./resources/ro/images.json";
import roDocs from "./resources/ro/docs.json";
import roAccount from "./resources/ro/account.json";
import roPrivacy from "./resources/ro/privacy.json";

import kaCommon from "./resources/ka/common.json";
import kaMission from "./resources/ka/mission.json";
import kaChat from "./resources/ka/chat.json";
import kaImages from "./resources/ka/images.json";
import kaDocs from "./resources/ka/docs.json";
import kaAccount from "./resources/ka/account.json";
import kaPrivacy from "./resources/ka/privacy.json";

import trCommon from "./resources/tr/common.json";
import trMission from "./resources/tr/mission.json";
import trChat from "./resources/tr/chat.json";
import trImages from "./resources/tr/images.json";
import trDocs from "./resources/tr/docs.json";
import trAccount from "./resources/tr/account.json";
import trPrivacy from "./resources/tr/privacy.json";

import plCommon from "./resources/pl/common.json";
import plMission from "./resources/pl/mission.json";
import plChat from "./resources/pl/chat.json";
import plImages from "./resources/pl/images.json";
import plDocs from "./resources/pl/docs.json";
import plAccount from "./resources/pl/account.json";
import plPrivacy from "./resources/pl/privacy.json";

import ukCommon from "./resources/uk/common.json";
import ukMission from "./resources/uk/mission.json";
import ukChat from "./resources/uk/chat.json";
import ukImages from "./resources/uk/images.json";
import ukDocs from "./resources/uk/docs.json";
import ukAccount from "./resources/uk/account.json";
import ukPrivacy from "./resources/uk/privacy.json";

import nlCommon from "./resources/nl/common.json";
import nlMission from "./resources/nl/mission.json";
import nlChat from "./resources/nl/chat.json";
import nlImages from "./resources/nl/images.json";
import nlDocs from "./resources/nl/docs.json";
import nlAccount from "./resources/nl/account.json";
import nlPrivacy from "./resources/nl/privacy.json";

import srCommon from "./resources/sr/common.json";
import srMission from "./resources/sr/mission.json";
import srChat from "./resources/sr/chat.json";
import srImages from "./resources/sr/images.json";
import srDocs from "./resources/sr/docs.json";
import srAccount from "./resources/sr/account.json";
import srPrivacy from "./resources/sr/privacy.json";

import kkCommon from "./resources/kk/common.json";
import kkMission from "./resources/kk/mission.json";
import kkChat from "./resources/kk/chat.json";
import kkImages from "./resources/kk/images.json";
import kkDocs from "./resources/kk/docs.json";
import kkAccount from "./resources/kk/account.json";
import kkPrivacy from "./resources/kk/privacy.json";

import uzCommon from "./resources/uz/common.json";
import uzMission from "./resources/uz/mission.json";
import uzChat from "./resources/uz/chat.json";
import uzImages from "./resources/uz/images.json";
import uzDocs from "./resources/uz/docs.json";
import uzAccount from "./resources/uz/account.json";
import uzPrivacy from "./resources/uz/privacy.json";

// EU-24 completion + Catalan (2026-07-16)
import caCommon from "./resources/ca/common.json";
import caMission from "./resources/ca/mission.json";
import caChat from "./resources/ca/chat.json";
import caImages from "./resources/ca/images.json";
import caDocs from "./resources/ca/docs.json";
import caAccount from "./resources/ca/account.json";
import caPrivacy from "./resources/ca/privacy.json";
import csCommon from "./resources/cs/common.json";
import csMission from "./resources/cs/mission.json";
import csChat from "./resources/cs/chat.json";
import csImages from "./resources/cs/images.json";
import csDocs from "./resources/cs/docs.json";
import csAccount from "./resources/cs/account.json";
import csPrivacy from "./resources/cs/privacy.json";
import gaCommon from "./resources/ga/common.json";
import gaMission from "./resources/ga/mission.json";
import gaChat from "./resources/ga/chat.json";
import gaImages from "./resources/ga/images.json";
import gaDocs from "./resources/ga/docs.json";
import gaAccount from "./resources/ga/account.json";
import gaPrivacy from "./resources/ga/privacy.json";
import hrCommon from "./resources/hr/common.json";
import hrMission from "./resources/hr/mission.json";
import hrChat from "./resources/hr/chat.json";
import hrImages from "./resources/hr/images.json";
import hrDocs from "./resources/hr/docs.json";
import hrAccount from "./resources/hr/account.json";
import hrPrivacy from "./resources/hr/privacy.json";
import huCommon from "./resources/hu/common.json";
import huMission from "./resources/hu/mission.json";
import huChat from "./resources/hu/chat.json";
import huImages from "./resources/hu/images.json";
import huDocs from "./resources/hu/docs.json";
import huAccount from "./resources/hu/account.json";
import huPrivacy from "./resources/hu/privacy.json";
import ltCommon from "./resources/lt/common.json";
import ltMission from "./resources/lt/mission.json";
import ltChat from "./resources/lt/chat.json";
import ltImages from "./resources/lt/images.json";
import ltDocs from "./resources/lt/docs.json";
import ltAccount from "./resources/lt/account.json";
import ltPrivacy from "./resources/lt/privacy.json";
import lvCommon from "./resources/lv/common.json";
import lvMission from "./resources/lv/mission.json";
import lvChat from "./resources/lv/chat.json";
import lvImages from "./resources/lv/images.json";
import lvDocs from "./resources/lv/docs.json";
import lvAccount from "./resources/lv/account.json";
import lvPrivacy from "./resources/lv/privacy.json";
import mtCommon from "./resources/mt/common.json";
import mtMission from "./resources/mt/mission.json";
import mtChat from "./resources/mt/chat.json";
import mtImages from "./resources/mt/images.json";
import mtDocs from "./resources/mt/docs.json";
import mtAccount from "./resources/mt/account.json";
import mtPrivacy from "./resources/mt/privacy.json";
import skCommon from "./resources/sk/common.json";
import skMission from "./resources/sk/mission.json";
import skChat from "./resources/sk/chat.json";
import skImages from "./resources/sk/images.json";
import skDocs from "./resources/sk/docs.json";
import skAccount from "./resources/sk/account.json";
import skPrivacy from "./resources/sk/privacy.json";
import slCommon from "./resources/sl/common.json";
import slMission from "./resources/sl/mission.json";
import slChat from "./resources/sl/chat.json";
import slImages from "./resources/sl/images.json";
import slDocs from "./resources/sl/docs.json";
import slAccount from "./resources/sl/account.json";
import slPrivacy from "./resources/sl/privacy.json";

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
    images: enImages,
    docs: enDocs,
    account: enAccount,
    privacy: enPrivacy,
  },
  ru: {
    common: ruCommon,
    mission: ruMission,
    chat: ruChat,
    images: ruImages,
    docs: ruDocs,
    account: ruAccount,
    privacy: ruPrivacy,
  },
  bg: {
    common: bgCommon,
    mission: bgMission,
    chat: bgChat,
    images: bgImages,
    docs: bgDocs,
    account: bgAccount,
    privacy: bgPrivacy,
  },
  da: {
    common: daCommon,
    mission: daMission,
    chat: daChat,
    images: daImages,
    docs: daDocs,
    account: daAccount,
    privacy: daPrivacy,
  },
  et: {
    common: etCommon,
    mission: etMission,
    chat: etChat,
    images: etImages,
    docs: etDocs,
    account: etAccount,
    privacy: etPrivacy,
  },
  fi: {
    common: fiCommon,
    mission: fiMission,
    chat: fiChat,
    images: fiImages,
    docs: fiDocs,
    account: fiAccount,
    privacy: fiPrivacy,
  },
  kk: {
    common: kkCommon,
    mission: kkMission,
    chat: kkChat,
    images: kkImages,
    docs: kkDocs,
    account: kkAccount,
    privacy: kkPrivacy,
  },
  uz: {
    common: uzCommon,
    mission: uzMission,
    chat: uzChat,
    images: uzImages,
    docs: uzDocs,
    account: uzAccount,
    privacy: uzPrivacy,
  },

  // African & MENA languages (LTR unless marked RTL via isRtlLanguage)
  sw: {
    common: swCommon,
    mission: swMission,
    chat: swChat,
    images: swImages,
    docs: swDocs,
    account: swAccount,
    privacy: swPrivacy,
  },
  ar: {
    common: arCommon,
    mission: arMission,
    chat: arChat,
    images: arImages,
    docs: arDocs,
    account: arAccount,
    privacy: arPrivacy,
  },
  fa: {
    common: faCommon,
    mission: faMission,
    chat: faChat,
    images: faImages,
    docs: faDocs,
    account: faAccount,
    privacy: faPrivacy,
  },
  ha: {
    common: haCommon,
    mission: haMission,
    chat: haChat,
    images: haImages,
    docs: haDocs,
    account: haAccount,
    privacy: haPrivacy,
  },
  am: {
    common: amCommon,
    mission: amMission,
    chat: amChat,
    images: amImages,
    docs: amDocs,
    account: amAccount,
    privacy: amPrivacy,
  },
  yo: {
    common: yoCommon,
    mission: yoMission,
    chat: yoChat,
    images: yoImages,
    docs: yoDocs,
    account: yoAccount,
    privacy: yoPrivacy,
  },
  zu: {
    common: zuCommon,
    mission: zuMission,
    chat: zuChat,
    images: zuImages,
    docs: zuDocs,
    account: zuAccount,
    privacy: zuPrivacy,
  },
  ma: {
    common: maCommon,
    mission: maMission,
    chat: maChat,
    images: maImages,
    docs: maDocs,
    account: maAccount,
    privacy: maPrivacy,
  },

  // European & other languages
  es: {
    common: esCommon,
    mission: esMission,
    chat: esChat,
    images: esImages,
    docs: esDocs,
    account: esAccount,
    privacy: esPrivacy,
  },
  fr: {
    common: frCommon,
    mission: frMission,
    chat: frChat,
    images: frImages,
    docs: frDocs,
    account: frAccount,
    privacy: frPrivacy,
  },
  de: {
    common: deCommon,
    mission: deMission,
    chat: deChat,
    images: deImages,
    docs: deDocs,
    account: deAccount,
    privacy: dePrivacy,
  },
  el: {
    common: elCommon,
    mission: elMission,
    chat: elChat,
    images: elImages,
    docs: elDocs,
    account: elAccount,
    privacy: elPrivacy,
  },
  it: {
    common: itCommon,
    mission: itMission,
    chat: itChat,
    images: itImages,
    docs: itDocs,
    account: itAccount,
    privacy: itPrivacy,
  },
  he: {
    common: heCommon,
    mission: heMission,
    chat: heChat,
    images: heImages,
    docs: heDocs,
    account: heAccount,
    privacy: hePrivacy,
  },
  pt: {
    common: ptCommon,
    mission: ptMission,
    chat: ptChat,
    images: ptImages,
    docs: ptDocs,
    account: ptAccount,
    privacy: ptPrivacy,
  },
  ro: {
    common: roCommon,
    mission: roMission,
    chat: roChat,
    images: roImages,
    docs: roDocs,
    account: roAccount,
    privacy: roPrivacy,
  },
  ka: {
    common: kaCommon,
    mission: kaMission,
    chat: kaChat,
    images: kaImages,
    docs: kaDocs,
    account: kaAccount,
    privacy: kaPrivacy,
  },
  tr: {
    common: trCommon,
    mission: trMission,
    chat: trChat,
    images: trImages,
    docs: trDocs,
    account: trAccount,
    privacy: trPrivacy,
  },
  pl: {
    common: plCommon,
    mission: plMission,
    chat: plChat,
    images: plImages,
    docs: plDocs,
    account: plAccount,
    privacy: plPrivacy,
  },
  uk: {
    common: ukCommon,
    mission: ukMission,
    chat: ukChat,
    images: ukImages,
    docs: ukDocs,
    account: ukAccount,
    privacy: ukPrivacy,
  },
  nl: {
    common: nlCommon,
    mission: nlMission,
    chat: nlChat,
    images: nlImages,
    docs: nlDocs,
    account: nlAccount,
    privacy: nlPrivacy,
  },
  sr: {
    common: srCommon,
    mission: srMission,
    chat: srChat,
    images: srImages,
    docs: srDocs,
    account: srAccount,
    privacy: srPrivacy,
  },
  no: {
    common: noCommon,
    mission: noMission,
    chat: noChat,
    images: noImages,
    docs: noDocs,
    account: noAccount,
    privacy: noPrivacy,
  },
  sv: {
    common: svCommon,
    mission: svMission,
    chat: svChat,
    images: svImages,
    docs: svDocs,
    account: svAccount,
    privacy: svPrivacy,
  },
  // EU-24 completion + Catalan (2026-07-16)
  ca: {
    common: caCommon,
    mission: caMission,
    chat: caChat,
    images: caImages,
    docs: caDocs,
    account: caAccount,
    privacy: caPrivacy,
  },
  cs: {
    common: csCommon,
    mission: csMission,
    chat: csChat,
    images: csImages,
    docs: csDocs,
    account: csAccount,
    privacy: csPrivacy,
  },
  ga: {
    common: gaCommon,
    mission: gaMission,
    chat: gaChat,
    images: gaImages,
    docs: gaDocs,
    account: gaAccount,
    privacy: gaPrivacy,
  },
  hr: {
    common: hrCommon,
    mission: hrMission,
    chat: hrChat,
    images: hrImages,
    docs: hrDocs,
    account: hrAccount,
    privacy: hrPrivacy,
  },
  hu: {
    common: huCommon,
    mission: huMission,
    chat: huChat,
    images: huImages,
    docs: huDocs,
    account: huAccount,
    privacy: huPrivacy,
  },
  lt: {
    common: ltCommon,
    mission: ltMission,
    chat: ltChat,
    images: ltImages,
    docs: ltDocs,
    account: ltAccount,
    privacy: ltPrivacy,
  },
  lv: {
    common: lvCommon,
    mission: lvMission,
    chat: lvChat,
    images: lvImages,
    docs: lvDocs,
    account: lvAccount,
    privacy: lvPrivacy,
  },
  mt: {
    common: mtCommon,
    mission: mtMission,
    chat: mtChat,
    images: mtImages,
    docs: mtDocs,
    account: mtAccount,
    privacy: mtPrivacy,
  },
  sk: {
    common: skCommon,
    mission: skMission,
    chat: skChat,
    images: skImages,
    docs: skDocs,
    account: skAccount,
    privacy: skPrivacy,
  },
  sl: {
    common: slCommon,
    mission: slMission,
    chat: slChat,
    images: slImages,
    docs: slDocs,
    account: slAccount,
    privacy: slPrivacy,
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
  // index.html ships `lang="en"`. Without this the document claims English
  // while rendering Arabic — which is what a screen reader announces from,
  // and what the browser picks a font and hyphenation dictionary with.
  document.documentElement.lang = browserLang;
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
  ns: ["common", "mission", "chat", "images", "account", "privacy"],
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
// and the document language both track the new language.
i18n.on("languageChanged", (lng) => {
  if (typeof document === "undefined") return;
  const lang = normalizeLocaleToLanguage(lng);
  document.documentElement.dir = isRtlLanguage(lang) ? "rtl" : "ltr";
  document.documentElement.lang = lang;
});

export default i18n;
