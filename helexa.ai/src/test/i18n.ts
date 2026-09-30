// A minimal i18n instance for component tests: English only, so tests can
// find elements by the text a visitor actually reads.
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import enCommon from "../i18n/resources/en/common.json";
import enDecisions from "../i18n/resources/en/decisions.json";

void i18n.use(initReactI18next).init({
  lng: "en",
  fallbackLng: "en",
  resources: { en: { common: enCommon, decisions: enDecisions } },
  ns: ["common", "decisions"],
  defaultNS: "common",
  interpolation: { escapeValue: false },
  react: { useSuspense: false },
});

export default i18n;
