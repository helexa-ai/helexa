#!/usr/bin/env node

/**
 * helexa.ai i18n language label consistency check
 *
 * This script validates that for every supported language code in
 * `SUPPORTED_LANGUAGES`:
 *
 *   1. The English `common.lang` map (`src/i18n/resources/en/common.json`)
 *      contains a human-readable label at `lang.<code>`.
 *   2. Optionally (and more leniently), other languages that define a
 *      `lang` block do not omit supported language codes.
 *
 * At minimum, it enforces that the English UI always has labels for all
 * supported languages, since the header language selector renders
 * `t("lang.<code>")` using the active language.
 *
 * Exit codes:
 *   - 0: all checks pass
 *   - 1: one or more inconsistencies found
 */

import fs from "fs";
import path from "path";
import url from "url";

const ROOT = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const I18N_DIR = path.join(ROOT, "src", "i18n");
const LANGUAGES_TS = path.join(I18N_DIR, "languages.ts");
const EN_COMMON_JSON = path.join(I18N_DIR, "resources", "en", "common.json");

/**
 * Utility: read a file as UTF-8 or throw with a helpful message.
 */
function readFileOrDie(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch (err) {
    throw new Error(`Failed to read ${filePath}: ${err.message}`);
  }
}

/**
 * Parse SUPPORTED_LANGUAGES from languages.ts.
 *
 * Expects a definition like:
 *   export const SUPPORTED_LANGUAGES: LanguageCode[] = [
 *     "en",
 *     "bg",
 *   ];
 */
function parseSupportedLanguages(source) {
  const marker = "export const SUPPORTED_LANGUAGES";
  const start = source.indexOf(marker);
  if (start === -1) {
    throw new Error("Could not find `SUPPORTED_LANGUAGES` in languages.ts");
  }

  const after = source.slice(start);
  const bracketIndex = after.indexOf("[");
  const closingIndex = after.indexOf("];");
  if (bracketIndex === -1 || closingIndex === -1) {
    throw new Error("Malformed SUPPORTED_LANGUAGES array");
  }

  const arraySlice = after.slice(bracketIndex + 1, closingIndex);
  const codes = new Set();
  const regex = /"([^"]+)"/g;
  let m;
  while ((m = regex.exec(arraySlice)) !== null) {
    codes.add(m[1]);
  }

  if (codes.size === 0) {
    throw new Error("No entries found in SUPPORTED_LANGUAGES");
  }

  return codes;
}

/**
 * Safely read and parse a JSON file.
 */
function readJsonOrDie(filePath) {
  const raw = readFileOrDie(filePath);
  try {
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(`Failed to parse JSON at ${filePath}: ${err.message}`);
  }
}

/**
 * Compute set difference: a \ b
 */
function difference(a, b) {
  const result = new Set();
  for (const x of a) {
    if (!b.has(x)) result.add(x);
  }
  return result;
}

/**
 * Convenience: sorted array from a Set.
 */
function toSortedArray(set) {
  return [...set].sort();
}

function main() {
  console.log("helexa.ai i18n language label check");
  console.log(`Root: ${ROOT}`);
  console.log(`Languages file: ${LANGUAGES_TS}`);
  console.log(`English common.json: ${EN_COMMON_JSON}`);
  console.log("");

  let hadIssues = false;

  // 1) Load SUPPORTED_LANGUAGES
  let languagesSource;
  try {
    languagesSource = readFileOrDie(LANGUAGES_TS);
  } catch (err) {
    console.error(err.message);
    process.exitCode = 1;
    return;
  }

  let supportedLanguages;
  try {
    supportedLanguages = parseSupportedLanguages(languagesSource);
    console.log(
      `SUPPORTED_LANGUAGES (${supportedLanguages.size}): ${toSortedArray(
        supportedLanguages,
      ).join(", ")}`,
    );
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  console.log("");

  // 2) Load English common.json and extract lang map
  let enCommon;
  try {
    enCommon = readJsonOrDie(EN_COMMON_JSON);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  const enLangMap = enCommon?.lang ?? {};
  if (!enLangMap || typeof enLangMap !== "object") {
    console.error("ERROR: `en/common.json` does not contain a `lang` object.");
    process.exitCode = 1;
    return;
  }

  const enLangKeys = new Set(Object.keys(enLangMap));
  console.log(
    `English lang map keys (${enLangKeys.size}): ${toSortedArray(
      enLangKeys,
    ).join(", ")}`,
  );
  console.log("");

  // 3) Ensure every supported language has a corresponding key in en.lang
  const missingInEnglish = difference(supportedLanguages, enLangKeys);
  if (missingInEnglish.size > 0) {
    hadIssues = true;
    console.error(
      "ERROR: The following supported languages are missing labels in `en/common.json` under `lang`:",
    );
    for (const code of toSortedArray(missingInEnglish)) {
      console.error(`  - lang.${code}`);
    }
  } else {
    console.log(
      "OK: Every supported language has a `lang.<code>` entry in `en/common.json`.",
    );
  }

  console.log("");

  // 4) Optional: scan other locales for informational consistency
  //    (non-fatal, just warnings).
  const resourcesDir = path.join(I18N_DIR, "resources");
  let otherLocales = [];
  try {
    otherLocales = fs
      .readdirSync(resourcesDir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && d.name !== "en")
      .map((d) => d.name);
  } catch (err) {
    console.warn(
      `WARN: Could not list locales in ${resourcesDir}: ${err.message}`,
    );
  }

  for (const locale of otherLocales) {
    const filePath = path.join(resourcesDir, locale, "common.json");
    if (!fs.existsSync(filePath)) continue;

    let localeJson;
    try {
      localeJson = readJsonOrDie(filePath);
    } catch (err) {
      console.warn(`WARN: Failed to read ${filePath}: ${err.message}`);
      continue;
    }

    const langObj = localeJson?.lang;
    if (!langObj || typeof langObj !== "object") {
      // Not all locales need to maintain a full lang map; skip silently.
      continue;
    }

    const localeKeys = new Set(Object.keys(langObj));
    const missingHere = difference(supportedLanguages, localeKeys);
    if (missingHere.size > 0) {
      console.warn(
        `WARN: Locale '${locale}' is missing some supported language labels under "lang":`,
      );
      for (const code of toSortedArray(missingHere)) {
        console.warn(`  - lang.${code}`);
      }
      console.warn("");
    }
  }

  if (hadIssues) {
    console.error("Language label check completed with inconsistencies.");
    process.exitCode = 1;
  } else {
    console.log("Language label check completed successfully.");
    process.exitCode = 0;
  }
}

try {
  main();
} catch (err) {
  console.error("Unexpected error while running language label check:", err);
  process.exitCode = 1;
}
