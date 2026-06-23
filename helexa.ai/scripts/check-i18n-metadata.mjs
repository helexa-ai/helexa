#!/usr/bin/env node

/**
 * helexa.ai i18n metadata consistency check
 *
 * This script validates that:
 *  - Every `LanguageCode` used in the project appears in `TRANSLATION_PRIORITY`
 *  - Every `TRANSLATION_PRIORITY` entry refers to a valid `LanguageCode`
 *  - `REMAINING_LANGUAGES` is a subset of `LanguageCode`
 *  - `REMAINING_LANGUAGES` is disjoint from `SUPPORTED_LANGUAGES`
 *
 * It is intentionally implemented as a standalone Node script (no TypeScript
 * build step required) and does a very lightweight parse of the TypeScript
 * source files to avoid pulling in a full TS compiler.
 *
 * Exit codes:
 *  - 0: all checks pass
 *  - 1: one or more metadata inconsistencies found
 */

import fs from "fs";
import path from "path";
import url from "url";

/**
 * Resolve project root as the directory containing this script.
 */
const ROOT = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const I18N_DIR = path.join(ROOT, "src", "i18n");
const LANGUAGES_TS = path.join(I18N_DIR, "languages.ts");
const PRIORITY_TS = path.join(I18N_DIR, "translation-priority.ts");

function readFileOrDie(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch (err) {
    console.error(`Failed to read ${filePath}: ${err.message}`);
    process.exitCode = 1;
    return "";
  }
}

/**
 * Extract union members from a definition like:
 *
 * export type LanguageCode =
 *   | "en"
 *   | "bg"
 *   | "cs";
 */
function parseLanguageCodeUnion(source) {
  const start = source.indexOf("export type LanguageCode");
  if (start === -1) {
    throw new Error("Could not find `export type LanguageCode` in languages.ts");
  }

  const after = source.slice(start);
  const eqIndex = after.indexOf("=");
  if (eqIndex === -1) {
    throw new Error("Malformed LanguageCode definition (no '=')");
  }

  const unionBlock = after.slice(eqIndex + 1);
  const semicolonIndex = unionBlock.indexOf(";");
  const unionSlice = semicolonIndex === -1 ? unionBlock : unionBlock.slice(0, semicolonIndex);

  const codes = new Set();
  const regex = /\|\s*"([^"]+)"/g;
  let m;
  while ((m = regex.exec(unionSlice)) !== null) {
    codes.add(m[1]);
  }

  if (codes.size === 0) {
    throw new Error("No language codes found in LanguageCode union");
  }

  return codes;
}

/**
 * Parse SUPPORTED_LANGUAGES from languages.ts
 *
 * export const SUPPORTED_LANGUAGES: LanguageCode[] = [
 *   "en",
 *   "bg",
 * ];
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
 * Parse REMAINING_LANGUAGES from translation-priority.ts
 *
 * export const REMAINING_LANGUAGES: LanguageCode[] = [
 *   "tr",
 *   "pl",
 * ];
 */
function parseRemainingLanguages(source) {
  const marker = "export const REMAINING_LANGUAGES";
  const start = source.indexOf(marker);
  if (start === -1) {
    // It's valid for this list to not exist; treat as empty if missing
    return new Set();
  }

  const after = source.slice(start);
  const bracketIndex = after.indexOf("[");
  const closingIndex = after.indexOf("];");
  if (bracketIndex === -1 || closingIndex === -1) {
    throw new Error("Malformed REMAINING_LANGUAGES array");
  }

  const arraySlice = after.slice(bracketIndex + 1, closingIndex);
  const codes = new Set();
  const regex = /"([^"]+)"/g;
  let m;
  while ((m = regex.exec(arraySlice)) !== null) {
    codes.add(m[1]);
  }

  return codes;
}

/**
 * Parse TRANSLATION_PRIORITY entries from translation-priority.ts
 *
 * export const TRANSLATION_PRIORITY: TranslationPriorityEntry[] = [
 *   {
 *     code: "tr",
 *     bucket: "high",
 *     nativeSpeakers: "70–90M",
 *   },
 *   ...
 * ];
 */
function parseTranslationPriority(source) {
  const marker = "export const TRANSLATION_PRIORITY";
  const start = source.indexOf(marker);
  if (start === -1) {
    throw new Error("Could not find `TRANSLATION_PRIORITY` in translation-priority.ts");
  }

  const after = source.slice(start);
  const bracketIndex = after.indexOf("[");
  const closingIndex = after.indexOf("];");
  if (bracketIndex === -1 || closingIndex === -1) {
    throw new Error("Malformed TRANSLATION_PRIORITY array");
  }

  const arraySlice = after.slice(bracketIndex + 1, closingIndex);

  // Simple heuristic: find code: "<value>" inside objects
  const codes = new Set();
  const regex = /code:\s*"([^"]+)"/g;
  let m;
  while ((m = regex.exec(arraySlice)) !== null) {
    codes.add(m[1]);
  }

  if (codes.size === 0) {
    throw new Error("No `code` entries found in TRANSLATION_PRIORITY");
  }

  return codes;
}

function difference(a, b) {
  const result = new Set();
  for (const x of a) {
    if (!b.has(x)) result.add(x);
  }
  return result;
}

function toSortedArray(set) {
  return [...set].sort();
}

function main() {
  console.log("helexa.ai i18n metadata consistency check");
  console.log(`Root: ${ROOT}`);
  console.log(`Languages file: ${LANGUAGES_TS}`);
  console.log(`Priority file: ${PRIORITY_TS}`);
  console.log("");

  const languagesSource = readFileOrDie(LANGUAGES_TS);
  const prioritySource = readFileOrDie(PRIORITY_TS);

  let hadIssues = false;

  let languageCodes;
  let supportedLanguages;
  let remainingLanguages;
  let priorityCodes;

  try {
    languageCodes = parseLanguageCodeUnion(languagesSource);
    console.log(`LanguageCode entries: ${languageCodes.size}`);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  try {
    supportedLanguages = parseSupportedLanguages(languagesSource);
    console.log(`SUPPORTED_LANGUAGES entries: ${supportedLanguages.size}`);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  try {
    remainingLanguages = parseRemainingLanguages(prioritySource);
    console.log(`REMAINING_LANGUAGES entries: ${remainingLanguages.size}`);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  try {
    priorityCodes = parseTranslationPriority(prioritySource);
    console.log(`TRANSLATION_PRIORITY entries: ${priorityCodes.size}`);
  } catch (err) {
    console.error(`ERROR: ${err.message}`);
    process.exitCode = 1;
    return;
  }

  console.log("");

  // 1) Every LanguageCode should have a TRANSLATION_PRIORITY entry
  const missingInPriority = difference(languageCodes, priorityCodes);
  if (missingInPriority.size > 0) {
    hadIssues = true;
    console.error("ERROR: The following LanguageCode values are missing from TRANSLATION_PRIORITY:");
    for (const code of toSortedArray(missingInPriority)) {
      console.error(`  - ${code}`);
    }
  } else {
    console.log("OK: All LanguageCode values are present in TRANSLATION_PRIORITY.");
  }

  // 2) Every TRANSLATION_PRIORITY code must be a valid LanguageCode
  const unknownPriorityCodes = difference(priorityCodes, languageCodes);
  if (unknownPriorityCodes.size > 0) {
    hadIssues = true;
    console.error("ERROR: The following TRANSLATION_PRIORITY codes are not present in LanguageCode:");
    for (const code of toSortedArray(unknownPriorityCodes)) {
      console.error(`  - ${code}`);
    }
  } else {
    console.log("OK: All TRANSLATION_PRIORITY codes are valid LanguageCode values.");
  }

  // 3) REMAINING_LANGUAGES must be subset of LanguageCode
  const remainingNotInLanguageCode = difference(remainingLanguages, languageCodes);
  if (remainingNotInLanguageCode.size > 0) {
    hadIssues = true;
    console.error("ERROR: The following REMAINING_LANGUAGES entries are not in LanguageCode:");
    for (const code of toSortedArray(remainingNotInLanguageCode)) {
      console.error(`  - ${code}`);
    }
  } else {
    console.log("OK: All REMAINING_LANGUAGES are valid LanguageCode values.");
  }

  // 4) REMAINING_LANGUAGES must be disjoint from SUPPORTED_LANGUAGES
  const remainingThatAreSupported = difference(remainingLanguages, difference(remainingLanguages, supportedLanguages));
  // Equivalent to intersection(remainingLanguages, supportedLanguages)
  if (remainingThatAreSupported.size > 0) {
    hadIssues = true;
    console.error("ERROR: The following REMAINING_LANGUAGES are already in SUPPORTED_LANGUAGES:");
    for (const code of toSortedArray(remainingThatAreSupported)) {
      console.error(`  - ${code}`);
    }
  } else {
    console.log("OK: REMAINING_LANGUAGES does not include any SUPPORTED_LANGUAGES.");
  }

  console.log("");

  if (hadIssues) {
    console.error("i18n metadata check completed with inconsistencies.");
    process.exitCode = 1;
  } else {
    console.log("i18n metadata check completed successfully. All metadata is consistent.");
    process.exitCode = 0;
  }
}

try {
  main();
} catch (err) {
  console.error("Unexpected error while running i18n metadata check:", err);
  process.exitCode = 1;
}
