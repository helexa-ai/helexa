#!/usr/bin/env node

/**
 * Simple i18n key consistency checker.
 *
 * Compares translation JSONs for all configured languages against the English
 * baseline, for each namespace (common, home, chat).
 *
 * Exit codes:
 *  - 0: all good
 *  - 1: inconsistencies found or unexpected error
 */

import fs from "fs";
import path from "path";
import url from "url";

// Adjust these if your i18n structure changes.
const ROOT = path.resolve(
  path.dirname(url.fileURLToPath(import.meta.url)),
  "..",
);
const RESOURCES_DIR = path.join(ROOT, "src", "i18n", "resources");

// Namespaces to validate.
const NAMESPACES = ["common", "mission", "chat"];

// Languages to validate should track SUPPORTED_LANGUAGES in src/i18n/languages.ts.
// NOTE: This list is intentionally narrower than SUPPORTED_LANGUAGES and does not
// enforce that every supported language has wired resources. That enforcement is
// implemented further below by checking the i18n index.
const LANGUAGES = [
  "bg",
  "de",
  "el",
  "en",
  "es",
  "et",
  "fr",
  "it",
  "pt",
  "ro",
  "ru",
];

function readJson(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    return JSON.parse(raw);
  } catch (err) {
    throw new Error(`Failed to read/parse JSON at ${filePath}: ${err.message}`);
  }
}

/**
 * Recursively walk an object and collect all key paths using dot-notation.
 * Arrays are traversed structurally but their indexes are not part of the key path
 * (we only care that the shape exists, not array lengths).
 */
function collectKeyPaths(obj, prefix = "") {
  const keys = new Set();

  if (obj === null || obj === undefined) {
    return keys;
  }

  if (typeof obj !== "object") {
    if (prefix) keys.add(prefix);
    return keys;
  }

  // If this is an array, walk its elements but don't add indices to the path.
  if (Array.isArray(obj)) {
    obj.forEach((item) => {
      for (const childKey of collectKeyPaths(item, prefix)) {
        keys.add(childKey);
      }
    });
    return keys;
  }

  // Plain object
  for (const [k, v] of Object.entries(obj)) {
    const nextPrefix = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object") {
      for (const childKey of collectKeyPaths(v, nextPrefix)) {
        keys.add(childKey);
      }
    } else {
      keys.add(nextPrefix);
    }
  }

  return keys;
}

function diffKeys(baseSet, targetSet) {
  const missing = [];
  const extra = [];

  for (const k of baseSet) {
    if (!targetSet.has(k)) missing.push(k);
  }

  for (const k of targetSet) {
    if (!baseSet.has(k)) extra.push(k);
  }

  missing.sort();
  extra.sort();

  return { missing, extra };
}

function logHeader(title) {
  // Simple console formatting without external deps.
  console.log("\n" + "=".repeat(title.length));
  console.log(title);
  console.log("=".repeat(title.length));
}

function main() {
  let hadIssues = false;

  console.log("helexa.ai i18n key consistency check");
  console.log(`Root: ${ROOT}`);
  console.log(`Resources dir: ${RESOURCES_DIR}`);
  console.log(`Languages: ${LANGUAGES.join(", ")}`);
  console.log(`Namespaces: ${NAMESPACES.join(", ")}`);

  // --- Wiring check: ensure each SUPPORTED_LANGUAGES entry has resources registered ---
  //
  // This prevents cases where a language is:
  //   - present in LanguageCode and SUPPORTED_LANGUAGES
  //   - has translation JSONs under src/i18n/resources/<code>/
  // but is *not* wired into the i18n `resources` object (and thus silently
  // falls back to English at runtime).
  try {
    const languagesTs = fs.readFileSync(
      path.join(ROOT, "src", "i18n", "languages.ts"),
      "utf8",
    );
    const i18nIndexTs = fs.readFileSync(
      path.join(ROOT, "src", "i18n", "index.ts"),
      "utf8",
    );

    // Parse SUPPORTED_LANGUAGES from languages.ts
    const marker = "export const SUPPORTED_LANGUAGES";
    const start = languagesTs.indexOf(marker);
    if (start === -1) {
      throw new Error("Could not find `SUPPORTED_LANGUAGES` in languages.ts");
    }
    const after = languagesTs.slice(start);
    const bracketIndex = after.indexOf("[");
    const closingIndex = after.indexOf("];");
    if (bracketIndex === -1 || closingIndex === -1) {
      throw new Error("Malformed SUPPORTED_LANGUAGES array");
    }
    const arraySlice = after.slice(bracketIndex + 1, closingIndex);
    const supportedFromTs = new Set();
    const codeRegex = /"([^"]+)"/g;
    let m;
    while ((m = codeRegex.exec(arraySlice)) !== null) {
      supportedFromTs.add(m[1]);
    }

    console.log(
      `SUPPORTED_LANGUAGES from TS for wiring check: ${Array.from(
        supportedFromTs,
      )
        .sort()
        .join(", ")}`,
    );

    // Languages that are intentionally "future" and may not yet have
    // resources wired in. Keep them out of the hard failure path so
    // CI does not break while their translations are still pending.
    const FUTURE_LANGUAGES = new Set(["ig", "om", "so", "ti", "wo"]);

    // Parse i18n resources object keys from index.ts.
    // We look for a block like:
    //   const resources: Resource = {
    //     en: { ... },
    //     fr: { ... },
    //   };
    const resourcesMarker = "const resources: Resource = {";
    const resStart = i18nIndexTs.indexOf(resourcesMarker);
    if (resStart === -1) {
      throw new Error("Could not find `const resources: Resource` in index.ts");
    }
    const resAfter = i18nIndexTs.slice(resStart + resourcesMarker.length);
    const resEndIndex = resAfter.indexOf("};");
    if (resEndIndex === -1) {
      throw new Error("Malformed resources object in index.ts");
    }
    const resourcesBlock = resAfter.slice(0, resEndIndex);

    // Extract top-level language keys: lines starting with two spaces then <code>:
    // Example: "  en: {" or "  fr: {"
    const wiredLangs = new Set();
    const lineRegex = /^\s*([a-z]{2}):\s*{\s*$/gm;
    let lm;
    while ((lm = lineRegex.exec(resourcesBlock)) !== null) {
      wiredLangs.add(lm[1]);
    }

    console.log(
      `Languages wired in i18n resources: ${Array.from(wiredLangs)
        .sort()
        .join(", ")}`,
    );

    // Now compare SUPPORTED_LANGUAGES vs wired resources.
    const missingWiring = [];
    for (const code of supportedFromTs) {
      if (FUTURE_LANGUAGES.has(code)) {
        // These are explicitly allowed to be missing until their
        // translations and wiring land.
        continue;
      }
      if (!wiredLangs.has(code)) {
        missingWiring.push(code);
      }
    }

    if (missingWiring.length > 0) {
      hadIssues = true;
      console.error(
        "\nERROR: Some SUPPORTED_LANGUAGES codes are not wired into the i18n `resources` object in src/i18n/index.ts:",
      );
      for (const code of missingWiring.sort()) {
        console.error(`  - ${code}`);
      }
      console.error(
        "These languages will silently fall back to English at runtime. Ensure that:",
      );
      console.error(
        "  1) ./resources/<code>/{common,home,chat}.json exist, and",
      );
      console.error(
        "  2) They are imported and registered in the `resources` object.",
      );
      console.error("");
    } else {
      console.log(
        "OK: Every SUPPORTED_LANGUAGES entry (excluding future languages) is wired into the i18n resources object.",
      );
    }
  } catch (err) {
    hadIssues = true;
    console.error(
      "ERROR: Failed while checking SUPPORTED_LANGUAGES wiring in i18n index:",
      err.message,
    );
  }

  for (const ns of NAMESPACES) {
    logHeader(`Namespace: ${ns}`);

    const basePath = path.join(RESOURCES_DIR, "en", `${ns}.json`);
    let baseJson;
    try {
      baseJson = readJson(basePath);
    } catch (err) {
      console.error(err.message);
      hadIssues = true;
      continue;
    }

    const baseKeys = collectKeyPaths(baseJson);
    console.log(`Baseline (en) keys: ${baseKeys.size}`);

    for (const lang of LANGUAGES) {
      if (lang === "en") continue;

      const langPath = path.join(RESOURCES_DIR, lang, `${ns}.json`);
      if (!fs.existsSync(langPath)) {
        console.error(`  [${lang}] MISSING file: ${langPath}`);
        hadIssues = true;
        continue;
      }

      let langJson;
      try {
        langJson = readJson(langPath);
      } catch (err) {
        console.error(`  [${lang}] ${err.message}`);
        hadIssues = true;
        continue;
      }

      const langKeys = collectKeyPaths(langJson);
      const { missing, extra } = diffKeys(baseKeys, langKeys);

      if (missing.length === 0 && extra.length === 0) {
        console.log(`  [${lang}] OK (keys: ${langKeys.size})`);
      } else {
        hadIssues = true;
        console.log(`  [${lang}] Issues found:`);
        if (missing.length > 0) {
          console.log("    Missing keys (present in en, absent here):");
          for (const k of missing) {
            console.log(`      - ${k}`);
          }
        }
        if (extra.length > 0) {
          console.log("    Extra keys (present here, absent in en):");
          for (const k of extra) {
            console.log(`      + ${k}`);
          }
        }
      }
    }
  }

  console.log("");
  if (hadIssues) {
    console.error("i18n check completed with inconsistencies.");
    process.exitCode = 1;
  } else {
    console.log("i18n check completed successfully. All keys are consistent.");
    process.exitCode = 0;
  }
}

try {
  main();
} catch (err) {
  console.error("Unexpected error while running i18n check:", err);
  process.exitCode = 1;
}
