//! Language and script detection for routing decision requests (#339).
//!
//! A port of `laya.lang` from the Laya SDK, the reference for which
//! Laya checkpoint may read a state. Laya's English checkpoint is a
//! ModernBERT with an English BPE vocabulary: on non-Latin scripts it is
//! near random *and still confident* (0.000 accuracy at 0.952 confidence
//! on Khmer), so a state it cannot read must go to the multilingual
//! checkpoint instead. The question this module answers is exactly that
//! one — "can the English checkpoint read this?" — with the language
//! guess and the evidence behind it for the response's `routing` block.
//!
//! The heuristics are the reference's, rule for rule: script counts over
//! alphabetic characters, a function-word vote with margins for
//! Latin-script text, a diacritic-rate fallback for Latin languages with
//! no word list, and a line-by-line / field-by-field rescan so a long
//! English part cannot outvote a short foreign message.
//!
//! # Matching Python's Unicode semantics
//!
//! The reference leans on Python's string and `re` semantics, and a port
//! that reads the same but uses Rust's defaults routes differently. Each
//! divergence is replicated explicitly:
//!
//! - `str.isalpha` is General_Category `L*` — not Rust's
//!   `char::is_alphabetic`, which also admits combining marks (so a
//!   Devanagari vowel sign would count as a letter) and letter numbers.
//! - `re`'s `\w` is "alphanumeric or underscore", where alphanumeric is
//!   `L*` or `N*`; `\d` is `Nd`. So the reference's word class
//!   `[^\W\d_]` is `L*` ∪ `Nl` ∪ `No` — letters and letter-like numerals
//!   (Ⅲ, ², ½), but never a combining mark. Rust's `regex` `\w` includes
//!   marks and would glue `मुझे` into one word where Python sees three.
//! - `str.split()` / `str.strip()` treat U+001C..U+001F as whitespace;
//!   `char::is_whitespace` does not.
//! - `unicodedata.combining` is the canonical combining class.
//! - `str.isupper` on a string means "no lowercase or titlecase
//!   character, and at least one uppercase one".
//! - Lengths, slices and budgets count code points, not bytes.
//! - `round(x, 4)` rounds the exact binary value half-to-even; Rust's
//!   fixed-precision formatting does the same.
//!
//! The reference's lookbehind regex for identifiers has no equivalent in
//! the `regex` crate and is a hand-written scanner here.
//!
//! One thing this module cannot fix on its own: for a JSON object state,
//! the order of its values decides which text is read first (and so what
//! falls inside the 4000-character window). Python keeps insertion order;
//! a `serde_json::Value` built without the `preserve_order` feature
//! iterates keys sorted. The callers' JSON must keep order for results to
//! match the reference on multi-field states.

use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::Value;
use unicode_normalization::char::canonical_combining_class;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// Characters of state read for detection (the reference's `max_chars`).
pub const MAX_CHARS: usize = 4000;

/// Nesting depth past which a state's leaves are not read.
const MAX_DEPTH: usize = 6;

/// A diacritic rate above this is evidence the text is not English, even
/// when no function-word list matches it.
const NON_EN_DIACRITIC_RATE: f64 = 0.02;

/// Above this rate, English function words cannot rescue text that has
/// non-English letters.
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;

/// Share of non-Latin letters that makes a Latin-plurality text non-Latin.
const NON_LATIN_FRACTION: f64 = 0.2;
/// A smaller share also counts once it amounts to this many letters.
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
const NON_LATIN_MIN_LETTERS: f64 = 10.0;

/// The reference's detection result, serialised with the same field
/// names and order as `laya.lang.analyse`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Analysis {
    /// Dominant script: `latin`, a named script (`han`, `devanagari`, …),
    /// `other` for letters no range claims, or `unknown` for no letters.
    pub script: &'static str,
    /// Fraction of alphabetic characters per script, Latin first, then
    /// in order of first appearance.
    #[serde(serialize_with = "serialize_profile")]
    pub script_profile: Vec<(&'static str, f64)>,
    /// Best-effort language code for Latin-script text (`en`, `fr`, …).
    pub language: Option<&'static str>,
    /// Whether the English checkpoint can be expected to read the state.
    pub is_english: bool,
    /// True when no language was named — which is not the same as English.
    pub language_undecided: bool,
    /// Share of characters that are letters ordinary English does not
    /// use, rounded to four places.
    pub diacritic_rate: f64,
    /// Share of letters that are not Latin, rounded to four places.
    pub non_latin_fraction: f64,
    /// The line or field that made a mostly English state non-English.
    pub mixed_segment: Option<String>,
}

fn serialize_profile<S: serde::Serializer>(
    profile: &[(&'static str, f64)],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let mut map = serializer.serialize_map(Some(profile.len()))?;
    for (k, v) in profile {
        map.serialize_entry(k, v)?;
    }
    map.end()
}

/// Full detection result for a state (`laya.lang.analyse`).
///
/// String values are what get read; object keys are ignored (they are
/// usually English field names). When a state has several strings, one
/// non-English value is enough to make it non-English.
pub fn analyse(state: &Value) -> Analysis {
    let leaves = iter_text(state);
    let mut result = analyse_text(&state_text(&leaves, MAX_CHARS));
    if result.script == "latin" && result.is_english {
        // A single line has no other part to be outvoted by, and was just
        // read whole.
        if (leaves.len() > 1 || leaves.iter().any(|l| l.contains('\n')))
            && let Some((lang, mixed)) = non_english_segment(&leaves, MAX_CHARS)
        {
            result.language = Some(lang);
            result.is_english = false;
            result.language_undecided = false;
            result.mixed_segment = Some(mixed);
        }
    }
    // A plain string was just read whole. A structured state can still
    // hide a message past the segment cap, or in a script the word lists
    // do not name.
    if matches!(state, Value::String(_) | Value::Null) || !result.is_english {
        return result;
    }
    let mut best_n: i64 = -1;
    let mut best: Option<Analysis> = None;
    for leaf in &leaves {
        let Some(det) = leaf_non_english(leaf) else {
            continue;
        };
        let n_alpha = leaf
            .chars()
            .take(MAX_CHARS)
            .filter(|&c| is_alpha(c))
            .count() as i64;
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    if let Some(best) = best {
        result.language = best.language;
        result.is_english = false;
        result.language_undecided = best.language_undecided;
    }
    result
}

/// True when the English checkpoint can be expected to read this state.
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
}

// ── character classes (Python semantics) ─────────────────────────────

/// `str.isalpha`: General_Category L*.
fn is_alpha(c: char) -> bool {
    use GeneralCategory::*;
    matches!(
        c.general_category(),
        UppercaseLetter | LowercaseLetter | TitlecaseLetter | ModifierLetter | OtherLetter
    )
}

/// `re` `[^\W\d_]`: letters and letter-like numerals, never a mark.
fn is_word_letter(c: char) -> bool {
    is_alpha(c)
        || matches!(
            c.general_category(),
            GeneralCategory::LetterNumber | GeneralCategory::OtherNumber
        )
}

/// `re` `[^\W_]`: letters and numbers of every kind.
fn is_alnum(c: char) -> bool {
    is_word_letter(c) || c.general_category() == GeneralCategory::DecimalNumber
}

/// `re` `\w`.
fn is_word(c: char) -> bool {
    c == '_' || is_alnum(c)
}

/// `str.isspace`: Rust's White_Space plus the information separators.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn is_title(c: char) -> bool {
    c.general_category() == GeneralCategory::TitlecaseLetter
}

/// `str.isupper`: no lowercase or titlecase character, and at least one
/// uppercase one.
fn py_isupper(chars: &[char]) -> bool {
    let mut cased = false;
    for &c in chars {
        if c.is_lowercase() || is_title(c) {
            return false;
        }
        cased |= c.is_uppercase();
    }
    cased
}

fn py_strip_is_empty(chars: &[char]) -> bool {
    chars.iter().all(|&c| is_py_space(c))
}

fn py_strip(chars: &[char]) -> String {
    let start = chars.iter().position(|&c| !is_py_space(c));
    let end = chars.iter().rposition(|&c| !is_py_space(c));
    match (start, end) {
        (Some(s), Some(e)) => chars[s..=e].iter().collect(),
        _ => String::new(),
    }
}

/// `str.split()` with no separator.
fn py_split_whitespace(chars: &[char]) -> Vec<&[char]> {
    chars
        .split(|&c| is_py_space(c))
        .filter(|t| !t.is_empty())
        .collect()
}

/// Maximal runs of characters matching `pred` (`re.findall` of `[...]+`).
fn runs(chars: &[char], pred: impl Fn(char) -> bool) -> Vec<&[char]> {
    chars
        .split(|&c| !pred(c))
        .filter(|r| !r.is_empty())
        .collect()
}

/// `round(x, 4)`: half-to-even on the exact binary value, as Python.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

fn chars_of(s: &str) -> Vec<char> {
    s.chars().collect()
}

// ── flattening a state ───────────────────────────────────────────────

/// The string leaves of a state, in order (`_iter_text`).
fn iter_text(state: &Value) -> Vec<String> {
    let mut out = Vec::new();
    collect_leaves(state, 0, &mut out);
    out
}

fn collect_leaves(v: &Value, depth: usize, out: &mut Vec<String>) {
    if depth > MAX_DEPTH {
        return;
    }
    match v {
        Value::String(s) => out.push(s.clone()),
        Value::Object(m) => m.values().for_each(|x| collect_leaves(x, depth + 1, out)),
        Value::Array(a) => a.iter().for_each(|x| collect_leaves(x, depth + 1, out)),
        _ => {}
    }
}

/// The detection window: leaves joined by spaces, at most `max_chars`
/// characters (`state_text`).
fn state_text(leaves: &[String], max_chars: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut budget = max_chars as i64;
    for leaf in leaves {
        if budget <= 0 {
            break;
        }
        let len = leaf.chars().count() as i64;
        if len > budget {
            parts.push(leaf.chars().take(budget as usize).collect());
            break;
        }
        parts.push(leaf.clone());
        budget -= len + 1;
    }
    parts.join(" ").chars().take(max_chars).collect()
}

// ── scripts ──────────────────────────────────────────────────────────

fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi)
}

fn is_latin_cp(cp: u32, below: u32) -> bool {
    cp < below
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
}

/// Alphabetic characters per script, in order of first appearance, with
/// Latin last so a named script wins a tie against it (`_script_counts`).
fn script_counts(text: &[char]) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut latin = 0;
    for &c in text {
        if !is_alpha(c) {
            continue;
        }
        let cp = c as u32;
        // Latin, IPA Extensions, Latin Extended Additional, fullwidth.
        if is_latin_cp(cp, 0x02B0) {
            latin += 1;
            continue;
        }
        let name = SCRIPT_RANGES
            .iter()
            .find(|(_, r)| in_ranges(cp, r))
            .map(|(n, _)| *n)
            .unwrap_or("other");
        match counts.iter_mut().find(|(n, _)| *n == name) {
            Some((_, k)) => *k += 1,
            None => counts.push((name, 1)),
        }
    }
    counts.push(("latin", latin));
    counts
}

/// First key with the largest count, `unknown` when there are no letters.
fn script_from_counts(counts: &[(&'static str, usize)]) -> &'static str {
    let mut best: Option<(&'static str, usize)> = None;
    for &(name, k) in counts {
        if best.is_none_or(|(_, b)| k > b) {
            best = Some((name, k));
        }
    }
    match best {
        Some((name, k)) if k > 0 => name,
        _ => "unknown",
    }
}

fn profile_from_counts(counts: &[(&'static str, usize)]) -> Vec<(&'static str, f64)> {
    let total: usize = counts.iter().map(|(_, k)| k).sum();
    if total == 0 {
        return Vec::new();
    }
    let latin = counts.iter().find(|(n, _)| *n == "latin").map(|(_, k)| *k);
    let ordered = latin
        .filter(|&k| k > 0)
        .map(|k| ("latin", k))
        .into_iter()
        .chain(counts.iter().copied().filter(|(n, _)| *n != "latin"));
    ordered
        .filter(|&(_, k)| k > 0)
        .map(|(n, k)| (n, k as f64 / total as f64))
        .collect()
}

/// The named non-Latin script of one letter, else `None` (`_script_of`).
/// Latin here stops at U+024F: IPA letters are claimed by no script.
fn script_of(c: char) -> Option<&'static str> {
    let cp = c as u32;
    if is_latin_cp(cp, 0x0250) {
        return None;
    }
    SCRIPT_RANGES
        .iter()
        .find(|(_, r)| in_ranges(cp, r))
        .map(|(n, _)| *n)
}

/// Non-Latin runs that read as words rather than as a symbol, a proper
/// name or a pronunciation inside English prose (`_non_latin_words`).
fn non_latin_words(text: &[char]) -> Vec<Vec<char>> {
    let mut out: Vec<Vec<char>> = Vec::new();
    let mut cur: Vec<char> = Vec::new();
    let mut script: Option<&'static str> = None;
    for &c in text {
        // A combining mark belongs to the letter before it and never
        // splits a word.
        if canonical_combining_class(c) != 0 {
            continue;
        }
        let s = script_of(c);
        if s.is_some() && s == script {
            cur.push(c);
            continue;
        }
        if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        match s {
            Some(_) => {
                cur = vec![c];
                script = s;
            }
            None => {
                cur.clear();
                script = None;
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|w| w.len() >= 2 && !w[0].is_uppercase());
    out
}

// ── function words ───────────────────────────────────────────────────

fn in_list(list: &[&str], w: &str) -> bool {
    list.binary_search(&w).is_ok()
}

fn stop_list(lang: &str) -> &'static [&'static str] {
    STOP.iter()
        .find(|(l, _)| *l == lang)
        .map(|(_, w)| *w)
        .unwrap_or(&[])
}

/// A word more than one list claims names no particular language.
fn is_shared(w: &str) -> bool {
    STOP.iter().filter(|(_, list)| in_list(list, w)).count() > 1
}

/// An English function word no other list holds.
fn is_en_only(w: &str) -> bool {
    in_list(stop_list("en"), w) && !is_shared(w)
}

fn is_non_en_diacritic(c: char) -> bool {
    NON_EN_DIACRITICS.contains(c)
}

/// `_IDENTIFIER.sub(" ", text)`: replace every `[\w-]*(?:[.@][\w-]+)+`
/// not preceded by `[\w-]` with a space. Links, e-mail addresses and
/// dotted names split into pieces that collide with function words.
fn strip_identifiers(text: &[char]) -> Vec<char> {
    let ident = |c: char| is_word(c) || c == '-';
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if i == 0 || !ident(text[i - 1]) {
            let mut j = i;
            while j < text.len() && ident(text[j]) {
                j += 1;
            }
            let mut end = None;
            while j + 1 < text.len() && matches!(text[j], '.' | '@') && ident(text[j + 1]) {
                j += 1;
                while j < text.len() && ident(text[j]) {
                    j += 1;
                }
                end = Some(j);
            }
            if let Some(e) = end {
                out.push(' ');
                i = e;
                continue;
            }
        }
        out.push(text[i]);
        i += 1;
    }
    out
}

/// Evidence behind the Latin-script language guess (`latin_profile`).
struct LatinProfile {
    language: Option<&'static str>,
    diacritic_rate: f64,
    looks_non_english: bool,
}

fn latin_profile(text: &[char]) -> LatinProfile {
    // 'İ'.lower() is 'i' + a combining dot, which matches no word list.
    let stripped: String = strip_identifiers(text)
        .into_iter()
        .map(|c| if c == 'İ' { 'i' } else { c })
        .collect::<String>()
        .to_lowercase();
    let stripped = chars_of(&stripped);
    let words: Vec<String> = runs(&stripped, is_word_letter)
        .into_iter()
        .map(|w| w.iter().collect())
        .collect();
    let lowered = chars_of(&text.iter().collect::<String>().to_lowercase());
    let diac = lowered.iter().filter(|&&c| is_non_en_diacritic(c)).count();
    let diacritic_rate = diac as f64 / lowered.len().max(1) as f64;
    let looks_non_english = diacritic_rate >= NON_EN_DIACRITIC_RATE;
    let mut profile = LatinProfile {
        language: None,
        diacritic_rate,
        looks_non_english,
    };
    if words.len() < 4 {
        return profile;
    }
    let mut distinct: Vec<&str> = words.iter().map(String::as_str).collect();
    distinct.sort_unstable();
    distinct.dedup();

    let score = |list: &[&str]| words.iter().filter(|w| in_list(list, w)).count();
    let en = score(stop_list("en"));
    // Only a language that matched a word no other list claims may be
    // named; the first of equal scores wins, in list order.
    let mut best: Option<(&'static str, usize)> = None;
    for (lg, list) in STOP.iter().filter(|(lg, _)| *lg != "en") {
        let evidenced = distinct.iter().any(|w| in_list(list, w) && !is_shared(w));
        if !evidenced {
            continue;
        }
        let s = score(list);
        if best.is_none_or(|(_, b)| s > b) {
            best = Some((lg, s));
        }
    }
    profile.language = match best {
        // A non-English language needs a clear margin over English.
        Some((lg, s)) if s >= 2.max(en + 2) => Some(lg),
        // Or two hits plus the diacritics.
        Some((lg, s)) if looks_non_english && s >= 2.max(en) => Some(lg),
        _ if en > 0 && (!looks_non_english || english_rescued(&distinct, diacritic_rate)) => {
            Some("en")
        }
        _ => None,
    };
    profile
}

/// Whether plain-English function words outvote a marginal diacritic
/// rate: two distinct English-only words, at most one word carrying a
/// non-English letter, and a rate below the rescue ceiling.
fn english_rescued(distinct: &[&str], diacritic_rate: f64) -> bool {
    if diacritic_rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    if distinct.iter().filter(|w| is_en_only(w)).count() < 2 {
        return false;
    }
    distinct
        .iter()
        .filter(|w| w.chars().any(is_non_en_diacritic))
        .count()
        <= 1
}

// ── segments ─────────────────────────────────────────────────────────

/// A line carrying code syntax: `=`, `;`, braces, brackets or a call
/// `name(` (`_CODE_LINE`).
fn is_code_line(text: &[char]) -> bool {
    text.iter().any(|c| "=;{}[]".contains(*c))
        || text.windows(2).any(|w| is_word(w[0]) && w[1] == '(')
}

/// An identifier or compound: alnum, joiner, alnum (`_JOINED`).
fn is_joined(token: &[char]) -> bool {
    token
        .windows(3)
        .any(|w| is_alnum(w[0]) && "._/\\".contains(w[1]) && is_alnum(w[2]))
}

/// Language code for one non-code line, or `None` when it does not name
/// a foreign language (`_named_prose_language`).
fn named_prose_language(segment: &[char]) -> Option<&'static str> {
    if py_strip_is_empty(segment) || is_code_line(segment) {
        return None;
    }
    let tokens: Vec<&[char]> = py_split_whitespace(segment)
        .into_iter()
        .filter(|t| !is_joined(t))
        .collect();
    let mut prose: Vec<char> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            prose.push(' ');
        }
        prose.extend_from_slice(t);
    }
    // An all-caps token inside mixed-case text is an acronym or a code.
    if prose.iter().any(|c| c.is_lowercase()) {
        prose = blank_uppercase_runs(&prose);
    }
    let words = runs(&prose, is_word_letter);
    if words.len() < 4 {
        return None;
    }
    let lang = latin_profile(&prose).language?;
    if lang == "en" {
        return None;
    }
    let list = stop_list(lang);
    let mut hits: Vec<String> = words
        .iter()
        .map(|w| w.iter().collect::<String>().to_lowercase())
        .filter(|w| in_list(list, w))
        .collect();
    hits.sort_unstable();
    hits.dedup();
    (hits.len() >= 2).then_some(lang)
}

/// `_LETTER_RUN.sub(...)`: runs of two or more word letters that are all
/// capitals become a space.
fn blank_uppercase_runs(text: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if is_word_letter(text[i]) {
            let mut j = i;
            while j < text.len() && is_word_letter(text[j]) {
                j += 1;
            }
            let run = &text[i..j];
            if run.len() >= 2 && py_isupper(run) {
                out.push(' ');
            } else {
                out.extend_from_slice(run);
            }
            i = j;
        } else {
            out.push(text[i]);
            i += 1;
        }
    }
    out
}

/// The first line or field that, read on its own, names a non-English
/// language, reading at most `max_chars` characters in all
/// (`_non_english_segment`).
fn non_english_segment(leaves: &[String], max_chars: usize) -> Option<(&'static str, String)> {
    let mut seen = 0usize;
    for leaf in leaves {
        for seg in leaf.split('\n') {
            if seen >= max_chars {
                return None;
            }
            let seg: Vec<char> = seg.chars().take(max_chars - seen).collect();
            seen += seg.len();
            if let Some(lang) = named_prose_language(&seg) {
                return Some((lang, py_strip(&seg)));
            }
        }
    }
    None
}

// ── one string ───────────────────────────────────────────────────────

/// Detection for one flattened string (`_analyse_text`).
fn analyse_text(text: &str) -> Analysis {
    let chars = chars_of(text);
    let counts = script_counts(&chars);
    let prof = profile_from_counts(&counts);
    let mut script = script_from_counts(&counts);
    let latin_share = prof
        .iter()
        .find(|(n, _)| *n == "latin")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        round4(1.0 - latin_share)
    };
    let n_alpha = chars.iter().filter(|&&c| is_alpha(c)).count() as f64;
    let n_non_latin = (non_latin * n_alpha).round_ties_even();
    if script == "latin"
        && !non_latin_words(&chars).is_empty()
        && (non_latin >= NON_LATIN_FRACTION
            || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS))
    {
        let mut best: Option<(&'static str, f64)> = None;
        for &(n, v) in prof.iter().filter(|(n, _)| *n != "latin") {
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((n, v));
            }
        }
        if let Some((n, _)) = best {
            script = n;
        }
    }
    let base = Analysis {
        script,
        script_profile: prof,
        language: None,
        is_english: true,
        language_undecided: true,
        diacritic_rate: 0.0,
        non_latin_fraction: 0.0,
        mixed_segment: None,
    };
    if script == "unknown" {
        return base;
    }
    if script != "latin" {
        return Analysis {
            is_english: false,
            non_latin_fraction: non_latin,
            ..base
        };
    }
    let lat = latin_profile(&chars);
    // Undecided is not English: with nothing naming the language,
    // non-English letters are enough to prefer the multilingual checkpoint.
    let undecided = lat.language.is_none();
    Analysis {
        language: lat.language,
        is_english: lat.language == Some("en") || (undecided && !lat.looks_non_english),
        language_undecided: undecided,
        diacritic_rate: round4(lat.diacritic_rate),
        non_latin_fraction: non_latin,
        ..base
    }
}

/// A string value that is itself not safe for the English checkpoint,
/// read line by line, each capped at `MAX_CHARS` (`_leaf_non_english`).
fn leaf_non_english(leaf: &str) -> Option<Analysis> {
    let mut best_n: i64 = -1;
    let mut best: Option<Analysis> = None;
    for line in leaf.split('\n') {
        // Too short to reach four words or ten letters.
        if line.chars().count() < 7 {
            continue;
        }
        let sample: Vec<char> = line.chars().take(MAX_CHARS).collect();
        if py_strip_is_empty(&sample) || is_code_line(&sample) {
            continue;
        }
        let text: String = sample.iter().collect();
        let det = analyse_text(&text);
        if det.is_english {
            continue;
        }
        let n_alpha = sample.iter().filter(|&&c| is_alpha(c)).count();
        if !matches!(det.language, None | Some("en")) {
            if named_prose_language(&sample).is_none() {
                continue;
            }
        } else if !matches!(det.script, "latin" | "unknown") {
            if non_latin_words(&sample).is_empty() || (n_alpha as f64) < NON_LATIN_MIN_LETTERS {
                continue;
            }
        } else if !(det.language_undecided
            && det.diacritic_rate >= NON_EN_DIACRITIC_RATE
            && runs(&sample, is_word_letter).len() >= 4)
        {
            continue;
        }
        if n_alpha as i64 > best_n {
            best_n = n_alpha as i64;
            best = Some(det);
        }
    }
    best
}

// ── tables ───────────────────────────────────────────────────────────

// Tables generated from laya 0.3.21 `laya/lang.py`; keep them in step with it.

/// Unicode blocks per named non-Latin script, in the reference's order.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    (
        "cyrillic",
        &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    ),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    (
        "hangul",
        &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)],
    ),
    (
        "kana",
        &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    ),
    (
        "han",
        &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)],
    ),
];

/// Function-word lists, in the reference's order (which breaks score
/// ties). Each list is sorted for binary search.
const STOP: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "and", "are", "as", "at", "be", "but", "can", "for", "from", "has", "have", "i", "in",
            "is", "it", "not", "of", "on", "please", "that", "the", "their", "there", "this", "to",
            "was", "we", "were", "what", "which", "will", "with", "would", "you",
        ],
    ),
    (
        "fr",
        &[
            "alors", "au", "aux", "avec", "bien", "bonjour", "ce", "ces", "cette", "comment",
            "dans", "des", "deux", "dois", "doit", "donc", "du", "elle", "elles", "est", "et",
            "fait", "fois", "il", "ils", "je", "jour", "jours", "la", "le", "les", "ma", "mais",
            "merci", "mes", "mois", "mon", "nous", "ont", "ou", "pas", "peut", "peux", "plus",
            "pour", "pourquoi", "quand", "que", "qui", "sa", "ses", "sont", "sur", "ta", "tes",
            "ton", "tous", "tout", "toute", "trois", "très", "tu", "une", "veut", "veux", "vous",
            "être",
        ],
    ),
    (
        "de",
        &[
            "aber", "auch", "auf", "aus", "bei", "bitte", "das", "dem", "den", "der", "dich",
            "die", "diese", "diesen", "dieser", "dieses", "dir", "ein", "eine", "einem", "einen",
            "einer", "für", "gibt", "habe", "haben", "heute", "ich", "im", "in", "ist", "jetzt",
            "kann", "kannst", "mein", "meine", "meinem", "meinen", "meiner", "mich", "mir", "mit",
            "nach", "nicht", "noch", "oder", "sich", "sind", "und", "uns", "von", "wann", "was",
            "welche", "werden", "wie", "wir", "wird", "wo", "wurde", "zu", "zum", "zur",
        ],
    ),
    (
        "es",
        &[
            "al", "algo", "aquí", "aunque", "como", "con", "cuando", "del", "donde", "dos", "el",
            "entre", "es", "esa", "ese", "eso", "esta", "este", "esto", "está", "fue", "fueron",
            "gracias", "han", "hay", "hemos", "hoy", "la", "las", "le", "les", "lo", "los", "mi",
            "muy", "más", "nada", "necesito", "ni", "nos", "para", "pero", "por", "porque",
            "puede", "pueden", "que", "quiero", "se", "ser", "sobre", "son", "su", "sus",
            "también", "tengo", "tiene", "tienen", "todo", "tres", "tu", "un", "una", "y", "ya",
        ],
    ),
    (
        "pt",
        &[
            "agora", "ainda", "alguem", "alguém", "ali", "antes", "ao", "aos", "aqui", "as", "até",
            "boa", "cadê", "com", "como", "consigo", "da", "das", "depois", "deu", "do", "dois",
            "dos", "e", "em", "entao", "então", "era", "esta", "estamos", "estava", "este",
            "estou", "está", "eu", "ficou", "fiz", "foi", "gostaria", "hoje", "isso", "isto", "ja",
            "já", "mais", "mas", "meu", "meus", "minha", "minhas", "muito", "na", "nada", "nao",
            "nas", "nenhum", "nenhuma", "ninguem", "ninguém", "noite", "nos", "nossa", "nosso",
            "não", "o", "obrigada", "obrigado", "olá", "onde", "ontem", "os", "para", "pela",
            "pelo", "pode", "podem", "por", "porque", "pra", "preciso", "quando", "que", "quero",
            "sao", "se", "ser", "seu", "sou", "sua", "são", "tambem", "também", "tarde", "tem",
            "tenho", "três", "tudo", "tá", "um", "uma", "vc", "vcs", "voce", "voces", "você",
            "vocês", "é",
        ],
    ),
    (
        "it",
        &[
            "abbiamo", "adesso", "agli", "alla", "alle", "anche", "ancora", "avete", "che", "ci",
            "ciao", "col", "come", "con", "da", "dagli", "dal", "dalla", "dallo", "degli", "dei",
            "del", "della", "delle", "dello", "deve", "devo", "devono", "di", "dove", "e", "ed",
            "era", "fra", "già", "gli", "grazie", "ha", "hai", "hanno", "ho", "ieri", "il", "la",
            "le", "lo", "mai", "mi", "mia", "mio", "molto", "ne", "negli", "nel", "nell", "nella",
            "non", "o", "oggi", "per", "perche", "più", "poco", "quando", "questa", "questo",
            "scusa", "sempre", "si", "sono", "stata", "stato", "su", "sua", "sul", "sulla",
            "sulle", "tra", "tuo", "un", "una", "uno", "voglio", "vorrei", "è",
        ],
    ),
    (
        "nl",
        &[
            "aan", "dat", "deze", "door", "een", "het", "is", "maar", "met", "naar", "niet", "ook",
            "op", "te", "van", "voor", "worden", "wordt", "zijn",
        ],
    ),
    (
        "ro",
        &[
            "aceasta", "această", "acest", "acesta", "acum", "ale", "care", "dar", "din", "după",
            "este", "foarte", "fost", "fără", "lui", "mi", "nu", "pentru", "până", "sunt", "să",
            "trebuie", "vreau", "vă", "în", "și", "ți",
        ],
    ),
    (
        "bn",
        &[
            "abar",
            "ajke",
            "akhon",
            "amader",
            "amake",
            "amar",
            "ami",
            "amra",
            "apnake",
            "apnar",
            "apnara",
            "apni",
            "ar",
            "asbe",
            "bhai",
            "bhalo",
            "bolte",
            "bolun",
            "chai",
            "chaina",
            "dhonnobad",
            "dilam",
            "dite",
            "diye",
            "diyechi",
            "dorkar",
            "duibar",
            "ei",
            "eita",
            "ekbar",
            "ekhon",
            "ekhono",
            "ekta",
            "ferot",
            "geche",
            "gese",
            "hobe",
            "hocche",
            "hoise",
            "hoye",
            "hoyeche",
            "hoyni",
            "jabe",
            "jodi",
            "jonno",
            "kalke",
            "keno",
            "keu",
            "kharap",
            "khub",
            "ki",
            "kibhabe",
            "kichu",
            "kintu",
            "kivabe",
            "kobe",
            "kokhon",
            "korbo",
            "korchi",
            "kore",
            "koreche",
            "korechi",
            "koren",
            "korlam",
            "korsi",
            "korte",
            "korun",
            "kothay",
            "koto",
            "lagbe",
            "moddhe",
            "nai",
            "niye",
            "oi",
            "oita",
            "onek",
            "ota",
            "pabo",
            "paini",
            "parben",
            "parbo",
            "parchi",
            "parchina",
            "peyechi",
            "sathe",
            "shathe",
            "shob",
            "shomossa",
            "somossa",
            "tader",
            "tahole",
            "taka",
            "theke",
            "tomake",
            "tomar",
            "tomra",
            "tumi",
            "valo",
        ],
    ),
    (
        "az",
        &[
            "amma", "ancaq", "artiq", "artıq", "bir", "biz", "bu", "cox", "daha", "deyil", "də",
            "eger", "görə", "həm", "hər", "ile", "ilə", "isə", "kimi", "lakin", "mən", "nə",
            "olan", "olmasa", "olub", "onlar", "siz", "sonra", "sən", "ucun", "var", "ve", "və",
            "yalniz", "yalnız", "yox", "yoxdur", "çox", "üçün", "əgər",
        ],
    ),
];

/// Letters ordinary English does not use.
const NON_EN_DIACRITICS: &str =
    "ßàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿāăąćčďđēęěğģīıķļłńņňőœřśşšţťūůűźżžșțə";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `laya.lang.analyse` over the corpus `script/laya-lang-reference.py`
    /// records: every script range, each function-word language accented
    /// and stripped, code and identifiers, multi-line and structured
    /// states, the character budgets, Unicode edge cases and a seeded fuzz.
    const CORPUS: &str = include_str!("testdata/laya/lang_corpus.json");

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    /// Every field of one analysis against the reference, as a list of
    /// human-readable mismatches.
    fn diff(case: &Value, got: &Analysis) -> Vec<String> {
        let want = &case["analysis"];
        let mut out = Vec::new();
        if want["script"] != got.script {
            out.push(format!("script {} != {}", got.script, want["script"]));
        }
        if want["language"].as_str() != got.language {
            out.push(format!(
                "language {:?} != {}",
                got.language, want["language"]
            ));
        }
        for (k, g) in [
            ("is_english", got.is_english),
            ("language_undecided", got.language_undecided),
        ] {
            if want[k].as_bool() != Some(g) {
                out.push(format!("{k} {g} != {}", want[k]));
            }
        }
        for (k, g) in [
            ("diacritic_rate", got.diacritic_rate),
            ("non_latin_fraction", got.non_latin_fraction),
        ] {
            if !close(want[k].as_f64().unwrap_or(f64::NAN), g) {
                out.push(format!("{k} {g} != {}", want[k]));
            }
        }
        if want["mixed_segment"].as_str() != got.mixed_segment.as_deref() {
            out.push(format!(
                "mixed_segment {:?} != {}",
                got.mixed_segment, want["mixed_segment"]
            ));
        }
        let order: Vec<&str> = case["profile_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let got_order: Vec<&str> = got.script_profile.iter().map(|(k, _)| *k).collect();
        if order != got_order {
            out.push(format!("profile order {got_order:?} != {order:?}"));
        }
        for (k, v) in &got.script_profile {
            if !want["script_profile"][*k]
                .as_f64()
                .is_some_and(|w| close(w, *v))
            {
                out.push(format!(
                    "profile[{k}] {v} != {}",
                    want["script_profile"][*k]
                ));
            }
        }
        out
    }

    #[test]
    fn matches_the_reference_on_the_corpus() {
        let corpus: Value = serde_json::from_str(CORPUS).expect("corpus parses");
        let cases = corpus["cases"].as_array().expect("cases");
        assert!(cases.len() > 800, "corpus unexpectedly small");
        let mut failures = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            let state = &case["state"];
            let leaves: Vec<Value> = iter_text(state).into_iter().map(Value::String).collect();
            if case["leaves"].as_array() != Some(&leaves) {
                failures.push(format!("#{i} leaves differ for {state}"));
                continue;
            }
            if let (Some(want), Some(s)) = (case["words"].as_array(), state.as_str()) {
                let chars = chars_of(s);
                let got: Vec<Value> = runs(&chars, is_word_letter)
                    .into_iter()
                    .map(|w| Value::String(w.iter().collect()))
                    .collect();
                if *want != got {
                    failures.push(format!("#{i} words differ for {state}"));
                }
            }
            let d = diff(case, &analyse(state));
            if !d.is_empty() {
                failures.push(format!("#{i} {state}: {}", d.join("; ")));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases differ from the reference:\n{}",
            failures.len(),
            cases.len(),
            failures
                .iter()
                .take(25)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    #[test]
    fn serialises_with_the_reference_field_order() {
        let a = analyse(&json!(
            "Please refund the duplicate charge, it was billed twice."
        ));
        let s = serde_json::to_string(&a).unwrap();
        let keys = [
            "\"script\"",
            "\"script_profile\"",
            "\"language\"",
            "\"is_english\"",
            "\"language_undecided\"",
            "\"diacritic_rate\"",
            "\"non_latin_fraction\"",
            "\"mixed_segment\"",
        ];
        let pos: Vec<usize> = keys.iter().map(|k| s.find(k).unwrap()).collect();
        assert!(pos.windows(2).all(|w| w[0] < w[1]), "{s}");
    }

    #[test]
    fn round4_is_half_to_even_on_the_binary_value() {
        // Exact binary ties go to even, as Python's round(x, 4).
        assert_eq!(round4(0.03125), 0.0312);
        assert_eq!(round4(0.09375), 0.0938);
        assert_eq!(round4(0.15625), 0.1562);
        // 0.00005 is slightly above the tie in binary.
        assert_eq!(round4(0.00005), 0.0001);
    }

    #[test]
    fn python_whitespace_includes_the_information_separators() {
        let chars = chars_of("a\u{1c}b\u{1f}c d");
        assert_eq!(py_split_whitespace(&chars).len(), 4);
    }

    #[test]
    fn combining_marks_split_words_like_python() {
        // Devanagari vowel signs are marks: Python's word class stops at
        // them: `मुझे` is म + vowel sign + झ + vowel sign, and Python's
        // `re.findall(r"[^\W\d_]+", "मुझे")` is `['म', 'झ']`.
        let chars = chars_of("मुझे");
        let words: Vec<String> = runs(&chars, is_word_letter)
            .into_iter()
            .map(|w| w.iter().collect())
            .collect();
        assert_eq!(words, ["म", "झ"]);
    }
}
