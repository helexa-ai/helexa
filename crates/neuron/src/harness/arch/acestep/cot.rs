//! The planner's chain of thought, as text (#384).
//!
//! The planner writes its plan as YAML between `<think>` tags. Upstream
//! reads it back with `parse_lm_output`, then writes it out again with
//! `yaml.dump` to build the prompt the audio codes are generated from. Those
//! bytes are model input, so the dump has to be reproduced exactly,
//! including where PyYAML folds a long caption and when it quotes a value.
//!
//! Reference: `acestep/llm_inference.py` (`parse_lm_output`,
//! `_format_metadata_as_cot`) and `acestep/constrained_logits_processor.py`
//! (`postprocess_caption`) at ACE-Step 1.5 `ca1e85fe`; PyYAML's
//! `emitter.py` for the scalar writers.

use std::collections::BTreeMap;

/// A plan field's value as upstream holds it: `int(...)` where the text is
/// a whole number, the text otherwise. The distinction decides how YAML
/// writes it.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    Int(i64),
    Str(String),
}

impl MetaValue {
    pub fn as_text(&self) -> String {
        match self {
            Self::Int(i) => i.to_string(),
            Self::Str(s) => s.clone(),
        }
    }
}

/// The plan's fields, by name.
pub type Metadata = BTreeMap<String, MetaValue>;

/// Fields the codes-phase prompt carries, in the order YAML sorts them.
const COT_FIELDS: [&str; 6] = [
    "bpm",
    "caption",
    "duration",
    "keyscale",
    "language",
    "timesignature",
];

/// Join a YAML-folded caption back into one line: every line trimmed,
/// blank lines dropped, joined with single spaces.
pub fn postprocess_caption(caption: &str) -> String {
    caption
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Python's `str.isdigit` for the ASCII digits upstream's values use.
fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Read a plan out of the planner's output, as `parse_lm_output` does.
///
/// The CoT phase stops before `</think>`, so the reasoning section is the
/// text between the tags when both are present, else everything before the
/// first audio code. A field is a line that does not start with whitespace
/// and has a colon; indented lines continue the previous field.
pub fn parse_plan(output: &str) -> Metadata {
    let reasoning = match (output.find("<think>"), output.find("</think>")) {
        (Some(s), Some(e)) if e > s => output[s + "<think>".len()..e].trim().to_string(),
        _ => output
            .split("<|audio_code_")
            .next()
            .unwrap_or("")
            .trim()
            .to_string(),
    };

    let mut meta = Metadata::new();
    let mut key: Option<String> = None;
    let mut lines: Vec<&str> = Vec::new();
    let save = |key: &mut Option<String>, lines: &mut Vec<&str>, meta: &mut Metadata| {
        if let Some(k) = key.take()
            && !lines.is_empty()
        {
            let value = lines.join("\n");
            let v = match k.as_str() {
                "bpm" | "duration" => match value.trim().parse::<i64>() {
                    Ok(i) => MetaValue::Int(i),
                    Err(_) => MetaValue::Str(value.trim().to_string()),
                },
                "caption" => MetaValue::Str(postprocess_caption(&value)),
                "genres" | "keyscale" | "language" | "timesignature" => {
                    MetaValue::Str(value.trim().to_string())
                }
                _ => {
                    lines.clear();
                    return;
                }
            };
            meta.insert(k, v);
        }
        lines.clear();
    };
    for line in reasoning.split('\n') {
        if line.trim().starts_with('<') {
            continue;
        }
        let starts_blank = line.chars().next().is_some_and(char::is_whitespace);
        if !line.is_empty() && !starts_blank && line.contains(':') {
            save(&mut key, &mut lines, &mut meta);
            let (k, v) = line.split_once(':').unwrap_or((line, ""));
            key = Some(k.trim().to_lowercase());
            if !v.trim().is_empty() {
                lines.push(v);
            }
        } else if (line.starts_with(' ') || line.starts_with('\t')) && key.is_some() {
            lines.push(line);
        }
    }
    save(&mut key, &mut lines, &mut meta);
    meta
}

/// The CoT block the codes phase is prompted with, as
/// `_format_metadata_as_cot` builds it: `<think>\n{yaml}\n</think>`.
pub fn format_plan(meta: &Metadata) -> String {
    let mut yaml = String::new();
    for key in COT_FIELDS {
        let Some(value) = meta.get(key) else {
            continue;
        };
        let value = match value {
            MetaValue::Str(s) if key == "timesignature" && s.ends_with("/4") => {
                MetaValue::Str(s.split('/').next().unwrap_or("").to_string())
            }
            v => v.clone(),
        };
        let value = match value {
            MetaValue::Str(s) if is_digits(&s) => match s.parse::<i64>() {
                Ok(i) => MetaValue::Int(i),
                Err(_) => MetaValue::Str(s),
            },
            v => v,
        };
        yaml.push_str(&yaml_entry(key, &value));
    }
    format!("<think>\n{}\n</think>", yaml.trim())
}

// ── A PyYAML block-mapping entry ─────────────────────────────────────

/// PyYAML's default line width and mapping indent.
const BEST_WIDTH: usize = 80;
const INDENT: usize = 2;

/// One `key: value\n` line (or lines) of `yaml.dump(..., allow_unicode=True)`.
fn yaml_entry(key: &str, value: &MetaValue) -> String {
    let mut w = Writer {
        out: format!("{key}:"),
        column: key.chars().count() + 1,
    };
    match value {
        MetaValue::Int(i) => {
            w.out.push(' ');
            w.out.push_str(&i.to_string());
        }
        MetaValue::Str(s) => match scalar_style(s) {
            Style::Plain => w.plain(s),
            Style::SingleQuoted => w.single_quoted(s),
            Style::DoubleQuoted => w.double_quoted(s),
        },
    }
    w.out.push('\n');
    w.out
}

#[derive(Debug, PartialEq)]
enum Style {
    Plain,
    SingleQuoted,
    DoubleQuoted,
}

fn is_ws(c: Option<char>) -> bool {
    matches!(
        c,
        None | Some('\0' | ' ' | '\t' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
    )
}

fn is_break(c: char) -> bool {
    matches!(c, '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `choose_scalar_style` for a block-mapping value: plain when PyYAML's
/// analysis allows a block plain scalar and the text would read back as a
/// string, else single-quoted when allowed, else double-quoted.
fn scalar_style(s: &str) -> Style {
    if s.is_empty() {
        // An empty value would read back as null.
        return Style::SingleQuoted;
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut block_indicators = s.starts_with("---") || s.starts_with("...");
    let mut line_breaks = false;
    let mut special = false;
    let (mut leading_space, mut leading_break, mut trailing_space, mut trailing_break) =
        (false, false, false, false);
    let (mut break_space, mut space_break) = (false, false);
    let (mut previous_space, mut previous_break) = (false, false);
    let mut preceded_by_ws = true;
    for i in 0..n {
        let ch = chars[i];
        let followed_by_ws = is_ws(chars.get(i + 1).copied());
        if i == 0 {
            if "#,[]{}&*!|>'\"%@`".contains(ch) {
                block_indicators = true;
            }
            if "?:".contains(ch) && followed_by_ws {
                block_indicators = true;
            }
            if ch == '-' && followed_by_ws {
                block_indicators = true;
            }
        } else {
            if ch == ':' && followed_by_ws {
                block_indicators = true;
            }
            if ch == '#' && preceded_by_ws {
                block_indicators = true;
            }
        }
        if is_break(ch) {
            line_breaks = true;
        }
        if !(ch == '\n' || (' '..='~').contains(&ch)) {
            let printable = ch == '\u{85}'
                || ('\u{A0}'..='\u{D7FF}').contains(&ch)
                || ('\u{E000}'..='\u{FFFD}').contains(&ch)
                || ('\u{10000}'..'\u{10FFFF}').contains(&ch);
            // allow_unicode=True: printable non-ASCII is fine.
            if !printable || ch == '\u{FEFF}' {
                special = true;
            }
        }
        if ch == ' ' {
            leading_space |= i == 0;
            trailing_space |= i == n - 1;
            break_space |= previous_break;
            previous_space = true;
            previous_break = false;
        } else if is_break(ch) {
            leading_break |= i == 0;
            trailing_break |= i == n - 1;
            space_break |= previous_space;
            previous_space = false;
            previous_break = true;
        } else {
            previous_space = false;
            previous_break = false;
        }
        preceded_by_ws = is_ws(Some(ch));
    }
    let mut allow_plain = true;
    let mut allow_single = true;
    if leading_space || leading_break || trailing_space || trailing_break {
        allow_plain = false;
    }
    if break_space {
        allow_plain = false;
        allow_single = false;
    }
    if space_break || special {
        allow_plain = false;
        allow_single = false;
    }
    if line_breaks || block_indicators {
        allow_plain = false;
    }
    if allow_plain && resolves_as_string(s) {
        Style::Plain
    } else if allow_single {
        Style::SingleQuoted
    } else {
        Style::DoubleQuoted
    }
}

/// Whether a plain scalar with this text would read back as a string under
/// PyYAML's YAML 1.1 implicit resolvers (bool, int, float, null, merge,
/// value, timestamp). Anything else needs quoting.
fn resolves_as_string(s: &str) -> bool {
    const BOOLS: [&str; 18] = [
        "yes", "Yes", "YES", "no", "No", "NO", "true", "True", "TRUE", "false", "False", "FALSE",
        "on", "On", "ON", "off", "Off", "OFF",
    ];
    const NULLS: [&str; 4] = ["~", "null", "Null", "NULL"];
    if BOOLS.contains(&s) || NULLS.contains(&s) || s == "<<" || s == "=" {
        return false;
    }
    !(is_yaml_int(s) || is_yaml_float(s) || is_yaml_timestamp(s))
}

fn strip_sign(s: &str) -> &str {
    s.strip_prefix(['-', '+']).unwrap_or(s)
}

/// PyYAML's int resolver: binary, octal, decimal, hex and sexagesimal,
/// with `_` separators.
fn is_yaml_int(s: &str) -> bool {
    let b = strip_sign(s);
    let all = |t: &str, ok: fn(char) -> bool| !t.is_empty() && t.chars().all(|c| ok(c) || c == '_');
    if let Some(t) = b.strip_prefix("0b") {
        return all(t, |c| c == '0' || c == '1');
    }
    if let Some(t) = b.strip_prefix("0x") {
        return all(t, |c| c.is_ascii_hexdigit());
    }
    if b == "0" {
        return true;
    }
    if let Some(t) = b.strip_prefix('0') {
        return all(t, |c| ('0'..='7').contains(&c));
    }
    if b.starts_with(|c: char| ('1'..='9').contains(&c)) {
        if all(b, |c| c.is_ascii_digit()) {
            return true;
        }
        // Sexagesimal: [1-9][0-9_]*(:[0-5]?[0-9])+
        let mut parts = b.split(':');
        let head = parts.next().unwrap_or("");
        let rest: Vec<&str> = parts.collect();
        return !rest.is_empty()
            && all(head, |c| c.is_ascii_digit())
            && rest.iter().all(|p| {
                let p: Vec<char> = p.chars().collect();
                match p.as_slice() {
                    [d] => d.is_ascii_digit(),
                    [a, d] => ('0'..='5').contains(a) && d.is_ascii_digit(),
                    _ => false,
                }
            });
    }
    false
}

/// PyYAML's float resolver, including `.inf`, `.nan` and sexagesimal.
fn is_yaml_float(s: &str) -> bool {
    let b = strip_sign(s);
    if matches!(b, ".inf" | ".Inf" | ".INF") {
        return true;
    }
    if matches!(s, ".nan" | ".NaN" | ".NAN") {
        return true;
    }
    // [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?  and  \.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
    let (mantissa, exp) = match b.find(['e', 'E']) {
        Some(i) => (&b[..i], Some(&b[i + 1..])),
        None => (b, None),
    };
    if let Some(e) = exp {
        let digits = e.strip_prefix(['-', '+']);
        if !digits.is_some_and(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())) {
            return false;
        }
    }
    let Some((int, frac)) = mantissa.split_once('.') else {
        // Sexagesimal float: [0-9][0-9_]*(:[0-5]?[0-9])+\.[0-9_]*
        return false;
    };
    let digits_ = |t: &str| t.chars().all(|c| c.is_ascii_digit() || c == '_');
    if int.is_empty() {
        frac.starts_with(|c: char| c.is_ascii_digit()) && digits_(frac)
    } else {
        int.starts_with(|c: char| c.is_ascii_digit()) && digits_(int) && digits_(frac)
    }
}

/// PyYAML's timestamp resolver, enough to catch dates and date-times.
fn is_yaml_timestamp(s: &str) -> bool {
    let c: Vec<char> = s.chars().collect();
    c.len() >= 8
        && c[..4].iter().all(char::is_ascii_digit)
        && c[4] == '-'
        && c[5].is_ascii_digit()
        && c.iter().skip(5).take_while(|x| x.is_ascii_digit()).count() <= 2
        && s[5..].contains('-')
}

/// The emitter state a scalar writer needs: the text so far and the
/// current column.
struct Writer {
    out: String,
    column: usize,
}

impl Writer {
    fn write(&mut self, s: &str) {
        self.column += s.chars().count();
        self.out.push_str(s);
    }

    /// `write_indent` mid-scalar: a line break and the mapping indent.
    fn indent(&mut self) {
        self.out.push('\n');
        self.out.push_str(&" ".repeat(INDENT));
        self.column = INDENT;
    }

    /// `write_plain` with `split=True`: a run of exactly one space past
    /// the best width becomes a line break.
    fn plain(&mut self, text: &str) {
        self.write(" ");
        let chars: Vec<char> = text.chars().collect();
        let (mut spaces, mut start, mut end) = (false, 0usize, 0usize);
        while end <= chars.len() {
            let ch = chars.get(end).copied();
            if spaces {
                if ch != Some(' ') {
                    if start + 1 == end && self.column > BEST_WIDTH {
                        self.indent();
                    } else {
                        self.write(&chars[start..end].iter().collect::<String>());
                    }
                    start = end;
                }
            } else if ch.is_none() || ch == Some(' ') {
                self.write(&chars[start..end].iter().collect::<String>());
                start = end;
            }
            if let Some(c) = ch {
                spaces = c == ' ';
            }
            end += 1;
        }
    }

    /// `write_single_quoted` with `split=True` (no line breaks reach it:
    /// captions are joined to one line first).
    fn single_quoted(&mut self, text: &str) {
        self.write(" '");
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let (mut spaces, mut start, mut end) = (false, 0usize, 0usize);
        while end <= n {
            let ch = chars.get(end).copied();
            if spaces {
                if ch != Some(' ') {
                    if start + 1 == end && self.column > BEST_WIDTH && start != 0 && end != n {
                        self.indent();
                    } else {
                        self.write(&chars[start..end].iter().collect::<String>());
                    }
                    start = end;
                }
            } else if (ch.is_none() || ch == Some(' ') || ch == Some('\'')) && start < end {
                self.write(&chars[start..end].iter().collect::<String>());
                start = end;
            }
            if ch == Some('\'') {
                self.write("''");
                start = end + 1;
            }
            if let Some(c) = ch {
                spaces = c == ' ';
            }
            end += 1;
        }
        self.write("'");
    }

    /// Double quotes for text with control characters. PyYAML folds these
    /// too; a plan never contains them, so this writes one line.
    fn double_quoted(&mut self, text: &str) {
        let mut s = String::from(" \"");
        for c in text.chars() {
            match c {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                '\n' => s.push_str("\\n"),
                '\t' => s.push_str("\\t"),
                c if (c as u32) < 0x20 => s.push_str(&format!("\\x{:02X}", c as u32)),
                c => s.push(c),
            }
        }
        s.push('"');
        self.write(&s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_planner() -> serde_json::Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/harness/testdata/acestep/vocal_en_30s_plan__cpu_f32.json");
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        meta["planner"].clone()
    }

    /// The codes-phase prompt upstream built from its own CoT output: the
    /// plan read back, then dumped as YAML, folded at column 80.
    #[test]
    fn the_plan_round_trips_to_upstream_s_codes_prompt() {
        let planner = fixture_planner();
        let cot_out = planner[0]["output_text"].as_str().unwrap();
        let codes_prompt = planner[1]["formatted_prompt"].as_str().unwrap();
        let plan = parse_plan(cot_out);
        assert_eq!(plan["bpm"], MetaValue::Int(130));
        assert_eq!(plan["duration"], MetaValue::Int(30));
        assert_eq!(plan["timesignature"], MetaValue::Str("4".into()));
        let MetaValue::Str(caption) = &plan["caption"] else {
            panic!("caption is text");
        };
        assert!(caption.starts_with("An energetic indie rock track"));
        assert!(!caption.contains('\n'), "the caption is unfolded");

        let cot = format_plan(&plan);
        let start = codes_prompt.find("<think>").unwrap();
        let want = &codes_prompt[start..codes_prompt.len() - "\n\n".len()];
        assert_eq!(cot, want);
    }

    #[test]
    fn values_that_would_read_back_as_something_else_are_quoted() {
        assert_eq!(
            yaml_entry("language", &MetaValue::Str("no".into())),
            "language: 'no'\n"
        );
        assert_eq!(
            yaml_entry("language", &MetaValue::Str("en".into())),
            "language: en\n"
        );
        assert_eq!(
            yaml_entry("keyscale", &MetaValue::Str("F# minor".into())),
            "keyscale: F# minor\n"
        );
        assert_eq!(
            yaml_entry("caption", &MetaValue::Str("Lo-fi: dusty drums".into())),
            "caption: 'Lo-fi: dusty drums'\n"
        );
        assert_eq!(
            yaml_entry("caption", &MetaValue::Str("'quoted' start".into())),
            "caption: '''quoted'' start'\n"
        );
        assert_eq!(
            yaml_entry("caption", &MetaValue::Str("1.5".into())),
            "caption: '1.5'\n"
        );
        assert_eq!(yaml_entry("bpm", &MetaValue::Int(92)), "bpm: 92\n");
    }

    /// Recorded from PyYAML 6 (`yaml.dump({"caption": v}, allow_unicode=True,
    /// sort_keys=True)`).
    #[test]
    fn scalars_match_pyyaml() {
        let folded = "A track: ".to_string() + &"word ".repeat(30) + "end";
        let cases: Vec<(String, &str)> = vec![
            ("no".into(), "caption: 'no'\n"),
            ("en".into(), "caption: en\n"),
            ("F# minor".into(), "caption: F# minor\n"),
            (
                "Lo-fi: dusty drums".into(),
                "caption: 'Lo-fi: dusty drums'\n",
            ),
            ("'quoted' start".into(), "caption: '''quoted'' start'\n"),
            ("1.5".into(), "caption: '1.5'\n"),
            (
                folded,
                "caption: 'A track: word word word word word word word word word word word word word\n  word word word word word word word word word word word word word word word word\n  word end'\n",
            ),
            ("yes".into(), "caption: 'yes'\n"),
            ("2024-01-02".into(), "caption: '2024-01-02'\n"),
            ("1:30".into(), "caption: '1:30'\n"),
            ("0x1F".into(), "caption: '0x1F'\n"),
            ("~".into(), "caption: '~'\n"),
            ("null".into(), "caption: 'null'\n"),
            ("- dash start".into(), "caption: '- dash start'\n"),
            ("tag #x".into(), "caption: 'tag #x'\n"),
            ("a#b".into(), "caption: a#b\n"),
            ("ends:".into(), "caption: 'ends:'\n"),
        ];
        for (v, want) in cases {
            assert_eq!(
                yaml_entry("caption", &MetaValue::Str(v.clone())),
                want,
                "{v:?}"
            );
        }
    }

    #[test]
    fn a_quoted_caption_folds_like_a_plain_one() {
        let caption = "A track: ".to_string() + &"word ".repeat(30) + "end";
        let entry = yaml_entry("caption", &MetaValue::Str(caption));
        let lines: Vec<&str> = entry.trim_end().split('\n').collect();
        assert!(lines[0].starts_with("caption: 'A track: word"));
        assert!(lines.len() > 1);
        assert!(lines[1..].iter().all(|l| l.starts_with("  ")));
        assert!(lines[0].chars().count() > BEST_WIDTH);
    }

    #[test]
    fn genres_and_unknown_fields_never_reach_the_prompt() {
        let plan = parse_plan("<think>\nbpm: 90\ngenres: jazz\nmood: calm\n</think>");
        assert!(plan.contains_key("genres"));
        assert!(!plan.contains_key("mood"));
        assert_eq!(format_plan(&plan), "<think>\nbpm: 90\n</think>");
    }
}
