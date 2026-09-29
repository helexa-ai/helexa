//! `/v1/systemone` wire types: typed decisions over a state (#333).
//!
//! The surface is TypeSafe Jev's decision protocol, as served by the Laya
//! reference server (`laya-serve`), so existing Jev and Laya clients work
//! unchanged:
//!
//! ```json
//! {"state": "I was charged twice", "questions": {
//!    "queue": {"type": "choice", "instructions": "Which team?",
//!              "criteria": {"billing": "billing and refunds", "tech": "login issues"}}}}
//! ```
//!
//! Two properties of the reference make a plain `serde_json::Value` the
//! wrong representation for the request:
//!
//! - **Order is meaning.** The order of a choice question's criteria is the
//!   order of its options in the model's input, and the order of the
//!   questions is the order of the answers. `serde_json::Value` (without the
//!   workspace-wide `preserve_order` feature) sorts object keys.
//! - **The state is re-serialized before it is tokenized**, with Python's
//!   `json.dumps(..., ensure_ascii=False)`. To feed the model the same bytes,
//!   the value must round-trip exactly as Python's `json.loads` read it:
//!   integers of any size, floats as the nearest double, `NaN`/`Infinity`
//!   (which Python accepts), and duplicate keys resolved last-value-wins at
//!   the first key's position.
//!
//! [`Json`] is that representation, with a parser matching `json.loads` and
//! the renderings the reference uses: [`Json::dumps`] (`json.dumps`),
//! [`Json::py_str`] (`str()`), [`Json::py_repr`] (`repr()`, for error
//! messages) and [`Json::json_key`] (a dict key as `json.dumps` writes it).

use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};
use std::fmt::Write as _;

/// Upper bounds the reference server enforces before tokenizing (`laya.serve`).
pub mod limits {
    /// Request body bytes (413).
    pub const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
    /// Questions per request (413).
    pub const MAX_QUESTIONS: usize = 64;
    /// Characters in the state (413). A non-string state is measured by its
    /// Python `repr`, as the reference does.
    pub const MAX_STATE_CHARS: usize = 50_000;
    /// Options per choice question (413).
    pub const MAX_CHOICE_OPTIONS: usize = 100;
    /// Levels per score question (413).
    pub const MAX_SCORE_LEVELS: usize = 32;
    /// Options across all questions (413).
    pub const MAX_TOTAL_OPTIONS: usize = 512;
    /// Default ceiling on a per-request `max_len` / `head_max_len` (422).
    pub const DEFAULT_MAX_TOKEN_BUDGET: u64 = 8192;
}

/// A JSON value as Python's `json.loads` produces it: object key order and
/// integer precision preserved.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// An integer literal, as canonical decimal digits (`-0` reads as `0`).
    Int(String),
    /// A literal with a fraction or exponent, or `NaN` / `Infinity`.
    Float(f64),
    String(String),
    Array(Vec<Json>),
    /// Keys in first-seen order; a repeated key keeps its first position and
    /// takes the last value, as a Python dict does.
    Object(Vec<(String, Json)>),
}

/// Why a body could not be read as JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonParseError {
    pub offset: usize,
    pub reason: &'static str,
}

impl std::fmt::Display for JsonParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.reason, self.offset)
    }
}

impl std::error::Error for JsonParseError {}

impl Json {
    /// Parse a request body the way `json.loads(bytes)` does for UTF-8 input.
    ///
    /// Accepts a leading UTF-8 BOM (Python decodes bytes as `utf-8-sig`),
    /// `NaN` / `Infinity` / `-Infinity`, and surrounding whitespace. Rejects
    /// lone UTF-16 surrogates in `\u` escapes: Python keeps them in a `str`,
    /// but they have no UTF-8 encoding and the tokenizer could not read them.
    pub fn parse(bytes: &[u8]) -> Result<Json, JsonParseError> {
        let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
        let text = std::str::from_utf8(bytes).map_err(|e| JsonParseError {
            offset: e.valid_up_to(),
            reason: "invalid UTF-8",
        })?;
        let mut p = Parser { s: text, i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != text.len() {
            return Err(p.err("trailing data"));
        }
        Ok(v)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// Look up an object member.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The Python type name, for error messages.
    pub fn py_type_name(&self) -> &'static str {
        match self {
            Json::Null => "NoneType",
            Json::Bool(_) => "bool",
            Json::Int(_) => "int",
            Json::Float(_) => "float",
            Json::String(_) => "str",
            Json::Array(_) => "list",
            Json::Object(_) => "dict",
        }
    }

    /// `json.dumps(self, ensure_ascii=False)`: separators `", "` and `": "`,
    /// non-ASCII kept as-is, floats as Python's `repr`.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.dumps_into(&mut out);
        out
    }

    fn dumps_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(true) => out.push_str("true"),
            Json::Bool(false) => out.push_str("false"),
            Json::Int(d) => out.push_str(d),
            Json::Float(x) => out.push_str(&json_float(*x)),
            Json::String(s) => dump_str(s, out),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.dumps_into(out);
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    dump_str(k, out);
                    out.push_str(": ");
                    v.dumps_into(out);
                }
                out.push('}');
            }
        }
    }

    /// Python's `str(value)`: a string is itself, everything else its `repr`.
    pub fn py_str(&self) -> String {
        match self {
            Json::String(s) => s.clone(),
            other => other.py_repr(),
        }
    }

    /// Python's `repr(value)` for the value `json.loads` would have built.
    pub fn py_repr(&self) -> String {
        let mut out = String::new();
        self.repr_into(&mut out);
        out
    }

    fn repr_into(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("None"),
            Json::Bool(true) => out.push_str("True"),
            Json::Bool(false) => out.push_str("False"),
            Json::Int(d) => out.push_str(d),
            Json::Float(x) => out.push_str(&py_float_repr(*x)),
            Json::String(s) => repr_str(s, out),
            Json::Array(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.repr_into(out);
                }
                out.push(']');
            }
            Json::Object(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    repr_str(k, out);
                    out.push_str(": ");
                    v.repr_into(out);
                }
                out.push('}');
            }
        }
    }

    /// The object key `json.dumps` writes for this value used as a dict key:
    /// strings as-is, `True` as `true`, `None` as `null`, numbers as text.
    /// `None` for values Python cannot use as a key (lists, dicts).
    pub fn json_key(&self) -> Option<String> {
        match self {
            Json::String(s) => Some(s.clone()),
            Json::Null => Some("null".into()),
            Json::Bool(b) => Some(if *b { "true" } else { "false" }.into()),
            Json::Int(d) => Some(d.clone()),
            Json::Float(x) => Some(json_float(*x)),
            Json::Array(_) | Json::Object(_) => None,
        }
    }

    /// The identity Python's dict uses for this value as a key, where the
    /// value is hashable. `1`, `1.0` and `True` are one key; `"1"` is another.
    /// A NaN is never equal to anything, including another NaN.
    pub fn hash_key(&self) -> Option<HashKey> {
        match self {
            Json::String(s) => Some(HashKey::Str(s.clone())),
            Json::Null => Some(HashKey::None),
            Json::Bool(b) => Some(HashKey::Num(if *b { "1" } else { "0" }.into())),
            Json::Int(d) => Some(HashKey::Num(d.clone())),
            Json::Float(x) => {
                if x.is_nan() {
                    Some(HashKey::Unique)
                } else if x.is_finite() && x.fract() == 0.0 {
                    Some(HashKey::Num(integral_float_digits(*x)))
                } else {
                    Some(HashKey::Float(x.to_bits()))
                }
            }
            Json::Array(_) | Json::Object(_) => None,
        }
    }
}

/// See [`Json::hash_key`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HashKey {
    None,
    Str(String),
    /// An integer-valued number, as canonical decimal digits.
    Num(String),
    /// A non-integral finite or infinite float, by bit pattern.
    Float(u64),
    /// NaN: equal to nothing.
    Unique,
}

impl Serialize for Json {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => s.serialize_unit(),
            Json::Bool(b) => s.serialize_bool(*b),
            Json::Int(d) => {
                if let Ok(v) = d.parse::<i64>() {
                    s.serialize_i64(v)
                } else if let Ok(v) = d.parse::<u64>() {
                    s.serialize_u64(v)
                } else {
                    // Beyond 64 bits there is no lossless serde number.
                    s.serialize_f64(d.parse::<f64>().unwrap_or(f64::NAN))
                }
            }
            Json::Float(x) => s.serialize_f64(*x),
            Json::String(v) => s.serialize_str(v),
            Json::Array(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Json::Object(m) => {
                let mut map = s.serialize_map(Some(m.len()))?;
                for (k, v) in m {
                    map.serialize_entry(k, v)?;
                }
                map.end()
            }
        }
    }
}

struct Parser<'a> {
    s: &'a str,
    i: usize,
}

/// Python's default recursion limit bounds nesting; stay well inside our stack.
const MAX_DEPTH: usize = 512;

impl Parser<'_> {
    fn err(&self, reason: &'static str) -> JsonParseError {
        JsonParseError {
            offset: self.i,
            reason,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.as_bytes().get(self.i).copied()
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonParseError> {
        if depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b'n') if self.eat("null") => Ok(Json::Null),
            Some(b't') if self.eat("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat("false") => Ok(Json::Bool(false)),
            Some(b'N') if self.eat("NaN") => Ok(Json::Float(f64::NAN)),
            Some(b'I') if self.eat("Infinity") => Ok(Json::Float(f64::INFINITY)),
            Some(b'-') if self.s[self.i..].starts_with("-Infinity") => {
                self.i += "-Infinity".len();
                Ok(Json::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.err("expecting value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonParseError> {
        self.i += 1;
        let mut members: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Object(members));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("expecting property name enclosed in double quotes"));
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err(self.err("expecting ':' delimiter"));
            }
            self.i += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            match members.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = v,
                None => members.push((key, v)),
            }
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Object(members));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonParseError> {
        self.i += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err(self.err("expecting ',' delimiter")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonParseError> {
        let h = self
            .s
            .get(self.i..self.i + 4)
            .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
            .ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        self.i += 4;
        Ok(u32::from_str_radix(h, 16).expect("four hex digits"))
    }

    fn string(&mut self) -> Result<String, JsonParseError> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            // Copy the run of ordinary characters in one go.
            while let Some(b) = self.peek() {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.i += 1;
            }
            out.push_str(&self.s[start..self.i]);
            match self.peek() {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if !self.eat("\\u") {
                                    return Err(self.err("lone surrogate in \\u escape"));
                                }
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("lone surrogate in \\u escape"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err(self.err("lone surrogate in \\u escape"));
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).expect("valid scalar value"));
                        }
                        _ => return Err(self.err("invalid \\escape")),
                    }
                }
                Some(_) => return Err(self.err("invalid control character in string")),
            }
        }
    }

    fn number(&mut self) -> Result<Json, JsonParseError> {
        let b = self.s.as_bytes();
        let start = self.i;
        if b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        match b.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                while matches!(b.get(self.i), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
            _ => return Err(self.err("expecting value")),
        }
        let int_end = self.i;
        let mut is_float = false;
        if b.get(self.i) == Some(&b'.') && matches!(b.get(self.i + 1), Some(b'0'..=b'9')) {
            is_float = true;
            self.i += 1;
            while matches!(b.get(self.i), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(b.get(self.i), Some(b'e' | b'E')) {
            let mut j = self.i + 1;
            if matches!(b.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if matches!(b.get(j), Some(b'0'..=b'9')) {
                is_float = true;
                self.i = j;
                while matches!(b.get(self.i), Some(b'0'..=b'9')) {
                    self.i += 1;
                }
            }
        }
        let text = &self.s[start..self.i];
        if is_float {
            // Rust's float parsing is correctly rounded, like Python's.
            let x: f64 = text.parse().map_err(|_| self.err("invalid number"))?;
            Ok(Json::Float(x))
        } else {
            let digits = &self.s[start..int_end];
            let canonical = if digits == "-0" { "0" } else { digits };
            Ok(Json::Int(canonical.to_string()))
        }
    }
}

fn dump_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(str)`: single quotes unless the text contains a single
/// quote and no double quote; non-printable characters escaped.
fn repr_str(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !py_printable(c) => {
                let cp = c as u32;
                let _ = if cp < 0x100 {
                    write!(out, "\\x{cp:02x}")
                } else if cp < 0x10000 {
                    write!(out, "\\u{cp:04x}")
                } else {
                    write!(out, "\\U{cp:08x}")
                };
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// An approximation of Python's `str.isprintable` for `repr`: control and
/// separator characters (other than the ASCII space) are escaped. Exact for
/// ASCII and Latin-1, which is what error messages quote in practice.
fn py_printable(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x20 || (0x7F..0xA1).contains(&cp) || cp == 0xAD {
        return false;
    }
    !matches!(
        cp,
        0x2028 | 0x2029 | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x2064 | 0xFEFF
    ) && !(0xD800..0xE000).contains(&cp)
        && !(0xE000..0xF900).contains(&cp)
        && !(c.is_whitespace() && c != ' ')
}

/// Decimal digits of an integral, finite float (`1e16` → `10000000000000000`).
fn integral_float_digits(x: f64) -> String {
    let s = format!("{x:.0}");
    if s == "-0" { "0".into() } else { s }
}

/// Python's `repr(float)`: the shortest round-tripping digits, positional
/// for exponents in `[-4, 16)` and scientific (`1e+16`, `1.5e-05`) outside.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    // `{:e}` is the shortest round-trip representation: "-1.2345e-7".
    let sci = format!("{x:e}");
    let (mant, exp) = sci
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let point = exp as usize + 1;
            if digits.len() <= point {
                out.push_str(&digits);
                out.push_str(&"0".repeat(point - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..point]);
                out.push('.');
                out.push_str(&digits[point..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        );
    }
    out
}

/// A float as `json.dumps` writes it: `repr`, except the non-finite values.
fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".into()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        py_float_repr(x)
    }
}

/// A rejected request: HTTP status plus the `detail` text the reference
/// server returns (FastAPI's `{"detail": ...}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionRequestError {
    pub status: u16,
    pub detail: String,
}

impl DecisionRequestError {
    pub fn new(status: u16, detail: impl Into<String>) -> Self {
        Self {
            status,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for DecisionRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.status, self.detail)
    }
}

impl std::error::Error for DecisionRequestError {}

/// A `/v1/systemone` request that has passed the reference server's
/// transport-level checks (body shape, size limits, budget parameters).
/// Per-question validation happens later, against the chosen checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneRequest {
    /// Never `Json::Null`: a missing or null state is a 400.
    pub state: Json,
    /// Question id → definition, in request order. The definitions are
    /// unvalidated.
    pub questions: Vec<(String, Json)>,
    /// The `model` field when it is a string. A Jev client sends its own
    /// service's name here; resolving it is the caller's job.
    pub model: Option<String>,
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

impl SystemOneRequest {
    /// Parse and check a body in the reference server's order: JSON (400),
    /// shape (400), size limits (400/413), then budget parameters (422).
    ///
    /// `max_token_budget` caps `max_len` / `head_max_len`
    /// ([`limits::DEFAULT_MAX_TOKEN_BUDGET`] unless configured). The body
    /// size limit is the transport's to enforce before this is called.
    pub fn parse(body: &[u8], max_token_budget: u64) -> Result<Self, DecisionRequestError> {
        let v = Json::parse(body)
            .map_err(|_| DecisionRequestError::new(400, "request body must be valid JSON"))?;
        let Json::Object(members) = v else {
            return Err(DecisionRequestError::new(
                400,
                "request body must be an object with a 'questions' field",
            ));
        };
        let field = |k: &str| members.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let Some(questions) = field("questions") else {
            return Err(DecisionRequestError::new(
                400,
                "request body must be an object with a 'questions' field",
            ));
        };
        let state = field("state").cloned().unwrap_or(Json::Null);
        check_request_limits(&state, questions)?;
        let Json::Object(questions) = questions.clone() else {
            unreachable!("check_request_limits rejects a non-object")
        };
        let model = match field("model") {
            Some(Json::String(s)) => Some(s.clone()),
            _ => None,
        };
        let max_len = budget_param(field("max_len"), "max_len", max_token_budget)?;
        let head_max_len = budget_param(field("head_max_len"), "head_max_len", max_token_budget)?;
        Ok(Self {
            state,
            questions,
            model,
            max_len,
            head_max_len,
        })
    }
}

fn check_request_limits(state: &Json, questions: &Json) -> Result<(), DecisionRequestError> {
    use limits::*;
    if *state == Json::Null {
        return Err(DecisionRequestError::new(400, "'state' is required"));
    }
    let Json::Object(qs) = questions else {
        return Err(DecisionRequestError::new(
            400,
            "'questions' must be an object",
        ));
    };
    if qs.len() > MAX_QUESTIONS {
        return Err(DecisionRequestError::new(
            413,
            format!("too many questions ({} > {MAX_QUESTIONS})", qs.len()),
        ));
    }
    let mut total = 0usize;
    for (qid, q) in qs {
        let qid_repr = Json::String(qid.clone()).py_repr();
        let crit = q.get("criteria");
        match (q.get("type").and_then(Json::as_str), crit) {
            (Some("choice"), Some(Json::Object(c))) => {
                total += c.len();
                if c.len() > MAX_CHOICE_OPTIONS {
                    return Err(DecisionRequestError::new(
                        413,
                        format!(
                            "too many choice options for {qid_repr} ({} > {MAX_CHOICE_OPTIONS})",
                            c.len()
                        ),
                    ));
                }
            }
            (Some("choice"), Some(Json::Array(c))) => {
                total += c.len();
                if c.len() > MAX_CHOICE_OPTIONS {
                    return Err(DecisionRequestError::new(
                        413,
                        format!(
                            "too many choice options for {qid_repr} ({} > {MAX_CHOICE_OPTIONS})",
                            c.len()
                        ),
                    ));
                }
            }
            (Some("score"), Some(Json::Array(c))) => {
                total += c.len();
                if c.len() > MAX_SCORE_LEVELS {
                    return Err(DecisionRequestError::new(
                        413,
                        format!(
                            "too many score levels for {qid_repr} ({} > {MAX_SCORE_LEVELS})",
                            c.len()
                        ),
                    ));
                }
            }
            _ => {}
        }
    }
    if total > MAX_TOTAL_OPTIONS {
        return Err(DecisionRequestError::new(
            413,
            format!("too many answer options across questions ({total} > {MAX_TOTAL_OPTIONS})"),
        ));
    }
    // The reference measures a string in code points and anything else by
    // the length of its Python `repr`.
    let state_len = match state {
        Json::String(s) => s.chars().count(),
        other => other.py_repr().chars().count(),
    };
    if state_len > MAX_STATE_CHARS {
        return Err(DecisionRequestError::new(
            413,
            format!("state too large ({state_len} > {MAX_STATE_CHARS} chars)"),
        ));
    }
    Ok(())
}

fn budget_param(
    v: Option<&Json>,
    key: &str,
    cap: u64,
) -> Result<Option<usize>, DecisionRequestError> {
    match v {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Int(d)) => {
            let negative = d.starts_with('-');
            if negative || d == "0" {
                return Err(DecisionRequestError::new(
                    422,
                    format!("{key} must be a positive integer"),
                ));
            }
            match d.parse::<u64>() {
                Ok(n) if n <= cap => Ok(Some(n as usize)),
                _ => Err(DecisionRequestError::new(
                    422,
                    format!("{key} exceeds server limit ({d} > {cap})"),
                )),
            }
        }
        Some(_) => Err(DecisionRequestError::new(
            422,
            format!("{key} must be an integer"),
        )),
    }
}

/// An ordered JSON object: serialized as a map in insertion order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OrderedMap<V>(pub Vec<(String, V)>);

impl<V: Serialize> Serialize for OrderedMap<V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

/// `answers[qid].action` — the act head's probability of acting.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionAction {
    pub act_probability: f64,
}

/// One answer. Every probability is rounded to 4 decimal places, as the
/// reference rounds them.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DecisionAnswer {
    Choice {
        /// The winning label, as the caller wrote it (a list-form label may
        /// be a number or bool).
        choice: Json,
        /// Keyed by each label as `json.dumps` writes a dict key.
        probabilities: OrderedMap<f64>,
        confidence: f64,
        answer_confidence: f64,
        action: DecisionAction,
    },
    Score {
        /// Expected level index; may fall between levels.
        score: f64,
        /// Level index (`"0"`..) → the caller's level description.
        legend: OrderedMap<Json>,
        probabilities: OrderedMap<f64>,
        confidence: f64,
        answer_confidence: f64,
        action: DecisionAction,
    },
    Noul {
        /// P(true).
        noul: f64,
        confidence: f64,
        answer_confidence: f64,
        action: DecisionAction,
    },
}

/// `usage.options[qid]`: a question whose options lost distinguishable
/// token spans to the head budget. Present only when that happened.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CollapsedOptions {
    pub total: usize,
    pub distinct: usize,
    pub tokens_per_option: Option<usize>,
}

/// Jev usage. `input_tokens` is every token the forward pass encoded — the
/// state is encoded once per question — and `output_tokens` is always 0.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OrderedMap<CollapsedOptions>>,
}

/// The `/v1/systemone` response body.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SystemOneResponse {
    /// The decision head's constant name (`laya-rl-agent`); the checkpoint
    /// that answered is in `routing`.
    pub model: String,
    pub answers: OrderedMap<DecisionAnswer>,
    pub usage: DecisionUsage,
    /// Which checkpoint answered and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing: Option<DecisionRouting>,
}

/// The `routing` block of a decision response: which checkpoint of the
/// served family answered, and why. Field order follows the reference.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionRouting {
    /// The checkpoint's routing name (`english`, `multilingual`,
    /// `typed-decisions`).
    pub model: String,
    /// `<repo>` for the root checkpoint, `<repo>/<subfolder>` otherwise.
    pub repo: String,
    pub reason: String,
    /// The language/script analysis the choice was made on; `None` when
    /// the caller pinned the checkpoint.
    pub detection: Option<Json>,
    /// The typed-decisions workflow the question ids match, when one does.
    pub workflow: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j(s: &str) -> Json {
        Json::parse(s.as_bytes()).unwrap()
    }

    #[test]
    fn dumps_matches_python_json_dumps() {
        // Expected strings are `json.dumps(json.loads(src), ensure_ascii=False)`.
        let cases = [
            (
                r#"{"b": 1, "a": [1.0, 2.5, 1e16, 1.5e-5, -0.0, true, null]}"#,
                r#"{"b": 1, "a": [1.0, 2.5, 1e+16, 1.5e-05, -0.0, true, null]}"#,
            ),
            (
                r#"{"from": "Zoë", "t": "a\"b\\c\nd\u0001 请"}"#,
                r#"{"from": "Zoë", "t": "a\"b\\c\nd\u0001 请"}"#,
            ),
            (
                r#"[NaN, Infinity, -Infinity, 123456789012345678901234567890, -0]"#,
                r#"[NaN, Infinity, -Infinity, 123456789012345678901234567890, 0]"#,
            ),
            (r#"{"k": 1, "k": 2, "z": 3}"#, r#"{"k": 2, "z": 3}"#),
            (
                r#"[0.1, 100.0, 1e22, 0.0001, 0.00001, 12345678901234567.0]"#,
                r#"[0.1, 100.0, 1e+22, 0.0001, 1e-05, 1.2345678901234568e+16]"#,
            ),
            ("\u{feff}[\"\\ud83d\\ude00\"]", "[\"😀\"]"),
        ];
        for (src, want) in cases {
            assert_eq!(j(src).dumps(), want, "{src}");
        }
    }

    #[test]
    fn repr_and_str_match_python() {
        assert_eq!(j(r#""it's""#).py_repr(), r#""it's""#);
        assert_eq!(j(r#""a\nb""#).py_repr(), r#"'a\nb'"#);
        assert_eq!(
            j(r#"{"a": [1, null, true, 2.0]}"#).py_repr(),
            "{'a': [1, None, True, 2.0]}"
        );
        assert_eq!(j("true").py_str(), "True");
        assert_eq!(j("1e16").py_str(), "1e+16");
        assert_eq!(j(r#""x""#).py_str(), "x");
    }

    #[test]
    fn hash_keys_follow_python_equality() {
        assert_eq!(j("1").hash_key(), j("1.0").hash_key());
        assert_eq!(j("1").hash_key(), j("true").hash_key());
        assert_ne!(j("1").hash_key(), j(r#""1""#).hash_key());
        assert_ne!(j("1.5").hash_key(), j("1").hash_key());
        assert_eq!(j("NaN").hash_key(), Some(HashKey::Unique));
        assert_eq!(j("[1]").hash_key(), None);
    }

    #[test]
    fn parse_rejects_what_python_rejects() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "\"\u{1}\"",
            "01",
            "1.",
            "tru",
            "[1] x",
        ] {
            assert!(Json::parse(bad.as_bytes()).is_err(), "{bad:?}");
        }
        assert!(Json::parse(br#""\ud800""#).is_err(), "lone surrogate");
    }

    fn req(body: &str) -> Result<SystemOneRequest, DecisionRequestError> {
        SystemOneRequest::parse(body.as_bytes(), limits::DEFAULT_MAX_TOKEN_BUDGET)
    }

    #[test]
    fn request_checks_follow_the_reference_server() {
        assert_eq!(req("{").unwrap_err().status, 400);
        assert_eq!(req("[]").unwrap_err().status, 400);
        assert_eq!(
            req(r#"{"questions": {}}"#).unwrap_err().detail,
            "'state' is required"
        );
        assert_eq!(
            req(r#"{"state": "x", "questions": []}"#)
                .unwrap_err()
                .detail,
            "'questions' must be an object"
        );
        let many: Vec<String> = (0..65).map(|i| format!("\"q{i}\": {{}}")).collect();
        let e = req(&format!(
            r#"{{"state": "x", "questions": {{{}}}}}"#,
            many.join(",")
        ))
        .unwrap_err();
        assert_eq!(
            (e.status, e.detail.as_str()),
            (413, "too many questions (65 > 64)")
        );
        let levels: Vec<String> = (0..33).map(|i| format!("\"l{i}\"")).collect();
        let e = req(&format!(
            r#"{{"state": "x", "questions": {{"s": {{"type": "score", "criteria": [{}]}}}}}}"#,
            levels.join(",")
        ))
        .unwrap_err();
        assert_eq!(e.detail, "too many score levels for 's' (33 > 32)");
        let big = "x".repeat(50_001);
        let e = req(&format!(r#"{{"state": "{big}", "questions": {{}}}}"#)).unwrap_err();
        assert_eq!(e.detail, "state too large (50001 > 50000 chars)");
        for (v, detail) in [
            ("5.0", "max_len must be an integer"),
            ("true", "max_len must be an integer"),
            ("0", "max_len must be a positive integer"),
            ("-3", "max_len must be a positive integer"),
            ("9000", "max_len exceeds server limit (9000 > 8192)"),
        ] {
            let e = req(&format!(
                r#"{{"state": "x", "questions": {{}}, "max_len": {v}}}"#
            ))
            .unwrap_err();
            assert_eq!((e.status, e.detail.as_str()), (422, detail), "{v}");
        }
        let ok = req(r#"{"state": {"b": 1, "a": 2}, "questions": {"z": {}, "a": {}}, "model": "jev-1", "head_max_len": 64}"#).unwrap();
        assert_eq!(
            ok.questions
                .iter()
                .map(|(k, _)| k.as_str())
                .collect::<Vec<_>>(),
            ["z", "a"]
        );
        assert_eq!(ok.state.dumps(), r#"{"b": 1, "a": 2}"#);
        assert_eq!(
            (ok.model.as_deref(), ok.head_max_len),
            (Some("jev-1"), Some(64))
        );
    }

    #[test]
    fn response_serializes_in_order() {
        let r = SystemOneResponse {
            model: "laya-rl-agent".into(),
            answers: OrderedMap(vec![(
                "q".into(),
                DecisionAnswer::Noul {
                    noul: 0.25,
                    confidence: 0.75,
                    answer_confidence: 0.75,
                    action: DecisionAction {
                        act_probability: 1.0,
                    },
                },
            )]),
            usage: DecisionUsage {
                input_tokens: 3,
                output_tokens: 0,
                options: None,
            },
            routing: None,
        };
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"model":"laya-rl-agent","answers":{"q":{"type":"noul","noul":0.25,"confidence":0.75,"answer_confidence":0.75,"action":{"act_probability":1.0}}},"usage":{"input_tokens":3,"output_tokens":0}}"#
        );
    }
}
