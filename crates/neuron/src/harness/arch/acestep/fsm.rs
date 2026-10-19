//! Constrained decoding of the planner's chain of thought (#384).
//!
//! A port of upstream's `MetadataConstrainedLogitsProcessor` as the CoT
//! phase uses it: genres skipped, caption and language generated, no
//! user-provided fields, and generation stopped where `</think>` would
//! begin (`stop_at_reasoning`). It walks
//!
//! ```text
//! <think> \n bpm: <30..300> \n caption: <free text> \n duration: <10..600> \n
//! keyscale: <note[acc] major|minor> \n language: <code> \n timesignature: <2|3|4|6> \n
//! ```
//!
//! and ends in EOS. Each step it masks the logits to what the current state
//! allows; after sampling it advances on the token chosen. Field values are
//! matched as token-id sequences against prefix trees built from how the
//! tokenizer actually splits `"bpm: 120"` and the like, as upstream does.
//!
//! Reference: `acestep/constrained_logits_processor.py` at ACE-Step 1.5
//! `ca1e85fe` (`_process_single_sequence`, `update_state`, and the prefix
//! tree builders).

use std::collections::{BTreeSet, HashMap};

use candle_core::Result;

/// Upstream's `BPM_MIN..=BPM_MAX`.
const BPM: (u32, u32) = (30, 300);
/// Upstream's `DURATION_MIN..=DURATION_MAX`.
const DURATION: (u32, u32) = (10, 600);
const TIME_SIGNATURES: [&str; 4] = ["2", "3", "4", "6"];
const KEYSCALE_NOTES: [&str; 7] = ["A", "B", "C", "D", "E", "F", "G"];
const KEYSCALE_ACCIDENTALS: [&str; 5] = ["", "#", "b", "♯", "♭"];
const KEYSCALE_MODES: [&str; 2] = ["major", "minor"];
/// Upstream's `VALID_LANGUAGES`, `unknown` included.
pub const LANGUAGES: [&str; 51] = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi",
    "hr", "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl",
    "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi",
    "yue", "zh", "unknown",
];
/// Caption tokens before the caption is forced to end.
const MAX_CAPTION_TOKENS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    ThinkTag,
    NewlineAfterThink,
    BpmName,
    BpmValue,
    CaptionName,
    CaptionValue,
    DurationName,
    DurationValue,
    KeyscaleName,
    KeyscaleValue,
    LanguageName,
    LanguageValue,
    TimesigName,
    TimesigValue,
    ThinkEndTag,
    Completed,
}

impl State {
    fn fixed(self) -> Option<&'static str> {
        Some(match self {
            Self::ThinkTag => "<think>",
            Self::NewlineAfterThink => "\n",
            Self::BpmName => "bpm:",
            Self::CaptionName => "caption:",
            Self::DurationName => "duration:",
            Self::KeyscaleName => "keyscale:",
            Self::LanguageName => "language:",
            Self::TimesigName => "timesignature:",
            Self::ThinkEndTag => "</think>",
            _ => return None,
        })
    }

    /// Field order with genres skipped.
    fn next(self) -> Self {
        match self {
            Self::ThinkTag => Self::NewlineAfterThink,
            Self::NewlineAfterThink => Self::BpmName,
            Self::BpmName => Self::BpmValue,
            Self::BpmValue => Self::CaptionName,
            Self::CaptionName => Self::CaptionValue,
            Self::CaptionValue => Self::DurationName,
            Self::DurationName => Self::DurationValue,
            Self::DurationValue => Self::KeyscaleName,
            Self::KeyscaleName => Self::KeyscaleValue,
            Self::KeyscaleValue => Self::LanguageName,
            Self::LanguageName => Self::LanguageValue,
            Self::LanguageValue => Self::TimesigName,
            Self::TimesigName => Self::TimesigValue,
            Self::TimesigValue => Self::ThinkEndTag,
            Self::ThinkEndTag | Self::Completed => Self::Completed,
        }
    }
}

/// Token-id prefix → the tokens that may follow it.
type PrefixTree = HashMap<Vec<u32>, BTreeSet<u32>>;

/// The tokenizer, as the state machine needs it.
pub trait Vocab {
    /// Encode without special tokens added (special tokens in the text
    /// still encode to their ids).
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    /// One token's text, special tokens kept.
    fn decode(&self, token: u32) -> Result<String>;
}

/// Build a prefix tree for `values` as they tokenize after `"{field}: "`,
/// keyed from where `"{field}:"` ends. A value whose tokens do not start
/// with the field's own tokens is skipped, as upstream does.
fn prefix_tree(
    vocab: &dyn Vocab,
    field: &str,
    values: &[String],
    newline: u32,
    first_must_be_note: bool,
) -> Result<PrefixTree> {
    let context = vocab.encode(&format!("{field}:"))?;
    let mut tree = PrefixTree::new();
    for value in values {
        let full = vocab.encode(&format!("{field}: {value}"))?;
        if full.len() < context.len() || full[..context.len()] != context[..] {
            continue;
        }
        let ids = &full[context.len()..];
        if ids.is_empty() {
            continue;
        }
        if first_must_be_note {
            let first = vocab.decode(ids[0])?;
            let c = first
                .trim_start()
                .chars()
                .next()
                .map(|c| c.to_ascii_uppercase());
            if !c.is_some_and(|c| "ABCDEFG".contains(c)) {
                continue;
            }
        }
        for i in 0..=ids.len() {
            let entry = tree.entry(ids[..i].to_vec()).or_default();
            entry.insert(if i < ids.len() { ids[i] } else { newline });
        }
    }
    Ok(tree)
}

/// Keep only `allowed`; with nothing allowed, every logit is `-inf`.
fn whitelist(logits: &mut [f32], allowed: &[u32]) {
    let keep: Vec<(usize, f32)> = allowed
        .iter()
        .filter_map(|&t| logits.get(t as usize).map(|&v| (t as usize, v)))
        .collect();
    logits.fill(f32::NEG_INFINITY);
    for (i, v) in keep {
        logits[i] = v;
    }
}

/// `torch.argmax`: the first maximum.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The CoT phase's constraint, one per generation.
pub struct MetaFsm {
    state: State,
    position: usize,
    acc_ids: Vec<u32>,
    acc_value: String,
    caption_after_newline: bool,
    caption_tokens: usize,
    caption_ending: bool,
    pending_field: String,

    newline: u32,
    eos: u32,
    backtick: Option<u32>,
    audio_codes: (u32, u32),
    bpm: PrefixTree,
    duration: PrefixTree,
    timesig: PrefixTree,
    keyscale: PrefixTree,
    language: PrefixTree,
}

impl MetaFsm {
    /// `eos` is `<|im_end|>`; `audio_codes` the inclusive id range of the
    /// `<|audio_code_N|>` tokens, which a caption may never contain.
    pub fn new(vocab: &dyn Vocab, eos: u32, audio_codes: (u32, u32)) -> Result<Self> {
        let last = |s: &str| -> Result<Option<u32>> { Ok(vocab.encode(s)?.last().copied()) };
        let newline =
            last("\n")?.ok_or_else(|| candle_core::Error::Msg("newline has no token".into()))?;
        let range = |(lo, hi): (u32, u32)| (lo..=hi).map(|v| v.to_string()).collect::<Vec<_>>();
        let mut keyscales = Vec::new();
        for note in KEYSCALE_NOTES {
            for acc in KEYSCALE_ACCIDENTALS {
                for mode in KEYSCALE_MODES {
                    keyscales.push(format!("{note}{acc} {mode}"));
                }
            }
        }
        let strings = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        Ok(Self {
            state: State::ThinkTag,
            position: 0,
            acc_ids: Vec::new(),
            acc_value: String::new(),
            caption_after_newline: false,
            caption_tokens: 0,
            caption_ending: false,
            pending_field: String::new(),
            newline,
            eos,
            backtick: last("`")?,
            audio_codes,
            bpm: prefix_tree(vocab, "bpm", &range(BPM), newline, false)?,
            duration: prefix_tree(vocab, "duration", &range(DURATION), newline, false)?,
            timesig: prefix_tree(
                vocab,
                "timesignature",
                &strings(&TIME_SIGNATURES),
                newline,
                false,
            )?,
            keyscale: prefix_tree(vocab, "keyscale", &keyscales, newline, true)?,
            language: prefix_tree(vocab, "language", &strings(&LANGUAGES), newline, false)?,
        })
    }

    pub fn is_completed(&self) -> bool {
        self.state == State::Completed
    }

    /// The longest prefix of what remains of a fixed string that is one
    /// token; failing that, the first token of the remainder.
    fn fixed_token(&self, vocab: &dyn Vocab, fixed: &str) -> Result<Option<u32>> {
        let rest: Vec<char> = fixed.chars().skip(self.position).collect();
        if rest.is_empty() {
            return Ok(None);
        }
        for end in (1..=rest.len()).rev() {
            let ids = vocab.encode(&rest[..end].iter().collect::<String>())?;
            if ids.len() == 1 {
                return Ok(Some(ids[0]));
            }
        }
        Ok(vocab
            .encode(&rest.iter().collect::<String>())?
            .first()
            .copied())
    }

    fn transition(&mut self) {
        self.state = self.state.next();
        self.position = 0;
        self.acc_value.clear();
        self.acc_ids.clear();
        self.caption_after_newline = false;
        self.caption_tokens = 0;
        self.caption_ending = false;
        self.pending_field.clear();
    }

    fn tree(&self) -> &PrefixTree {
        match self.state {
            State::BpmValue => &self.bpm,
            State::DurationValue => &self.duration,
            State::TimesigValue => &self.timesig,
            State::KeyscaleValue => &self.keyscale,
            _ => &self.language,
        }
    }

    fn allows_newline(&self) -> bool {
        self.tree()
            .get(&self.acc_ids)
            .is_some_and(|s| s.contains(&self.newline))
    }

    fn continuations(&self) -> Vec<u32> {
        self.tree()
            .get(&self.acc_ids)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Mask `logits` to what the current state allows.
    pub fn constrain(&mut self, vocab: &dyn Vocab, logits: &mut [f32]) -> Result<()> {
        if self.state == State::Completed {
            return Ok(());
        }
        if let Some(fixed) = self.state.fixed() {
            if self.state == State::ThinkEndTag {
                // stop_at_reasoning: end where `</think>` would begin.
                whitelist(logits, &[self.eos]);
                return Ok(());
            }
            match self.fixed_token(vocab, fixed)? {
                Some(t) => whitelist(logits, &[t]),
                None => {
                    self.transition();
                    if self.state.fixed().is_none() {
                        logits.fill(0.0);
                        return self.constrain(vocab, logits);
                    }
                }
            }
            return Ok(());
        }
        match self.state {
            State::BpmValue | State::DurationValue => {
                let mut allowed = self.continuations();
                if self.allows_newline() {
                    allowed.push(self.newline);
                }
                whitelist(logits, &allowed);
            }
            State::CaptionValue => {
                if self.caption_after_newline {
                    self.caption_after_newline = false;
                    let top = vocab.decode(argmax(logits))?;
                    if top.chars().next().is_some_and(|c| c != ' ' && c != '\t') {
                        // The model is starting the next field's name: let
                        // it, and pick the field up from what it writes.
                        self.caption_ending = true;
                        self.pending_field.clear();
                        return Ok(());
                    }
                }
                if self.caption_ending {
                    return Ok(());
                }
                if let Some(b) = self.backtick
                    && let Some(v) = logits.get_mut(b as usize)
                {
                    *v = f32::NEG_INFINITY;
                }
                let (lo, hi) = self.audio_codes;
                for v in logits.iter_mut().take(hi as usize + 1).skip(lo as usize) {
                    *v = f32::NEG_INFINITY;
                }
                if self.caption_tokens >= MAX_CAPTION_TOKENS {
                    whitelist(logits, &[self.newline]);
                }
            }
            State::KeyscaleValue | State::TimesigValue => {
                if self.allows_newline() {
                    whitelist(logits, &[self.newline]);
                } else {
                    let allowed = self.continuations();
                    if allowed.is_empty() && self.state == State::KeyscaleValue {
                        whitelist(logits, &[self.newline]);
                    } else {
                        whitelist(logits, &allowed);
                    }
                }
            }
            State::LanguageValue => {
                if self.acc_ids.is_empty() {
                    // The single best first token among every language's.
                    let candidates = self.continuations();
                    match candidates.iter().copied().reduce(|best, t| {
                        let v = |t: u32| logits.get(t as usize).copied().unwrap_or(f32::MIN);
                        if v(t) > v(best) { t } else { best }
                    }) {
                        Some(t) => whitelist(logits, &[t]),
                        None => whitelist(logits, &[self.newline]),
                    }
                } else if self.allows_newline() {
                    whitelist(logits, &[self.newline]);
                } else {
                    let allowed = self.continuations();
                    if allowed.is_empty() {
                        whitelist(logits, &[self.newline]);
                    } else {
                        whitelist(logits, &allowed);
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Advance on the token that was chosen.
    pub fn update(&mut self, vocab: &dyn Vocab, token: u32) -> Result<()> {
        if self.state == State::Completed {
            return Ok(());
        }
        let text = vocab.decode(token)?;
        if let Some(fixed) = self.state.fixed() {
            self.position += text.chars().count();
            if self.state == State::ThinkEndTag && token == self.eos {
                self.state = State::Completed;
            } else if self.position >= fixed.chars().count() {
                self.transition();
            }
            return Ok(());
        }
        match self.state {
            State::BpmValue
            | State::DurationValue
            | State::TimesigValue
            | State::KeyscaleValue
            | State::LanguageValue => {
                if token == self.newline {
                    self.transition();
                } else {
                    self.acc_ids.push(token);
                    self.acc_value.push_str(&text);
                }
            }
            State::CaptionValue => {
                self.caption_tokens += 1;
                self.acc_value.push_str(&text);
                self.caption_after_newline = text.contains('\n');
                if self.caption_ending {
                    self.pending_field.push_str(&text);
                    if text.contains(':') {
                        let name = self
                            .pending_field
                            .trim()
                            .trim_end_matches(':')
                            .trim()
                            .to_lowercase();
                        let target = match name.as_str() {
                            "duration" => Some(State::DurationValue),
                            "keyscale" => Some(State::KeyscaleValue),
                            "language" => Some(State::LanguageValue),
                            "timesignature" => Some(State::TimesigValue),
                            _ => None,
                        };
                        match target {
                            Some(s) => {
                                self.transition();
                                self.state = s;
                            }
                            None => self.transition(),
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary that splits text into characters, except that a space
    /// joins the character after it (as `" G"` is one token in the real
    /// one), with `<think>`, `</think>` and `<|im_end|>` as specials.
    /// Enough to drive every state; the real tokenizer is exercised by the
    /// planner parity test.
    struct Chars;
    const SPECIALS: [&str; 3] = ["<think>", "</think>", "<|im_end|>"];
    const BASE: u32 = 1000;
    const SPACED: u32 = 3000;

    impl Vocab for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            if let Some(i) = SPECIALS.iter().position(|s| *s == text) {
                return Ok(vec![BASE + i as u32]);
            }
            let chars: Vec<char> = text.chars().collect();
            let mut ids = Vec::new();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == ' '
                    && i + 1 < chars.len()
                    && (chars[i + 1] as u32) < 128
                    && chars[i + 1] != ' '
                {
                    ids.push(SPACED + chars[i + 1] as u32);
                    i += 2;
                } else {
                    ids.push(chars[i] as u32);
                    i += 1;
                }
            }
            Ok(ids)
        }
        fn decode(&self, token: u32) -> Result<String> {
            Ok(match token {
                t if (BASE..BASE + 3).contains(&t) => SPECIALS[(t - BASE) as usize].to_string(),
                t if (SPACED..SPACED + 128).contains(&t) => {
                    format!(" {}", char::from_u32(t - SPACED).unwrap_or('?'))
                }
                t => char::from_u32(t).unwrap_or('?').to_string(),
            })
        }
    }

    /// Drive the machine with a model that always prefers the next
    /// character of `wanted`; return what it generates.
    fn run(wanted: &str) -> String {
        let v = Chars;
        let eos = BASE + 2;
        let mut fsm = MetaFsm::new(&v, eos, (2000, 2010)).unwrap();
        let want: Vec<u32> = {
            let mut ids = vec![BASE];
            ids.extend(v.encode(wanted).unwrap());
            ids
        };
        let mut out = String::new();
        for step in 0..400 {
            let mut logits = vec![0f32; 4000];
            if let Some(&t) = want.get(step) {
                logits[t as usize] = 10.0;
            }
            fsm.constrain(&v, &mut logits).unwrap();
            let t = argmax(&logits);
            fsm.update(&v, t).unwrap();
            if t == eos {
                out.push_str("<EOS>");
                break;
            }
            out.push_str(&v.decode(t).unwrap());
        }
        out
    }

    #[test]
    fn a_well_formed_plan_passes_through_and_stops_before_the_close_tag() {
        let plan = "\nbpm: 120\ncaption: A calm piano piece\n  with soft strings\nduration: 30\nkeyscale: C major\nlanguage: en\ntimesignature: 4\n";
        assert_eq!(run(plan), format!("<think>{plan}<EOS>"));
    }

    #[test]
    fn out_of_range_and_off_vocabulary_values_are_steered() {
        // bpm 999 is out of range; keyscale H is not a note; language xx
        // is not a code. The model's preference is followed only where the
        // grammar allows it.
        let plan = "\nbpm: 999\ncaption: x\nduration: 30\nkeyscale: H major\nlanguage: xx\ntimesignature: 5\n";
        let got = run(plan);
        assert!(got.starts_with("<think>\nbpm: 99"), "{got}");
        let bpm: u32 = got["<think>\nbpm: ".len()..]
            .split('\n')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!((BPM.0..=BPM.1).contains(&bpm), "{got}");
        assert!(got.contains("\nkeyscale: A"), "{got}");
        let lang = got
            .split("language: ")
            .nth(1)
            .unwrap()
            .split('\n')
            .next()
            .unwrap();
        assert!(LANGUAGES.contains(&lang), "{got}");
        let ts = got
            .split("timesignature: ")
            .nth(1)
            .unwrap()
            .split('\n')
            .next()
            .unwrap();
        assert!(TIME_SIGNATURES.contains(&ts), "{got}");
        assert!(got.ends_with("<EOS>"));
    }
}
