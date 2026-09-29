//! Sequence assembly: one token sequence per question, under the model's
//! token budgets.
//!
//! ```text
//! [CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]
//! ```
//!
//! The budget rules, in the order they apply:
//!
//! 1. Each option is `[MASK]` plus at most 48 tokens of its text.
//! 2. `head_max_len` bounds the instruction text and all options together.
//!    If fewer than 16 tokens would remain for the instructions, every
//!    option is cut to the same `max(4, (head_max_len - 16) / options)`
//!    tokens (marker included).
//! 3. The instructions get what is left of `head_max_len`, but never fewer
//!    than 8 tokens.
//! 4. The state fills the rest of `max_len`. A conversation (a JSON list)
//!    keeps its newest turns, so it is cut from the left; anything else is
//!    cut from the right.
//! 5. The whole sequence is cut to `max_len`, dropping any marker that no
//!    longer fits — a question whose markers do not all fit is refused.

use super::question::{QType, Question, questions_from_request};
use super::tokenizer::TextEncoder;
use super::{DecisionConfig, DecisionRequestError, unprocessable};
use cortex_core::decisions::Json;

/// Option text tokens kept before the head budget is considered.
pub const MAX_OPTION_TOKENS: usize = 48;
/// Head tokens the instructions are guaranteed before options are shrunk.
const MIN_INSTRUCTION_ROOM: usize = 16;
/// Per-option floor (marker included) when options are shrunk.
const MIN_TOKENS_PER_OPTION: usize = 4;
/// Instruction tokens kept however many the options take.
const MIN_INSTRUCTION_TOKENS: usize = 8;

/// Per-request overrides of the checkpoint's token budgets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budget {
    pub max_len: Option<usize>,
    pub head_max_len: Option<usize>,
}

/// One model input row: one question over the state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub input_ids: Vec<u32>,
    /// Position of each option's `[MASK]` marker, in option order.
    pub markers: Vec<usize>,
    pub qtype: QType,
}

/// What the head budget did to a question's options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionStats {
    pub options: usize,
    /// Options whose token span is still unique after the cut. Fewer than
    /// `options` means the model can no longer tell some of them apart.
    pub options_distinct: usize,
    /// The per-option cap applied, when the options had to be shrunk.
    pub tokens_per_option: Option<usize>,
}

/// A request turned into model input: one [`Row`] per question, in request
/// order. With no questions there are no rows and nothing to run.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub questions: Vec<Question>,
    pub rows: Vec<Row>,
    pub stats: Vec<OptionStats>,
    /// Every token the forward pass encodes: the metered input.
    pub input_tokens: usize,
    /// The id batched rows are padded with.
    pub pad_id: u32,
}

/// The text a state is tokenized from: a string as-is, anything else as
/// `json.dumps(state, ensure_ascii=False)`.
pub fn serialize_state(state: &Json) -> String {
    match state {
        Json::String(s) => s.clone(),
        other => other.dumps(),
    }
}

/// Validate a request's questions and build their model input.
///
/// `questions` are the request's definitions in request order; `cfg` and
/// `tok` belong to the checkpoint the request was routed to.
pub fn prepare(
    state: &Json,
    questions: &[(String, Json)],
    cfg: &DecisionConfig,
    tok: &dyn TextEncoder,
    budget: Budget,
) -> Result<Prepared, DecisionRequestError> {
    let special = tok.special();
    if questions.is_empty() {
        return Ok(Prepared {
            questions: Vec::new(),
            rows: Vec::new(),
            stats: Vec::new(),
            input_tokens: 0,
            pad_id: special.pad_id,
        });
    }
    let questions = questions_from_request(questions)?;
    let max_len = budget.max_len.unwrap_or(cfg.max_len);
    let head_max_len = budget.head_max_len.unwrap_or(cfg.head_max_len);
    let encode = |text: &str| {
        tok.encode(text)
            .map_err(|e| DecisionRequestError::new(500, format!("tokenizer failed: {e}")))
    };
    // Tokenized once and sliced per question.
    let state_ids = encode(&serialize_state(state).replace(&special.mask_token, " "))?;
    let truncate_left = matches!(state, Json::Array(_));
    let frame = Frame {
        cls: special.cls_id,
        sep: special.sep_id,
        max_len,
        head_max_len,
    };

    let mut rows = Vec::with_capacity(questions.len());
    let mut stats = Vec::with_capacity(questions.len());
    for question in &questions {
        let instructions = question.instructions.replace(&special.mask_token, " ");
        let head_ids = encode(&format!(
            "{} question: {instructions}",
            question.kind.name()
        ))?;
        let mut option_ids = Vec::with_capacity(question.options.len());
        for option in &question.options {
            let mut ids = vec![special.mask_id];
            let text = encode(&format!(" {}", option.replace(&special.mask_token, " ")))?;
            ids.extend(text.into_iter().take(MAX_OPTION_TOKENS));
            option_ids.push(ids);
        }
        let (row, s) = frame.assemble(head_ids, option_ids, &state_ids, truncate_left);
        if row.markers.len() != question.options.len() {
            return Err(unprocessable(format!(
                "question {} options exceed head_max_len={head_max_len}",
                Json::String(question.id.clone()).py_repr()
            )));
        }
        rows.push(Row {
            qtype: question.kind,
            ..row
        });
        stats.push(s);
    }
    let input_tokens = rows.iter().map(|r| r.input_ids.len()).sum();
    Ok(Prepared {
        questions,
        rows,
        stats,
        input_tokens,
        pad_id: special.pad_id,
    })
}

/// The fixed parts of every sequence for one request.
struct Frame {
    cls: u32,
    sep: u32,
    max_len: usize,
    head_max_len: usize,
}

impl Frame {
    /// The pure budget arithmetic of one sequence, over already-tokenized parts.
    fn assemble(
        &self,
        mut head_ids: Vec<u32>,
        mut option_ids: Vec<Vec<u32>>,
        state_ids: &[u32],
        truncate_left: bool,
    ) -> (Row, OptionStats) {
        let Frame {
            cls,
            sep,
            max_len,
            head_max_len,
        } = *self;
        let used = |o: &[Vec<u32>]| o.iter().map(Vec::len).sum::<usize>();
        // Signed: options can overrun the head budget.
        let mut opt_budget = head_max_len as isize - used(&option_ids) as isize;
        let mut tokens_per_option = None;
        if opt_budget < MIN_INSTRUCTION_ROOM as isize {
            let per = MIN_TOKENS_PER_OPTION
                .max(head_max_len.saturating_sub(MIN_INSTRUCTION_ROOM) / option_ids.len().max(1));
            tokens_per_option = Some(per);
            for o in &mut option_ids {
                o.truncate(per);
            }
            opt_budget = head_max_len as isize - used(&option_ids) as isize;
        }
        head_ids.truncate(opt_budget.max(MIN_INSTRUCTION_TOKENS as isize) as usize);

        let mut ids = Vec::with_capacity(max_len);
        ids.push(cls);
        ids.extend(head_ids);
        ids.push(sep);
        let mut markers = Vec::with_capacity(option_ids.len());
        for o in &option_ids {
            markers.push(ids.len());
            ids.extend_from_slice(o);
        }
        ids.push(sep);
        let room = max_len.saturating_sub(ids.len() + 1);
        let state = if truncate_left {
            &state_ids[state_ids.len().saturating_sub(room)..]
        } else {
            &state_ids[..room.min(state_ids.len())]
        };
        ids.extend_from_slice(state);
        ids.push(sep);
        ids.truncate(max_len);
        markers.retain(|&m| m < max_len);

        let mut distinct = option_ids.clone();
        distinct.sort();
        distinct.dedup();
        (
            Row {
                input_ids: ids,
                markers,
                qtype: QType::Choice,
            },
            OptionStats {
                options: option_ids.len(),
                options_distinct: distinct.len(),
                tokens_per_option,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLS: u32 = 1;
    const SEP: u32 = 2;
    const M: u32 = 9;

    fn frame(max_len: usize, head_max_len: usize) -> Frame {
        Frame {
            cls: CLS,
            sep: SEP,
            max_len,
            head_max_len,
        }
    }

    fn opts(lens: &[usize]) -> Vec<Vec<u32>> {
        lens.iter()
            .enumerate()
            .map(|(i, &n)| {
                let mut o = vec![M];
                o.extend((0..n - 1).map(|j| 100 + (i * 50 + j) as u32));
                o
            })
            .collect()
    }

    #[test]
    fn roomy_budget_keeps_everything() {
        let (row, s) = frame(64, 32).assemble(vec![10, 11], opts(&[2, 3]), &[50, 51], false);
        assert_eq!(
            row.input_ids,
            [CLS, 10, 11, SEP, M, 100, M, 150, 151, SEP, 50, 51, SEP]
        );
        assert_eq!(row.markers, [4, 6]);
        assert_eq!(
            (s.options, s.options_distinct, s.tokens_per_option),
            (2, 2, None)
        );
    }

    #[test]
    fn crowded_head_shrinks_options_evenly_and_keeps_eight_instruction_tokens() {
        // 3 options of 10 = 30 tokens against a 40-token head: 10 left < 16.
        let head: Vec<u32> = (10..40).collect();
        let (row, s) = frame(128, 40).assemble(head, opts(&[10, 10, 10]), &[], false);
        // per = max(4, (40 - 16) / 3) = 8; 40 - 24 = 16 instruction tokens.
        assert_eq!(s.tokens_per_option, Some(8));
        assert_eq!(row.markers, [18, 26, 34]);
        let (row, s) = frame(256, 40).assemble((10..40).collect(), opts(&[10; 12]), &[], false);
        // per = max(4, 24 / 12 = 2) = 4; the budget goes negative, 8 tokens survive.
        assert_eq!(s.tokens_per_option, Some(4));
        assert_eq!(row.markers[0], 1 + 8 + 1);
    }

    #[test]
    fn state_is_cut_from_the_side_the_state_shape_asks_for() {
        let state: Vec<u32> = (50..60).collect();
        let (right, _) = frame(8, 32).assemble(vec![], opts(&[1]), &state, false);
        let (left, _) = frame(8, 32).assemble(vec![], opts(&[1]), &state, true);
        // [CLS] [SEP] [MASK] [SEP] = 4 tokens, 1 reserved for the last [SEP]: room 3.
        assert_eq!(right.input_ids, [CLS, SEP, M, SEP, 50, 51, 52, SEP]);
        assert_eq!(left.input_ids, [CLS, SEP, M, SEP, 57, 58, 59, SEP]);
        // No room at all: none of the state, not all of it.
        let (none, _) = frame(5, 32).assemble(vec![], opts(&[1]), &state, true);
        assert_eq!(none.input_ids, [CLS, SEP, M, SEP, SEP]);
    }

    #[test]
    fn markers_past_max_len_are_dropped_and_collisions_counted() {
        let same = vec![vec![M, 7, 7], vec![M, 7, 7]];
        let (row, s) = frame(5, 64).assemble(vec![], same, &[], false);
        // [CLS] [SEP] then markers at 2 and 5; only the first fits in 5.
        assert_eq!(row.markers, [2]);
        assert_eq!((s.options, s.options_distinct), (2, 1));
    }
}
