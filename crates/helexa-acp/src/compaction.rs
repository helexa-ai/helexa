//! Rolling-conversation compaction for small-context local models.
//!
//! The tool-call loop in [`crate::agent`] grows the message vec it
//! sends upstream every round. On a frontier model that's fine; on a
//! 32 K Qwen3 the first few `read_file` results can push the prompt
//! past the model's context window, at which point cortex/neuron
//! refuses with `prompt_too_long` and the whole turn dies. Long-form
//! local agents are unusable without something here.
//!
//! Strategy (intentionally simple — no LLM-summarization round-trip,
//! no tokenizer dependency):
//!
//! 1. **Protect** the things the model cannot reason without:
//!    - The system prompt (idx 0).
//!    - Every `Role::User` turn (the user's intent — irreplaceable).
//!    - The last [`KEEP_TAIL`] messages (most recent rounds stay
//!      verbatim so the model can keep working on what it just
//!      observed).
//! 2. **Elide** older `Role::Assistant` prose and older `Role::Tool`
//!    result content. The structure stays — `tool_call_id`s, tool
//!    names, and argument JSON survive intact — so OpenAI's strict
//!    `tool_calls` ↔ `tool` pairing schema remains satisfied. Only
//!    the *payload* shrinks to a one-line marker.
//! 3. Walk oldest→newest, recomputing the budget after each elision.
//!    Stop as soon as we fit; we don't compact more than necessary.
//! 4. If we still exceed budget after eliding everything we're
//!    allowed to, return what we have. The upstream will surface a
//!    `prompt_too_long` error and the user can intervene; that's
//!    better than silently dropping content the model needs.
//!
//! Token estimation uses a `chars / 3.5` heuristic — conservative
//! (over-estimates tokens slightly) so we compact a touch early
//! rather than a touch late.

use crate::provider::{Message, MessageContent, Role};

/// Most-recent N messages that are never elided. Roughly "the
/// current tool round in flight" — assistant turn that called the
/// tools + each tool result + a bit of slack.
const KEEP_TAIL: usize = 4;

/// Below this content size we don't bother eliding — the savings
/// don't outweigh the loss of detail. Roughly 60–80 tokens.
const ELIDE_MIN_CHARS: usize = 256;

/// Roughly tokens-per-character for English + code mixed in. The
/// actual per-tokenizer ratio varies (GPT-4o ≈ 4 chars/token on
/// English prose, ≈ 3 chars/token on code-heavy text). We pick a
/// value on the conservative end so the budget check fires *before*
/// the upstream tokenizer says no.
const CHARS_PER_TOKEN: f32 = 3.5;

/// Per-message envelope overhead (role + JSON framing). Comes out
/// to a few tokens; tiny but it adds up across long histories.
const ENVELOPE_TOKENS: usize = 8;

/// Stats reported back from [`compact_to_budget`] for the caller to
/// log. The numbers are estimates (see [`estimate_tokens`]), so
/// don't compare them to upstream-reported token counts as if they
/// were exact.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactionStats {
    /// Estimated tokens in the input messages.
    pub original_tokens: usize,
    /// Estimated tokens after compaction. Equal to `original_tokens`
    /// when no compaction was needed.
    pub final_tokens: usize,
    /// Number of messages whose content was elided. Zero is the
    /// hot path (nothing to do).
    pub elided_messages: usize,
}

impl CompactionStats {
    fn unchanged(tokens: usize) -> Self {
        Self {
            original_tokens: tokens,
            final_tokens: tokens,
            elided_messages: 0,
        }
    }
}

/// Approximate token count for one message. Sums the textual
/// payload's chars, divides by [`CHARS_PER_TOKEN`], and adds an
/// envelope constant. Cheap (no allocation) so safe to call once per
/// message per round.
pub fn estimate_tokens(msg: &Message) -> usize {
    let chars = match &msg.content {
        MessageContent::Text { text } => text.len(),
        MessageContent::ToolCalls { text, calls } => {
            let txt = text.as_deref().map(|s| s.len()).unwrap_or(0);
            let calls_size: usize = calls
                .iter()
                .map(|c| c.name.len() + c.arguments.len() + c.id.len())
                .sum();
            txt + calls_size
        }
        MessageContent::ToolResult {
            tool_call_id,
            content,
        } => tool_call_id.len() + content.len(),
    };
    ((chars as f32 / CHARS_PER_TOKEN) as usize) + ENVELOPE_TOKENS
}

/// Sum of [`estimate_tokens`] across all messages.
pub fn total_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_tokens).sum()
}

/// Project `messages` into a vec whose estimated token count fits in
/// `budget` tokens. Returns the projection plus stats about what
/// was done. When the input already fits, the projection is a clone
/// of the input and stats report zero elisions.
///
/// See module docs for the strategy and protected set.
pub fn compact_to_budget(messages: &[Message], budget: usize) -> (Vec<Message>, CompactionStats) {
    let original = total_tokens(messages);
    if original <= budget {
        return (messages.to_vec(), CompactionStats::unchanged(original));
    }

    let mut out = messages.to_vec();
    let len = out.len();
    let tail_start = len.saturating_sub(KEEP_TAIL);
    let mut elided = 0usize;

    // Two passes. First pass: ToolResult contents (largest savings
    // per elision — read_file payloads land here). Second pass: long
    // Assistant prose. We don't interleave because eliding a long
    // assistant turn before a really old read_file would do less
    // good per elision; oldest-first ordering is enforced *within*
    // each pass instead.
    for pass in 0..2 {
        for i in 1..tail_start {
            if matches!(out[i].role, Role::User) {
                continue;
            }
            let target_pass_2 = matches!(
                &out[i].content,
                MessageContent::Text { .. } | MessageContent::ToolCalls { .. }
            );
            let target_pass_1 = matches!(&out[i].content, MessageContent::ToolResult { .. });
            let in_pass = (pass == 0 && target_pass_1) || (pass == 1 && target_pass_2);
            if !in_pass {
                continue;
            }
            if elide_in_place(&mut out[i]) {
                elided += 1;
                if total_tokens(&out) <= budget {
                    let final_tokens = total_tokens(&out);
                    return (
                        out,
                        CompactionStats {
                            original_tokens: original,
                            final_tokens,
                            elided_messages: elided,
                        },
                    );
                }
            }
        }
    }

    let final_tokens = total_tokens(&out);
    (
        out,
        CompactionStats {
            original_tokens: original,
            final_tokens,
            elided_messages: elided,
        },
    )
}

/// Shrink one message's payload while keeping its structural role
/// (so tool_call_id pairing survives). Returns `true` when the
/// message changed.
///
/// - `ToolResult.content` → `(elided: N bytes of tool result)`
/// - `ToolCalls.text`     → `(elided: N bytes of assistant prose)`
/// - `Text` (assistant)   → `(elided: N bytes of assistant prose)`
///
/// Already-tiny payloads are skipped — eliding a 50-byte string
/// would *grow* it once the marker is in place.
fn elide_in_place(msg: &mut Message) -> bool {
    match &mut msg.content {
        MessageContent::ToolResult { content, .. } => {
            if content.len() < ELIDE_MIN_CHARS {
                return false;
            }
            *content = format!("(elided: {} bytes of tool result)", content.len());
            true
        }
        MessageContent::ToolCalls { text, .. } => match text {
            Some(t) if t.len() >= ELIDE_MIN_CHARS => {
                *text = Some(format!("(elided: {} bytes of assistant prose)", t.len()));
                true
            }
            _ => false,
        },
        MessageContent::Text { text } => {
            if text.len() < ELIDE_MIN_CHARS {
                return false;
            }
            *text = format!("(elided: {} bytes of assistant prose)", text.len());
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ToolCall;

    fn sys(text: &str) -> Message {
        Message {
            role: Role::System,
            content: MessageContent::Text { text: text.into() },
        }
    }
    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: MessageContent::Text { text: text.into() },
        }
    }
    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: MessageContent::Text { text: text.into() },
        }
    }
    fn assistant_calls(text: Option<&str>, name: &str, args: &str, id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: MessageContent::ToolCalls {
                text: text.map(|s| s.to_string()),
                calls: vec![ToolCall {
                    id: id.into(),
                    name: name.into(),
                    arguments: args.into(),
                }],
            },
        }
    }
    fn tool_result(id: &str, body: &str) -> Message {
        Message {
            role: Role::Tool,
            content: MessageContent::ToolResult {
                tool_call_id: id.into(),
                content: body.into(),
            },
        }
    }

    #[test]
    fn under_budget_is_a_no_op_clone() {
        let msgs = vec![sys("you are an agent"), user("hi"), assistant_text("hello")];
        let (out, stats) = compact_to_budget(&msgs, 10_000);
        assert_eq!(stats.elided_messages, 0);
        assert_eq!(stats.original_tokens, stats.final_tokens);
        assert_eq!(out.len(), msgs.len());
        // Strings unchanged.
        match &out[2].content {
            MessageContent::Text { text } => assert_eq!(text, "hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn elides_old_tool_result_before_old_assistant_prose() {
        // History: sys, user, assistant_calls, big_tool_result,
        //          assistant_with_big_text, user, assistant_calls,
        //          small_tool_result.
        // KEEP_TAIL=4 protects the last four; the big tool result
        // sits in the prunable range and should go first because
        // pass 0 (tool results) runs before pass 1 (prose).
        let big_result = "X".repeat(4096);
        let big_prose = "Y".repeat(2048);
        let msgs = vec![
            sys("preamble"),
            user("first ask"),
            assistant_calls(None, "read_file", r#"{"path":"/a"}"#, "c0"),
            tool_result("c0", &big_result),
            assistant_text(&big_prose),
            user("follow up"),
            assistant_calls(None, "read_file", r#"{"path":"/b"}"#, "c1"),
            tool_result("c1", "short result body"),
        ];
        let before = total_tokens(&msgs);
        // Force compaction by setting budget well below current.
        let budget = before / 2;
        let (out, stats) = compact_to_budget(&msgs, budget);

        assert!(
            stats.elided_messages >= 1,
            "expected at least one elision, got {stats:?}"
        );
        // The big tool result must be elided (oldest fat target).
        match &out[3].content {
            MessageContent::ToolResult { content, .. } => {
                assert!(
                    content.starts_with("(elided:"),
                    "tool result not elided: {content:?}"
                );
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
        // Last four messages must be untouched.
        assert!(matches!(
            &out[out.len() - 1].content,
            MessageContent::ToolResult { content, .. } if content == "short result body"
        ));
    }

    #[test]
    fn never_elides_system_or_user_turns() {
        let big_user = "U".repeat(8192);
        let msgs = vec![sys("preamble"), user(&big_user), assistant_text("ok")];
        let budget = 10; // way below — forces all possible elision
        let (out, _stats) = compact_to_budget(&msgs, budget);
        // System unchanged.
        match &out[0].content {
            MessageContent::Text { text } => assert_eq!(text, "preamble"),
            other => panic!("expected Text, got {other:?}"),
        }
        // User unchanged even though it's huge.
        match &out[1].content {
            MessageContent::Text { text } => assert_eq!(text.len(), big_user.len()),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn preserves_tool_call_id_pairing_after_elision() {
        // OpenAI strict mode rejects a tool-result whose tool_call_id
        // doesn't match a preceding assistant tool_call. Elision
        // must not break that linkage.
        let big = "Z".repeat(4096);
        let msgs = vec![
            sys("preamble"),
            user("first"),
            assistant_calls(None, "read_file", r#"{"path":"/a"}"#, "call_42"),
            tool_result("call_42", &big),
            // Tail messages.
            user("next"),
            assistant_calls(None, "read_file", r#"{"path":"/b"}"#, "call_43"),
            tool_result("call_43", "ok"),
            assistant_text("done"),
        ];
        let budget = total_tokens(&msgs) / 3;
        let (out, _stats) = compact_to_budget(&msgs, budget);
        // The assistant call and its result both carry call_42.
        let call_id = match &out[2].content {
            MessageContent::ToolCalls { calls, .. } => calls[0].id.clone(),
            other => panic!("expected ToolCalls, got {other:?}"),
        };
        match &out[3].content {
            MessageContent::ToolResult { tool_call_id, .. } => {
                assert_eq!(tool_call_id, &call_id, "pairing broken");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn estimate_tokens_grows_with_content() {
        let small = sys("hi");
        let large = sys(&"x".repeat(10_000));
        assert!(estimate_tokens(&large) > estimate_tokens(&small) * 100);
    }

    #[test]
    fn elide_in_place_skips_short_content() {
        let mut m = tool_result("c0", "tiny");
        assert!(!elide_in_place(&mut m));
        match m.content {
            MessageContent::ToolResult { content, .. } => assert_eq!(content, "tiny"),
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn returns_best_effort_when_budget_unmeetable() {
        // Single huge user message that cannot be elided. Budget 10.
        // We don't error — we return what we have and let upstream
        // refuse the prompt with its own error.
        let big_user = "U".repeat(100_000);
        let msgs = vec![sys("preamble"), user(&big_user)];
        let (out, stats) = compact_to_budget(&msgs, 10);
        assert_eq!(out.len(), msgs.len());
        assert!(stats.final_tokens > 10, "still over budget by design");
    }
}
