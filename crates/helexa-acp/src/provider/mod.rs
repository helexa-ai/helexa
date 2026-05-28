//! Provider trait — the seam between the ACP-side agent loop and
//! whatever wire protocol an endpoint actually speaks.
//!
//! Every concrete provider (OpenAI chat completions, OpenAI Responses,
//! Anthropic /v1/messages, Ollama native, …) implements
//! [`Provider`]. The agent constructs a [`CompletionRequest`] using
//! provider-agnostic types and consumes a stream of
//! [`CompletionEvent`]s — neither end knows which wire format is on
//! the other side of the trait.
//!
//! Day-1 provider: [`openai_chat::OpenAIChatProvider`]. Day-N
//! providers slot in without touching `agent.rs`.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

pub mod openai_chat;

/// Provider-agnostic LLM endpoint. Implementations translate between
/// [`CompletionRequest`] / [`CompletionEvent`] and whatever wire
/// format their endpoint speaks.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Endpoint name as configured by the user (e.g. `"helexa"`,
    /// `"openrouter"`). Used in logs and in the `endpoint:model`
    /// selector.
    fn name(&self) -> &str;

    /// List models available at this endpoint. Used to build the
    /// model-picker dropdown in editor clients (Stage 4). Should
    /// return quickly (cache if necessary).
    #[allow(dead_code)]
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>>;

    /// Run a chat completion. Returns a stream of provider-agnostic
    /// events. The stream stops when the upstream finishes, when
    /// `cancel` is fired, or when the stream is dropped.
    async fn complete(
        &self,
        request: CompletionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<CompletionEvent>>>;
}

/// One model exposed by a provider. Constructed by `list_models` —
/// Stage 4 is when the agent loop starts consuming it for the
/// model-picker dropdown.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// Human-friendly name, if the endpoint exposes one. Otherwise
    /// `id` is used as the display name.
    #[serde(default)]
    pub display_name: Option<String>,
}

/// Inputs to a completion. Provider-agnostic — concrete providers
/// translate this into their wire format.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    /// Endpoint-local model id (without the `endpoint:` prefix).
    pub model: String,
    pub messages: Vec<Message>,
    /// Tools the model is allowed to call. Empty list means no tool
    /// support advertised.
    pub tools: Vec<ToolSpec>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool result message. Provider impls turn this into whatever
    /// shape the upstream wire format wants (OpenAI uses
    /// `role: "tool"` + `tool_call_id`; Anthropic uses content blocks).
    /// Stage 3 (tools) constructs this; Stage 2 never does.
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    /// Plain text turn (system / user / assistant). Struct variant
    /// rather than newtype so the persisted JSON has an explicit
    /// `text` field — that lets us use internal tagging on the
    /// enum, which is incompatible with newtype-of-primitive
    /// variants.
    Text { text: String },
    /// Assistant turn that called one or more tools. Stage 3 starts
    /// constructing this when the provider stream yields a
    /// `ToolCallStart` / `ToolCallArgsDelta` sequence.
    ToolCalls {
        /// Optional text the assistant said alongside the tool calls.
        text: Option<String>,
        calls: Vec<ToolCall>,
    },
    /// Tool result. `tool_call_id` matches the assistant's call id.
    /// Stage 3 constructs this after the tool runner finishes.
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned id that ties the call to its result. The
    /// Qwen3 wire format we use today doesn't carry this on the
    /// model side (calls and results are matched positionally inside
    /// a turn), so the field looks unused in the prod build — but it
    /// flows through to `MessageContent::ToolResult.tool_call_id` for
    /// history bookkeeping and a future strict-OpenAI backend will
    /// consume it directly.
    #[allow(dead_code)]
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments. Kept as a string because providers
    /// stream argument bytes incrementally and only validate at the
    /// end; the agent decodes once the call is complete.
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
}

/// Events emitted by a provider during a streaming completion.
#[derive(Debug, Clone)]
pub enum CompletionEvent {
    /// Incremental visible text from the assistant.
    TextDelta(String),
    /// Incremental "reasoning" / thought text, if the model emits one
    /// (e.g. Qwen3 with `<think>` tags surfaced as a separate stream,
    /// or OpenAI reasoning models).
    ReasoningDelta(String),
    /// A new tool call has started. Stage 2 ignores the payload; the
    /// agent loop in Stage 3 reads `index` to correlate with
    /// [`Self::ToolCallArgsDelta`], `id` for the eventual tool-result
    /// turn, and `name` to dispatch the runner.
    #[allow(dead_code)]
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// More argument bytes for a tool call already announced via
    /// [`Self::ToolCallStart`]. Stage 2 ignores; Stage 3 accumulates
    /// the bytes by `index` until the call's arguments are complete.
    #[allow(dead_code)]
    ToolCallArgsDelta { index: usize, args_delta: String },
    /// A `<tool_call>` block whose JSON couldn't be parsed even with
    /// the qwen3 module's repair attempts. The agent surfaces this
    /// as a Failed `SessionUpdate::ToolCall` card with the raw body
    /// visible (so the editor renders structured failure UI rather
    /// than dumping the body inline in the message pane), and feeds
    /// a synthetic tool-error message back into history so the
    /// model can self-correct on the next round.
    MalformedToolCall { raw: String },
    /// Stream finished. Carries the upstream `finish_reason` if it
    /// gave one (`"stop"`, `"length"`, `"tool_calls"`, …).
    Finish { reason: Option<String> },
    /// Final usage stats, if the provider supplied them. Stage 2
    /// matches the variant to drop it; Stage 6b (token metrics) is
    /// when the payload starts being read.
    #[allow(dead_code)]
    Usage(UsageStats),
}

/// Token accounting reported by the provider at the end of a stream.
/// Stage 2 doesn't surface usage anywhere — the stable `PromptResponse`
/// has no usage field, and the unstable variant is gated. Stage 6b
/// turns these on with Prometheus metrics.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageStats {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}
