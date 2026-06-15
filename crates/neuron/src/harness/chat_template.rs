//! Chat-template rendering for the model-supplied Jinja templates
//! HuggingFace tokenizers ship in `tokenizer_config.json`.
//!
//! ## Background
//!
//! Every modern open-weight model bundles a `chat_template` field
//! in its `tokenizer_config.json` — a Jinja2 template string that
//! converts a sequence of `{role, content}` messages into the
//! exact prompt the model was trained on. Examples:
//!
//! - Qwen3-Coder: `<|im_start|>{role}\n{content}<|im_end|>\n…`
//!   with conditional `enable_thinking` handling that injects an
//!   empty `<think>\n\n</think>` block when set false.
//! - DeepSeek-R1: similar im_start framing with different special-
//!   token names.
//! - Mistral / Magistral: a `[INST]` / `[/INST]` framing.
//! - Claude / Llama: another shape again.
//!
//! Rendering the model's own template is the only way to get the
//! *exact* prompt format the model was trained on plus the
//! model-specific kwargs (`enable_thinking`, `tools`, …) without
//! hardcoding per-model logic. The alternative — neuron's previous
//! `format_qwen3_prompt` — was a hardcoded Qwen3 ChatML glue that
//! ignored kwargs entirely.
//!
//! ## Scope
//!
//! This module is request-side only: it builds the prompt string
//! the tokenizer ingests before inference. The reasoning- and
//! tool-call-marker token routing (issues #6, #8) is response-side
//! and stays in `wire::openai_chat` / the streaming inference
//! loops.
//!
//! ## Fallback
//!
//! When the model's `tokenizer_config.json` is missing, doesn't
//! parse, lacks a `chat_template`, or renders an error, the caller
//! falls back to `format_qwen3_prompt`. The
//! `NEURON_USE_CHAT_TEMPLATE=false` env var is a global kill
//! switch — if a deploy goes sideways and the renderer is to
//! blame, an operator can flip the env and restart neuron without
//! shipping a new build.

use anyhow::{Context, Result};
use cortex_core::openai::{ChatMessage, MessageContent};
use minijinja::{Environment, Error as MjError, ErrorKind as MjErrorKind, Value as MjValue};
use serde_json::Value;
use std::path::Path;

/// Environment variable that, when set to `false`/`0`/`no`,
/// forces every model to skip its `chat_template` and fall back
/// to `format_qwen3_prompt`. Default (unset) is "use chat
/// templates where available".
pub const KILL_SWITCH_ENV: &str = "NEURON_USE_CHAT_TEMPLATE";

/// Read the global kill switch. `true` means chat templates are
/// enabled; `false` forces the fallback path everywhere.
pub fn chat_templates_enabled() -> bool {
    match std::env::var(KILL_SWITCH_ENV).ok().as_deref() {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "false" | "0" | "no" | "off"
        ),
        None => true,
    }
}

/// Probe for the model's chat template in the same directory the
/// tokenizer was loaded from, following HuggingFace `transformers`
/// precedence: a standalone `chat_template.jinja` (then
/// `chat_template.json`) wins over the `chat_template` field in
/// `tokenizer_config.json`.
///
/// This matters for multimodal models: Qwen3-VL / Qwen3.6 ship their
/// vision-aware template (the one that emits
/// `<|vision_start|><|image_pad|><|vision_end|>` per image) **only** in
/// `chat_template.jinja`, and may not ship a `tokenizer_config.json` at
/// all. Reading `tokenizer_config.json` alone returned `None`, which
/// dropped image content into the text-only `format_qwen3_prompt`
/// fallback — so image requests rendered zero `<|image_pad|>` tokens
/// and the vision path bailed on the count mismatch.
pub fn load_chat_template_alongside(tokenizer_json_path: &Path) -> Option<String> {
    let parent = tokenizer_json_path.parent()?;

    // 1. Standalone Jinja file — raw template text, highest priority.
    let jinja_path = parent.join("chat_template.jinja");
    match std::fs::read_to_string(&jinja_path) {
        Ok(text) if !text.trim().is_empty() => {
            tracing::info!(
                path = %jinja_path.display(),
                "chat_template: loaded standalone chat_template.jinja"
            );
            return Some(text);
        }
        Ok(_) => {
            tracing::warn!(
                path = %jinja_path.display(),
                "chat_template: chat_template.jinja present but empty; trying other sources"
            );
        }
        Err(_) => {} // absent — fall through, common case
    }

    // 2. Standalone JSON file — `{"chat_template": "..."}` form.
    let json_path = parent.join("chat_template.json");
    if json_path.exists()
        && let Some(t) = load_chat_template_from(&json_path)
    {
        tracing::info!(
            path = %json_path.display(),
            "chat_template: loaded standalone chat_template.json"
        );
        return Some(t);
    }

    // 3. The `chat_template` field inside tokenizer_config.json.
    let config_path = parent.join("tokenizer_config.json");
    load_chat_template_from(&config_path)
}

/// Best-effort load of `chat_template` from a HuggingFace
/// `tokenizer_config.json`. Returns `None` when the file is
/// absent, doesn't parse, or lacks the `chat_template` field —
/// in all of those cases the caller falls back to
/// `format_qwen3_prompt`. Warnings are logged so an operator can
/// see why the fallback fired.
pub fn load_chat_template_from(path: &Path) -> Option<String> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            tracing::debug!(
                path = %path.display(),
                error = %e,
                "chat_template: tokenizer_config.json absent or unreadable; falling back"
            );
            return None;
        }
    };
    let value: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "chat_template: tokenizer_config.json failed to parse; falling back"
            );
            return None;
        }
    };
    // Some tokenizer_config.json files carry `chat_template` as an
    // array of `{name, template}` objects (multi-template models —
    // tool-use variant, default variant). For now we pick the first
    // entry; future iterations could honour a name hint.
    match value.get("chat_template") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(arr)) => {
            for entry in arr {
                if let Some(t) = entry.get("template").and_then(|v| v.as_str()) {
                    return Some(t.to_string());
                }
            }
            tracing::warn!(
                path = %path.display(),
                "chat_template: array form had no usable template entry; falling back"
            );
            None
        }
        _ => None,
    }
}

/// Render the chat template into the prompt the model expects.
///
/// `template` is the raw Jinja string from `tokenizer_config.json`.
/// `messages` is the conversation in order. `kwargs` is the
/// `chat_template_kwargs` object the client supplied on the
/// request (or `Value::Null` when absent). The function expands
/// the kwargs into the Jinja context alongside the standard
/// `messages` and `add_generation_prompt` variables HF templates
/// expect.
///
/// `tools` is the request's `tools` array (or `Value::Null`).
/// Some chat templates iterate it to emit native tool definitions
/// (Qwen3-Coder's tool-use template, Mistral's [TOOL_DEFINITIONS]
/// frame). We forward whatever the client sent without
/// interpretation.
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage],
    tools: &Value,
    kwargs: &Value,
) -> Result<String> {
    let mut env = Environment::new();

    // HF chat templates are authored against Python's Jinja2 with its
    // string semantics. Bridge the two so real model templates render:
    //
    // - `pycompat::unknown_method_callback` supplies Python str/list/dict
    //   methods minijinja lacks natively (`startswith`, `endswith`,
    //   `split`, `rstrip`, `lstrip`, …) — the Qwen3.6 template uses
    //   several in its think-block and tool-response handling.
    // - `raise_exception` is the global HF templates call to reject
    //   malformed inputs (e.g. an image in a system message). Map it to
    //   a render error so the caller falls back / surfaces it.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<MjValue, MjError> {
            Err(MjError::new(MjErrorKind::InvalidOperation, msg))
        },
    );

    // Compile the template against a fixed name so error messages
    // surface "chat_template" rather than `<template>`.
    env.add_template("chat_template", template)
        .context("compile chat_template")?;
    let tmpl = env.get_template("chat_template").unwrap();

    // Convert our internal ChatMessage shape into the
    // `[{role, content}]` shape HF templates iterate. Text content
    // becomes a string; Parts becomes an array of content blocks.
    // The HF templates handle both shapes via `content is string`
    // checks or content-array iteration.
    let mut messages_json: Vec<Value> = messages
        .iter()
        .map(|m| {
            let content_value = match &m.content {
                MessageContent::Text(s) => Value::String(s.clone()),
                MessageContent::Parts(parts) => Value::Array(parts.clone()),
            };
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Value::String(m.role.clone()));
            obj.insert("content".into(), content_value);
            // Forward extras (e.g. tool_calls on assistant turns,
            // tool_call_id on tool result turns). HF templates that
            // need them read e.g. `message.tool_calls`.
            if let Value::Object(extras) = &m.extra {
                for (k, v) in extras {
                    obj.insert(k.clone(), v.clone());
                }
            }
            Value::Object(obj)
        })
        .collect();

    // OpenAI clients (opencode, the OpenAI SDK) carry tool-call
    // `arguments` as a JSON *string*; Qwen3.6's template iterates it as a
    // dict, so normalise string args to objects before rendering. Without
    // this, `chat_template:120` errors "cannot convert value into pairs".
    normalize_tool_call_arguments(&mut messages_json);

    // Build the kwargs context. Add base bindings the template
    // expects (`messages`, `add_generation_prompt`, `tools`) plus
    // anything the caller passed in `chat_template_kwargs`. Caller
    // kwargs override the defaults so `add_generation_prompt: false`
    // from the request actually wins.
    let mut ctx_map = serde_json::Map::new();
    ctx_map.insert("messages".into(), Value::Array(messages_json));
    ctx_map.insert("add_generation_prompt".into(), Value::Bool(true));
    if !tools.is_null() {
        ctx_map.insert("tools".into(), tools.clone());
    }
    if let Value::Object(kwargs_obj) = kwargs {
        for (k, v) in kwargs_obj {
            ctx_map.insert(k.clone(), v.clone());
        }
    }
    // `Template::render` takes any Serialize value; serde_json's
    // `Value` implements it natively, so we pass the assembled
    // context object directly without going through the
    // `context!` macro (which expects minijinja-native values).
    tmpl.render(Value::Object(ctx_map))
        .context("render chat_template")
}

/// Normalize OpenAI-style tool-call `arguments` from JSON strings to
/// objects, in place, across all messages.
///
/// The OpenAI wire format carries `tool_calls[].function.arguments` as a
/// JSON *string*; HF chat templates (Qwen3.6 at `chat_template:120`)
/// iterate it as a dict (`arguments | items`), which throws "cannot
/// convert value into pairs" on a string. Parsing string args into the
/// object the template expects lets OpenAI and Anthropic clients both
/// render. A string that doesn't parse is left untouched — the render
/// then fails loudly rather than silently (see
/// `InferenceError::TemplateRenderFailed`).
fn normalize_tool_call_arguments(messages: &mut [Value]) {
    for msg in messages {
        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for tc in tool_calls {
            let Some(func) = tc.get_mut("function").and_then(Value::as_object_mut) else {
                continue;
            };
            let parsed = match func.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str::<Value>(s).ok(),
                _ => None,
            };
            if let Some(p) = parsed {
                func.insert("arguments".into(), p);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Reproduces the Qwen3.6 vision template's image-insertion
    /// condition against the OpenAI `image_url` content-part shape our
    /// renderer forwards. Confirms minijinja's `'image_url' in item`
    /// matches a serde_json object that carries that key — i.e. the
    /// template *can* emit `<|image_pad|>` for our parts.
    #[test]
    fn image_url_part_renders_image_pad() {
        // Condition copied from doc/vision-qwen3_6-spec.md (lines 8-18
        // of the real chat_template.jinja).
        let template = "{%- for message in messages -%}\
{%- if message.content is string -%}\
{{ message.content }}\
{%- else -%}\
{%- for item in message.content -%}\
{%- if 'image' in item or 'image_url' in item or item.type == 'image' -%}\
<|vision_start|><|image_pad|><|vision_end|>\
{%- elif item.type == 'text' -%}\
{{ item.text }}\
{%- endif -%}\
{%- endfor -%}\
{%- endif -%}\
{%- endfor -%}";
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                json!({"type": "text", "text": "what is this?"}),
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}}),
            ]),
            extra: Value::Object(Default::default()),
        }];
        let out = render_chat_template(template, &messages, &Value::Null, &Value::Null)
            .expect("render should succeed");
        assert!(
            out.contains("<|image_pad|>"),
            "expected the image_url part to emit <|image_pad|>; rendered: {out:?}"
        );
    }

    /// `chat_template.jinja` must win over `tokenizer_config.json`'s
    /// `chat_template` field — the transformers precedence Qwen3.6
    /// relies on (its vision template ships only in the `.jinja` file).
    #[test]
    fn standalone_jinja_template_takes_precedence() {
        let dir = std::env::temp_dir().join(format!(
            "neuron_ct_precedence_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "FROM_JINJA").unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": "FROM_CONFIG"}"#,
        )
        .unwrap();
        // tokenizer_json_path is the sibling the loader takes a parent of.
        let got = load_chat_template_alongside(&dir.join("tokenizer.json"));
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(got.as_deref(), Some("FROM_JINJA"));
    }

    /// With no standalone file, fall back to the tokenizer_config.json
    /// field — the text-only path stays unchanged.
    #[test]
    fn falls_back_to_tokenizer_config_when_no_standalone() {
        let dir = std::env::temp_dir().join(format!(
            "neuron_ct_fallback_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": "FROM_CONFIG"}"#,
        )
        .unwrap();
        let got = load_chat_template_alongside(&dir.join("tokenizer.json"));
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(got.as_deref(), Some("FROM_CONFIG"));
    }

    /// The *actual* Qwen3.6-27B `chat_template.jinja` (verbatim from
    /// beast's HF cache) must render in minijinja and emit exactly one
    /// `<|image_pad|>` for a text+image user turn. This is the real
    /// end-to-end check the unit tests above only approximate — it
    /// catches any minijinja incompatibility (namespace, macros,
    /// reverse slice, string methods) before it reaches production.
    #[test]
    fn real_qwen3_6_template_renders_one_image_pad() {
        let template = include_str!("testdata/qwen3_6_chat_template.jinja");
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                json!({"type": "text", "text": "what is this?"}),
                json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA="}}),
            ]),
            extra: Value::Object(Default::default()),
        }];
        let out = render_chat_template(template, &messages, &Value::Null, &Value::Null)
            .expect("real Qwen3.6 template should render in minijinja");
        let pads = out.matches("<|image_pad|>").count();
        assert_eq!(
            pads, 1,
            "expected exactly one <|image_pad|>; rendered:\n{out}"
        );
        assert!(out.contains("<|vision_start|>") && out.contains("<|vision_end|>"));
    }

    fn user_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: MessageContent::Text(text.into()),
            extra: Value::Object(Default::default()),
        }
    }

    fn assistant_msg(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: MessageContent::Text(text.into()),
            extra: Value::Object(Default::default()),
        }
    }

    /// Minimal Qwen3-style template — enough surface to confirm
    /// our renderer threads role + content correctly without
    /// loading a real model's tokenizer_config.json.
    const QWEN3_LIKE: &str = "{%- for message in messages -%}\
<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n\
{%- endfor -%}\
{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

    #[test]
    fn renders_basic_conversation() {
        let prompt = render_chat_template(
            QWEN3_LIKE,
            &[user_msg("hello"), assistant_msg("hi"), user_msg("bye")],
            &Value::Null,
            &Value::Null,
        )
        .unwrap();
        // Structural assertions — the exact whitespace produced
        // by a given template is a Jinja-trim concern that varies
        // per real chat_template. What matters is that every
        // turn's role + content thread through in order, and that
        // the generation cue lands at the end.
        assert!(
            prompt.contains("<|im_start|>user\nhello<|im_end|>"),
            "first user turn missing: {prompt}"
        );
        assert!(
            prompt.contains("<|im_start|>assistant\nhi<|im_end|>"),
            "assistant turn missing: {prompt}"
        );
        assert!(
            prompt.contains("<|im_start|>user\nbye<|im_end|>"),
            "second user turn missing: {prompt}"
        );
        assert!(
            prompt.ends_with("<|im_start|>assistant")
                || prompt.ends_with("<|im_start|>assistant\n"),
            "generation cue missing at end: {prompt}"
        );
    }

    #[test]
    fn kwargs_are_threaded_into_template_context() {
        // Replica of Qwen3's enable_thinking branch in
        // simplified form. When the kwarg is false, the model's
        // template injects an empty `<think>...</think>` block
        // before the generation cue — pre-filling the model's
        // reasoning slot with "no thinking" so the model emits
        // the answer directly.
        let template = "{%- if enable_thinking is defined and enable_thinking is false -%}\
NO_THINK\
{%- else -%}\
THINK_OK\
{%- endif -%}";
        let r_disabled = render_chat_template(
            template,
            &[],
            &Value::Null,
            &json!({ "enable_thinking": false }),
        )
        .unwrap();
        assert_eq!(r_disabled, "NO_THINK");
        let r_default = render_chat_template(template, &[], &Value::Null, &Value::Null).unwrap();
        assert_eq!(r_default, "THINK_OK");
    }

    #[test]
    fn missing_template_field_returns_none() {
        let tmp = std::env::temp_dir().join("neuron-test-tokenizer-missing-field.json");
        std::fs::write(&tmp, r#"{"some_other_field": 1}"#).unwrap();
        assert!(load_chat_template_from(&tmp).is_none());
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn load_template_from_string_field() {
        let tmp = std::env::temp_dir().join("neuron-test-tokenizer-string.json");
        std::fs::write(
            &tmp,
            r#"{"chat_template": "hello {{ messages[0].content }}"}"#,
        )
        .unwrap();
        let t = load_chat_template_from(&tmp).expect("template loaded");
        assert!(t.contains("messages[0].content"));
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn load_template_from_array_form() {
        // Some HF models ship `chat_template` as `[{name, template}, ...]`.
        let tmp = std::env::temp_dir().join("neuron-test-tokenizer-array.json");
        std::fs::write(
            &tmp,
            r#"{"chat_template": [{"name": "default", "template": "ARR"}]}"#,
        )
        .unwrap();
        let t = load_chat_template_from(&tmp).expect("template loaded");
        assert_eq!(t, "ARR");
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn missing_file_returns_none_quietly() {
        let absent = std::path::PathBuf::from("/definitely/not/a/real/path.json");
        assert!(load_chat_template_from(&absent).is_none());
    }

    #[test]
    fn unparseable_returns_none() {
        let tmp = std::env::temp_dir().join("neuron-test-tokenizer-garbage.json");
        std::fs::write(&tmp, b"{not valid json").unwrap();
        assert!(load_chat_template_from(&tmp).is_none());
        let _ = std::fs::remove_file(tmp);
    }

    #[test]
    fn kill_switch_recognises_truthy_falsy_values() {
        // Test against the actual env var so callers see the
        // same behaviour as production. Serialise via a
        // mutex — see path_util.rs for the pattern.
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap();
        let prior = std::env::var(KILL_SWITCH_ENV).ok();
        unsafe {
            std::env::remove_var(KILL_SWITCH_ENV);
        }
        assert!(chat_templates_enabled());
        for value in ["false", "0", "no", "off", "FALSE", "  no  "] {
            unsafe { std::env::set_var(KILL_SWITCH_ENV, value) };
            assert!(!chat_templates_enabled(), "value {value:?} should disable");
        }
        for value in ["true", "1", "yes", ""] {
            unsafe { std::env::set_var(KILL_SWITCH_ENV, value) };
            assert!(chat_templates_enabled(), "value {value:?} should enable");
        }
        unsafe {
            match prior {
                Some(p) => std::env::set_var(KILL_SWITCH_ENV, p),
                None => std::env::remove_var(KILL_SWITCH_ENV),
            }
        }
    }

    #[test]
    fn message_extras_thread_through_for_tool_calls() {
        // HF templates read assistant.tool_calls and tool
        // turns' tool_call_id. Confirm our extras flatten into
        // the message object the template iterates.
        let mut extras = serde_json::Map::new();
        extras.insert(
            "tool_calls".into(),
            json!([{"id": "t1", "function": {"name": "x", "arguments": "{}"}}]),
        );
        let msg = ChatMessage {
            role: "assistant".into(),
            content: MessageContent::Text(String::new()),
            extra: Value::Object(extras),
        };
        let template = "{{ messages[0].tool_calls[0].id }}";
        let rendered = render_chat_template(template, &[msg], &Value::Null, &Value::Null).unwrap();
        assert_eq!(rendered, "t1");
    }

    #[test]
    fn normalizes_openai_string_tool_call_arguments_to_object() {
        // The opencode / OpenAI-SDK shape: arguments as a JSON string.
        let mut messages = vec![json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "Read", "arguments": "{\"path\":\"/x\"}"}
            }]
        })];
        normalize_tool_call_arguments(&mut messages);
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            json!({"path": "/x"}),
            "string args must become the object the template iterates"
        );
    }

    #[test]
    fn leaves_object_args_and_non_tool_messages_untouched() {
        let mut messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": {"a": 1}}}
            ]}),
        ];
        normalize_tool_call_arguments(&mut messages);
        // Already-object args pass through unchanged (Anthropic path).
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            json!({"a": 1})
        );
        // Ordinary messages are not disturbed.
        assert_eq!(messages[0]["content"], "hi");
    }
}
