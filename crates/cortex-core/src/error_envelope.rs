//! The OpenAI-standard error envelope (#60) and the rejection contract
//! that rides on it (#63).
//!
//! Every non-2xx response cortex and neuron emit uses the shape
//!
//! ```json
//! { "error": { "message": "...", "type": "...", "code": "...", "param": null } }
//! ```
//!
//! because OpenAI-compatible clients (opencode, the AI SDK, litellm, the
//! OpenAI SDKs) read `error.type` / `error.code` to decide what to do —
//! most importantly `code == "context_length_exceeded"` triggers
//! auto-compaction, and a `429` with `Retry-After` makes them back off and
//! retry rather than surfacing an opaque failure. A flat `{"error":"..."}`
//! string is invisible to that logic.
//!
//! This module is the single source of truth for that envelope. It is
//! deliberately **axum-agnostic** — cortex-core is a pure types crate — so
//! it carries the response as data (`status`, `body()`, `retry_after_secs`)
//! and each HTTP crate (cortex-gateway, neuron) owns a tiny adapter that
//! turns an [`OpenAiError`] into its framework's response type, setting the
//! `Retry-After` header when present.
//!
//! Retryable conditions **must** carry `Retry-After` (per #63). The named
//! constructors below encode that: [`OpenAiError::rate_limit_exceeded`] and
//! [`OpenAiError::service_unavailable`] take a retry hint;
//! [`OpenAiError::insufficient_quota`] (hard balance, no reset) and
//! [`OpenAiError::context_length_exceeded`] / [`OpenAiError::invalid_api_key`]
//! (permanent) do not. `402 Payment Required` is banned by the contract — use
//! `429 insufficient_quota` for hard budget exhaustion.

use serde_json::{Map, Value, json};

/// A rejection rendered in the OpenAI error envelope.
///
/// Build with [`OpenAiError::new`] (or a named constructor), refine with the
/// `with_*` builders, then hand to the consuming crate's adapter to turn into
/// an HTTP response.
#[derive(Debug, Clone)]
pub struct OpenAiError {
    /// HTTP status code (e.g. `401`, `429`, `503`).
    pub status: u16,
    /// Broad OpenAI category — `"invalid_request_error"`, `"api_error"`,
    /// `"rate_limit_error"`, …
    pub error_type: String,
    /// Specific machine-readable code clients key on (`"invalid_api_key"`,
    /// `"rate_limit_exceeded"`, `"context_length_exceeded"`, …). `None`
    /// renders as JSON `null`.
    pub code: Option<String>,
    /// Human-readable, actionable message.
    pub message: String,
    /// OpenAI's `param` field — the offending request parameter, if any.
    pub param: Option<String>,
    /// Seconds to advertise in the `Retry-After` header. Set only on
    /// retryable conditions; `None` means no header.
    pub retry_after_secs: Option<u64>,
    /// Diagnostic fields merged *inside* the `error` object (e.g.
    /// `prompt_len`, `max`, `free_mb`) so they don't break the envelope
    /// shape. Clients ignore unknown keys.
    pub extra: Map<String, Value>,
}

impl OpenAiError {
    /// Construct an envelope with an explicit code. For a `null` code use
    /// [`OpenAiError::without_code`].
    pub fn new(
        status: u16,
        error_type: impl Into<String>,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            error_type: error_type.into(),
            code: Some(code.into()),
            message: message.into(),
            param: None,
            retry_after_secs: None,
            extra: Map::new(),
        }
    }

    /// Construct an envelope whose `code` is `null` (e.g. an unclassified
    /// internal error).
    pub fn without_code(
        status: u16,
        error_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            status,
            error_type: error_type.into(),
            code: None,
            message: message.into(),
            param: None,
            retry_after_secs: None,
            extra: Map::new(),
        }
    }

    /// Advertise a `Retry-After` (seconds). Use on retryable rejections.
    pub fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after_secs = Some(secs);
        self
    }

    /// Set the OpenAI `param` field.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// Merge one diagnostic field into the error object.
    pub fn with_extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }

    /// Merge a bag of diagnostic fields into the error object.
    pub fn with_extras(mut self, extras: Map<String, Value>) -> Self {
        for (k, v) in extras {
            self.extra.insert(k, v);
        }
        self
    }

    /// Render the `{ "error": { … } }` body. Field order is irrelevant to
    /// clients (they parse JSON); the standard keys come first, then any
    /// diagnostic extras.
    pub fn body(&self) -> Value {
        let mut error = Map::new();
        error.insert("message".into(), Value::String(self.message.clone()));
        error.insert("type".into(), Value::String(self.error_type.clone()));
        error.insert(
            "code".into(),
            self.code.clone().map(Value::String).unwrap_or(Value::Null),
        );
        error.insert(
            "param".into(),
            self.param.clone().map(Value::String).unwrap_or(Value::Null),
        );
        for (k, v) in &self.extra {
            error.insert(k.clone(), v.clone());
        }
        json!({ "error": Value::Object(error) })
    }

    // ── Named constructors for the #63 standard codes ──────────────────

    /// `401 invalid_api_key` — missing/invalid bearer token (#49). Permanent.
    pub fn invalid_api_key(message: impl Into<String>) -> Self {
        Self::new(401, "invalid_request_error", "invalid_api_key", message)
    }

    /// `429 rate_limit_exceeded` + `Retry-After` — transient overload,
    /// fair-share/in-flight cap, admission rejection, or a rolling budget
    /// window that resets (#52/#53/#54/#55). Clients back off and retry.
    pub fn rate_limit_exceeded(message: impl Into<String>, retry_after_secs: u64) -> Self {
        Self::new(429, "rate_limit_error", "rate_limit_exceeded", message)
            .with_retry_after(retry_after_secs)
    }

    /// `429 insufficient_quota` — hard balance exhausted, no reset (#52).
    /// No `Retry-After`; the client surfaces and stops. (Never `402`.)
    pub fn insufficient_quota(message: impl Into<String>) -> Self {
        Self::new(429, "insufficient_quota", "insufficient_quota", message)
    }

    /// `400 context_length_exceeded` — prompt exceeds the model's context
    /// window (#56/#60). Permanent for this request; opencode auto-compacts.
    pub fn context_length_exceeded(message: impl Into<String>) -> Self {
        Self::new(
            400,
            "invalid_request_error",
            "context_length_exceeded",
            message,
        )
    }

    /// `503 service_unavailable` + optional `Retry-After` — transient
    /// backend unavailability (no healthy nodes, recovery, fail-closed
    /// upstream). Retryable when a hint is given.
    pub fn service_unavailable(message: impl Into<String>, retry_after_secs: Option<u64>) -> Self {
        let mut err = Self::new(503, "api_error", "service_unavailable", message);
        err.retry_after_secs = retry_after_secs;
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_has_standard_envelope_shape() {
        let env = OpenAiError::new(429, "rate_limit_error", "rate_limit_exceeded", "slow down");
        let body = env.body();
        let error = body.get("error").and_then(Value::as_object).unwrap();
        assert_eq!(error["message"], "slow down");
        assert_eq!(error["type"], "rate_limit_error");
        assert_eq!(error["code"], "rate_limit_exceeded");
        assert_eq!(error["param"], Value::Null);
    }

    #[test]
    fn without_code_renders_null_code() {
        let env = OpenAiError::without_code(500, "api_error", "kaboom");
        assert_eq!(env.body()["error"]["code"], Value::Null);
    }

    #[test]
    fn extras_ride_inside_the_error_object() {
        let env = OpenAiError::context_length_exceeded("too long")
            .with_extra("prompt_len", json!(60_000))
            .with_extra("max", json!(49_152));
        let error = &env.body()["error"];
        assert_eq!(error["prompt_len"], 60_000);
        assert_eq!(error["max"], 49_152);
        assert_eq!(error["code"], "context_length_exceeded");
    }

    #[test]
    fn rolling_window_rejection_carries_retry_after() {
        let env = OpenAiError::rate_limit_exceeded("budget window", 30);
        assert_eq!(env.status, 429);
        assert_eq!(env.retry_after_secs, Some(30));
    }

    #[test]
    fn hard_balance_rejection_has_no_retry_after() {
        let env = OpenAiError::insufficient_quota("out of credit");
        assert_eq!(env.status, 429);
        assert_eq!(env.code.as_deref(), Some("insufficient_quota"));
        assert_eq!(env.retry_after_secs, None);
    }

    #[test]
    fn permanent_rejections_have_no_retry_after() {
        assert_eq!(OpenAiError::invalid_api_key("nope").retry_after_secs, None);
        assert_eq!(
            OpenAiError::context_length_exceeded("too long").retry_after_secs,
            None
        );
    }

    #[test]
    fn service_unavailable_retry_after_is_optional() {
        assert_eq!(
            OpenAiError::service_unavailable("recovering", Some(5)).retry_after_secs,
            Some(5)
        );
        assert_eq!(
            OpenAiError::service_unavailable("gone", None).retry_after_secs,
            None
        );
    }
}
