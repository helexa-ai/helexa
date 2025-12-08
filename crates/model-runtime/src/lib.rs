// SPDX-License-Identifier: PolyForm-Shield-1.0

use anyhow::Result;
use async_trait::async_trait;
use std::time::Duration;

/// Basic chat request type understood by runtime adapters.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Basic chat response type.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
}

/// Trait for chat-capable runtimes.
#[async_trait]
pub trait ChatInference: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;
}

/// HTTP-backed runtime that talks to an OpenAI-compatible
/// `/v1/chat/completions` endpoint.
///
/// This is a placeholder implementation intended to be wired
/// up by neuron, which will construct instances pointing at
/// backend processes (e.g. vLLM or llama.cpp) configured to
/// expose an OpenAI-style HTTP API.
///
/// The concrete HTTP behaviour is intentionally left
/// unimplemented so that incomplete behaviour is loud and
/// obvious during development.
pub struct ProcessRuntime {
    /// Base URL of the OpenAI-compatible endpoint, e.g.:
    /// `http://127.0.0.1:8000`.
    pub base_url: String,
    /// Timeout for HTTP requests to the backend.
    pub timeout: Duration,
    /// Optional model name override to send to the backend.
    pub model: Option<String>,
}

impl ProcessRuntime {
    /// Construct a new HTTP-backed process runtime.
    ///
    /// `base_url` should not include the path; the implementation
    /// will append `/v1/chat/completions` when sending requests.
    pub fn new<S: Into<String>>(base_url: S, timeout: Duration, model: Option<String>) -> Self {
        Self {
            base_url: base_url.into(),
            timeout,
            model,
        }
    }
}

#[async_trait]
impl ChatInference for ProcessRuntime {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse> {
        // TODO: implement HTTP client integration that:
        // - builds an OpenAI-style /v1/chat/completions request body
        //   from `ChatRequest`
        // - sends it to `{base_url}/v1/chat/completions`
        // - parses the response and maps it to `ChatResponse`
        //
        // This is intentionally left as an unimplemented placeholder
        // so that no one can accidentally rely on an incomplete or
        // silently stubbed implementation.
        unimplemented!("ProcessRuntime::chat is not implemented yet");
    }
}

/// Opaque handle to something that can do chat inference.
#[derive(Clone)]
pub struct ChatRuntimeHandle {
    inner: std::sync::Arc<dyn ChatInference>,
}

impl ChatRuntimeHandle {
    pub fn new(inner: std::sync::Arc<dyn ChatInference>) -> Self {
        Self { inner }
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.inner.chat(request).await
    }
}
