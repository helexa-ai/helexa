// SPDX-License-Identifier: PolyForm-Shield-1.0

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
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
/// This implementation is intentionally minimal and does not
/// attempt to cover every OpenAI feature; it is sufficient for
/// basic non-streaming chat completions driven by `ChatRequest`.
pub struct ProcessRuntime {
    /// Base URL of the OpenAI-compatible endpoint, e.g.:
    /// `http://127.0.0.1:8000`.
    pub base_url: String,
    /// Timeout for HTTP requests to the backend.
    pub timeout: Duration,
    /// Optional model name override to send to the backend.
    pub model: Option<String>,
    /// Underlying HTTP client.
    client: Client,
}

impl ProcessRuntime {
    /// Construct a new HTTP-backed process runtime.
    ///
    /// `base_url` should not include the path; the implementation
    /// will append `/v1/chat/completions` when sending requests.
    pub fn new<S: Into<String>>(base_url: S, timeout: Duration, model: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to construct HTTP client for ProcessRuntime");

        Self {
            base_url: base_url.into(),
            timeout,
            model,
            client,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoiceMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiChatResponseBody {
    choices: Vec<OpenAiChoice>,
}

#[async_trait]
impl ChatInference for ProcessRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        // Map internal ChatRequest into a minimal OpenAI-style request body.
        let messages = request
            .messages
            .iter()
            .map(|m| OpenAiChatMessage {
                role: match m.role {
                    ChatRole::System => "system".to_string(),
                    ChatRole::User => "user".to_string(),
                    ChatRole::Assistant => "assistant".to_string(),
                },
                content: m.content.clone(),
            })
            .collect::<Vec<_>>();

        let model = self
            .model
            .clone()
            .ok_or_else(|| anyhow!("ProcessRuntime requires a model name to call the backend"))?;

        let body = OpenAiChatRequest {
            model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: Some(false),
        };

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("HTTP request to backend failed: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(anyhow!(
                "backend returned error status {}: {}",
                status,
                text
            ));
        }

        let parsed: OpenAiChatResponseBody = resp
            .json()
            .await
            .map_err(|e| anyhow!("failed to parse backend response as JSON: {e}"))?;

        let content = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("backend response contained no choices"))?
            .message
            .content;

        Ok(ChatResponse { content })
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
