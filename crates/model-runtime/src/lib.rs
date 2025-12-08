use anyhow::Result;
use async_trait::async_trait;

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
