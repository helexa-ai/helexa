//! Translation between OpenAI and Anthropic request/response envelopes.
//!
//! This is a stateless transformation — no context is carried between requests.

use crate::anthropic::{
    AnthropicContent, AnthropicMessage, AnthropicUsage, ContentBlock, MessagesRequest,
    MessagesResponse, SystemPrompt,
};
use crate::openai::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Usage,
    MessageContent,
};
use serde_json::{json, Value};

/// Convert an Anthropic Messages request into an OpenAI ChatCompletion request.
pub fn anthropic_to_openai(req: MessagesRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // Anthropic `system` field becomes a system message.
    if let Some(system) = req.system {
        let content = match system {
            SystemPrompt::Text(t) => t,
            SystemPrompt::Blocks(blocks) => serde_json::to_string(&blocks).unwrap_or_default(),
        };
        messages.push(ChatMessage {
            role: "system".into(),
            content: MessageContent::Text(content),
            extra: Value::Null,
        });
    }

    // Convert message roles and content.
    for msg in req.messages {
        let content = match msg.content {
            AnthropicContent::Text(t) => MessageContent::Text(t),
            AnthropicContent::Blocks(blocks) => {
                // For simple text-only blocks, extract the text.
                // For mixed content (images, etc.), pass as parts.
                if blocks.len() == 1 && blocks[0].block_type == "text" {
                    let text = blocks[0]
                        .data
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    MessageContent::Text(text)
                } else {
                    MessageContent::Parts(
                        blocks.into_iter().map(|b| json!(b)).collect(),
                    )
                }
            }
        };
        messages.push(ChatMessage {
            role: msg.role,
            content,
            extra: Value::Null,
        });
    }

    ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: Some(req.max_tokens),
        stream: req.stream,
        extra: req.extra,
    }
}

/// Convert an OpenAI ChatCompletion response into an Anthropic Messages response.
pub fn openai_to_anthropic(resp: ChatCompletionResponse) -> MessagesResponse {
    let choice = resp.choices.into_iter().next();

    let (content_text, stop_reason) = match choice {
        Some(c) => {
            let text = match c.message.content {
                MessageContent::Text(t) => t,
                MessageContent::Parts(parts) => serde_json::to_string(&parts).unwrap_or_default(),
            };
            let stop = c.finish_reason.map(|r| match r.as_str() {
                "stop" => "end_turn".to_string(),
                "length" => "max_tokens".to_string(),
                other => other.to_string(),
            });
            (text, stop)
        }
        None => (String::new(), None),
    };

    let usage = resp.usage.unwrap_or(Usage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    MessagesResponse {
        id: resp.id,
        response_type: "message".into(),
        role: "assistant".into(),
        content: vec![ContentBlock {
            block_type: "text".into(),
            data: json!({ "text": content_text }),
        }],
        model: resp.model,
        stop_reason,
        usage: AnthropicUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
        },
        extra: Value::Null,
    }
}
