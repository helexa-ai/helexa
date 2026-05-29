//! helexa-acp — Agent Client Protocol bridge for multi-endpoint LLM
//! setups (helexa, LM Studio, Ollama, OpenRouter, OpenAI, Anthropic,
//! …) with a clean per-endpoint wire-format selector.
//!
//! Speaks ACP over stdio to an editor client (Zed today). Every
//! configured endpoint produces a wire-format-specific
//! [`provider::Provider`] implementation; the agent loop in
//! [`agent::Agent`] is provider-agnostic, so adding e.g. an Anthropic
//! /v1/messages provider doesn't touch `agent.rs`.
//!
//! Config: `$XDG_CONFIG_HOME/helexa-acp/config.toml` for the multi-
//! endpoint case; env vars (`HELEXA_ACP_BASE_URL`, etc.) for the
//! single-endpoint case when no config file exists.

use agent_client_protocol::{Result, Stdio};
use std::sync::Arc;

mod agent;
mod compaction;
mod config;
mod prompt;
mod provider;
mod qwen3;
mod session;
mod store;
mod tool_runner;
mod tools;

use agent::Agent;
use config::{Config, EndpointConfig, WireApi};
use provider::{Provider, openai_chat::OpenAIChatProvider};

/// Set up tracing. Logs go to stderr by default — stdout is
/// reserved for the JSON-RPC stream. Setting `HELEXA_ACP_LOG_FILE`
/// to an absolute path appends logs to that file instead, which is
/// the practical way to capture debug output when the agent runs
/// under an editor (Zed, etc.) that doesn't surface stderr.
///
/// `RUST_LOG` still controls levels (e.g. `helexa_acp=debug`).
/// ANSI colours are auto-stripped when writing to a file so the log
/// is plain text.
fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    let log_file = std::env::var("HELEXA_ACP_LOG_FILE")
        .ok()
        .filter(|s| !s.is_empty());

    match log_file {
        Some(path) => match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => {
                tracing_subscriber::fmt()
                    .with_writer(std::sync::Mutex::new(file))
                    .with_env_filter(env_filter)
                    .with_ansi(false)
                    .init();
            }
            Err(e) => {
                // Fall back to stderr and shout. We don't want a
                // typo'd log path to silence the agent entirely.
                tracing_subscriber::fmt()
                    .with_writer(std::io::stderr)
                    .with_env_filter(env_filter)
                    .init();
                tracing::warn!(
                    path = %path,
                    error = %e,
                    "HELEXA_ACP_LOG_FILE could not be opened; using stderr"
                );
            }
        },
        None => {
            tracing_subscriber::fmt()
                .with_writer(std::io::stderr)
                .with_env_filter(env_filter)
                .init();
        }
    }
}

/// Build a provider for `endpoint` according to its declared
/// `wire_api`. Future wire types (OpenAI Responses, Anthropic
/// /v1/messages, Ollama native) slot in here without changing the
/// caller.
fn build_provider(endpoint: EndpointConfig) -> anyhow::Result<Arc<dyn Provider>> {
    match endpoint.wire_api {
        WireApi::OpenAiChat => Ok(Arc::new(OpenAIChatProvider::new(endpoint)?)),
        WireApi::OpenAiResponses => Err(anyhow::anyhow!(
            "endpoint '{}' wire_api 'openai-responses' is reserved for a future provider; \
             use 'openai-chat' for now or wait for the OpenAIResponsesProvider impl",
            endpoint.name
        )),
        WireApi::AnthropicMessages => Err(anyhow::anyhow!(
            "endpoint '{}' wire_api 'anthropic-messages' is reserved for a future provider",
            endpoint.name
        )),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let cfg = Config::load()
        .map_err(|e| agent_client_protocol::util::internal_error(format!("config: {e:#}")))?;
    tracing::info!(
        endpoints = cfg.endpoints.len(),
        default_endpoint = %cfg.default_endpoint().name,
        default_model = ?cfg.default_endpoint().default_model,
        "helexa-acp starting"
    );

    // Build a provider for each configured endpoint up-front. Cheap —
    // just sets up a reqwest::Client and resolves the API key — and
    // surfaces config mistakes (missing API key env var, unsupported
    // wire_api) before the editor even sends an initialize request.
    let mut providers: Vec<Arc<dyn Provider>> = Vec::with_capacity(cfg.endpoints.len());
    for endpoint in &cfg.endpoints {
        match build_provider(endpoint.clone()) {
            Ok(p) => {
                tracing::info!(
                    endpoint = %endpoint.name,
                    base_url = %endpoint.base_url,
                    wire_api = ?endpoint.wire_api,
                    "registered provider"
                );
                providers.push(p);
            }
            Err(e) => {
                tracing::warn!(
                    endpoint = %endpoint.name,
                    error = %format!("{e:#}"),
                    "skipping endpoint with invalid config"
                );
            }
        }
    }

    let agent = Agent::new(&cfg, providers)
        .await
        .map_err(|e| agent_client_protocol::util::internal_error(format!("agent: {e:#}")))?;
    agent.serve(Stdio::new()).await
}
