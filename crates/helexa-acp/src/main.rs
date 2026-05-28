//! helexa-acp — Agent Client Protocol bridge for multi-endpoint LLM
//! setups (helexa, LM Studio, Ollama, OpenRouter, OpenAI, Anthropic,
//! …) with a clean per-endpoint wire-format selector.
//!
//! Speaks ACP over stdio to an editor client (Zed today). The
//! conversation is forwarded to one of the configured endpoints via
//! a wire-format-specific [`provider::Provider`] implementation.
//! The agent loop itself is provider-agnostic — adding e.g. an
//! Anthropic /v1/messages provider doesn't touch `agent.rs`.
//!
//! Config: `$XDG_CONFIG_HOME/helexa-acp/config.toml` for the multi-
//! endpoint case; env vars (`HELEXA_ACP_BASE_URL`, etc.) for the
//! single-endpoint case when no config file exists.

use agent_client_protocol::schema::{AgentCapabilities, InitializeRequest, InitializeResponse};
use agent_client_protocol::{Agent, Client, ConnectionTo, Dispatch, Result, Stdio};
use std::sync::Arc;

mod config;
mod provider;

use config::{Config, EndpointConfig, WireApi};
use provider::{Provider, openai_chat::OpenAIChatProvider};

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
    // Logs go to stderr — stdout is reserved for the JSON-RPC stream.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

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
    if providers.is_empty() {
        return Err(agent_client_protocol::util::internal_error(
            "no usable endpoints — check config",
        ));
    }

    Agent
        .builder()
        .name("helexa-acp")
        .on_receive_request(
            async move |initialize: InitializeRequest, responder, _connection| {
                // Phase 1 wiring — capabilities only. Real session
                // handling lands in the next iteration (agent.rs).
                responder.respond(
                    InitializeResponse::new(initialize.protocol_version)
                        .agent_capabilities(AgentCapabilities::new()),
                )
            },
            agent_client_protocol::on_receive_request!(),
        )
        .on_receive_dispatch(
            async move |message: Dispatch, cx: ConnectionTo<Client>| {
                tracing::warn!(method = ?message.method(), "unhandled ACP message");
                message.respond_with_error(
                    agent_client_protocol::util::internal_error("not implemented yet"),
                    cx,
                )
            },
            agent_client_protocol::on_receive_dispatch!(),
        )
        .connect_to(Stdio::new())
        .await
}
