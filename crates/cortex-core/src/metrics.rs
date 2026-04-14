//! Request-level metrics captured by the gateway proxy layer.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Metrics captured for a single proxied request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub node: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// Tokens per second for the generation phase.
    pub tok_per_sec: f64,
    /// Time from request start to first SSE chunk (streaming) or full response.
    pub time_to_first_token_ms: u64,
    /// Total request latency including proxy overhead.
    pub total_latency_ms: u64,
    /// Whether this request triggered a model load (cold start).
    pub cold_start: bool,
}
