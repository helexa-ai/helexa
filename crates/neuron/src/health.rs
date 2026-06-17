//! Cached GPU health monitoring via periodic nvidia-smi polling.

use cortex_core::discovery::HealthResponse;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Thread-safe cache for the latest GPU health reading.
pub struct HealthCache {
    inner: RwLock<HealthResponse>,
    has_gpus: RwLock<bool>,
}

impl Default for HealthCache {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthCache {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HealthResponse {
                uptime_secs: 0,
                devices: vec![],
                // The cache only owns the device-state half of /health;
                // the api handler overlays activation from the tracker.
                // Initialise with the default (Ready, empty lists) so a
                // direct read from the cache stays a well-typed
                // HealthResponse on the wire.
                activation: Default::default(),
                // Per-model admission load is overlaid by the api handler
                // from the candle harness (#53); the cache doesn't own it.
                models: Vec::new(),
            }),
            has_gpus: RwLock::new(false),
        }
    }

    /// Mark whether this node has GPUs (set after discovery).
    pub async fn set_has_gpus(&self, has_gpus: bool) {
        *self.has_gpus.write().await = has_gpus;
    }

    /// Get a snapshot of the current health state.
    pub async fn snapshot(&self) -> HealthResponse {
        self.inner.read().await.clone()
    }

    /// Run forever, polling nvidia-smi every 5 seconds and updating the cache.
    pub async fn poll_loop(&self, start_time: Instant) {
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;

            let uptime = start_time.elapsed().as_secs();

            if !*self.has_gpus.read().await {
                let mut health = self.inner.write().await;
                health.uptime_secs = uptime;
                continue;
            }

            match crate::discovery::query_health().await {
                Ok(devices) => {
                    let mut health = self.inner.write().await;
                    health.uptime_secs = uptime;
                    health.devices = devices;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "failed to poll GPU health");
                    // Keep last known reading, just update uptime.
                    let mut health = self.inner.write().await;
                    health.uptime_secs = uptime;
                }
            }
        }
    }
}
