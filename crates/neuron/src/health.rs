//! Cached GPU health monitoring via periodic nvidia-smi polling.

use cortex_core::discovery::HealthResponse;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Thread-safe cache for the latest GPU health reading.
pub struct HealthCache {
    inner: RwLock<HealthResponse>,
    has_gpus: RwLock<bool>,
    /// This host's power schedule (#397); unset for an always-on host.
    schedule: std::sync::OnceLock<crate::availability::Schedule>,
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
                // Overlaid by the api handler from `[availability]`.
                next_dark: None,
            }),
            has_gpus: RwLock::new(false),
            schedule: std::sync::OnceLock::new(),
        }
    }

    /// Mark whether this node has GPUs (set after discovery).
    pub async fn set_has_gpus(&self, has_gpus: bool) {
        *self.has_gpus.write().await = has_gpus;
    }

    /// Get a snapshot of the current health state.
    pub async fn snapshot(&self) -> HealthResponse {
        let mut snapshot = self.inner.read().await.clone();
        if let Some(schedule) = self.schedule.get() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            snapshot.next_dark = schedule.next_dark(now);
        }
        snapshot
    }

    /// Announce this host's power schedule on `/health` (#397). Set once,
    /// at startup.
    pub fn set_schedule(&self, schedule: crate::availability::Schedule) {
        let _ = self.schedule.set(schedule);
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
