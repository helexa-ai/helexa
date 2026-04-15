//! Hardware discovery and health types shared between cortex and neuron.

use serde::{Deserialize, Serialize};

/// Information about a single GPU device discovered on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub vram_total_mb: u64,
    pub compute_capability: String,
}

/// Full discovery response from a neuron endpoint.
/// Returned by `GET /discovery`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResponse {
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub cuda_version: Option<String>,
    pub driver_version: Option<String>,
    pub devices: Vec<DeviceInfo>,
    pub harnesses: Vec<String>,
}

/// Runtime health metrics for a single GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceHealth {
    pub index: u32,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub utilization_pct: u32,
    pub temp_c: u32,
}

/// Runtime health response from a neuron endpoint.
/// Returned by `GET /health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub uptime_secs: u64,
    pub devices: Vec<DeviceHealth>,
}
