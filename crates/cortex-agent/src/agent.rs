//! Per-node agent sidecar.
//!
//! This is a future component that runs on each GPU node alongside mistralrs.
//! It handles:
//!   - VRAM defragmentation (restarting the mistralrs systemd unit when the
//!     gateway signals that lifecycle_cycles has exceeded the threshold)
//!   - Local nvidia-smi polling for actual VRAM usage reporting
//!   - Systemd unit management for mistralrs process restarts
//!
//! For now this is a stub. The gateway's poller + evictor handle the critical
//! path (model lifecycle via the mistralrs HTTP API). The agent adds
//! operational niceties that can be built incrementally.

/// Placeholder for agent configuration.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// The local mistralrs endpoint to monitor.
    pub mistralrs_endpoint: String,
    /// The systemd unit name for mistralrs (e.g. "mistralrs.service").
    pub systemd_unit: String,
}

/// Restart the local mistralrs process via systemd.
/// This is the nuclear option for VRAM defragmentation.
pub async fn restart_mistralrs(config: &AgentConfig) -> anyhow::Result<()> {
    tracing::warn!(
        unit = %config.systemd_unit,
        "restarting mistralrs for VRAM defragmentation"
    );

    let output = tokio::process::Command::new("systemctl")
        .args(["restart", &config.systemd_unit])
        .output()
        .await?;

    if output.status.success() {
        tracing::info!(unit = %config.systemd_unit, "mistralrs restarted successfully");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("systemctl restart failed: {stderr}");
    }
}

/// Query nvidia-smi for current VRAM usage on this node.
/// Returns (used_mb, total_mb) for each GPU.
pub async fn query_vram() -> anyhow::Result<Vec<(u64, u64)>> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("nvidia-smi failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() == 2 {
            let used: u64 = parts[0].parse().unwrap_or(0);
            let total: u64 = parts[1].parse().unwrap_or(0);
            gpus.push((used, total));
        }
    }
    Ok(gpus)
}
