//! GPU discovery via nvidia-smi and system info gathering.
//!
//! Pure parsing functions are separated from command execution for testability.

use anyhow::{Context, Result};
use cortex_core::discovery::{DeviceHealth, DeviceInfo, DiscoveryResponse};

const NVIDIA_SMI_DISCOVERY_QUERY: &str = "index,name,memory.total,compute_cap,driver_version";
const NVIDIA_SMI_HEALTH_QUERY: &str =
    "index,memory.used,memory.free,utilization.gpu,temperature.gpu";

// ── Pure parsing functions (testable without GPU) ───────────────────

/// Parse nvidia-smi CSV output for device discovery.
///
/// Expected input format (one line per GPU):
/// ```text
/// 0, NVIDIA GeForce RTX 5090, 32614, 12.0, 570.86.16
/// 1, NVIDIA GeForce RTX 5090, 32614, 12.0, 570.86.16
/// ```
pub fn parse_gpu_info(csv_output: &str) -> Result<Vec<DeviceInfo>> {
    let mut devices = Vec::new();
    for line in csv_output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(5, ',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            anyhow::bail!("malformed nvidia-smi line (expected 5 fields): {line}");
        }
        devices.push(DeviceInfo {
            index: parts[0]
                .parse()
                .with_context(|| format!("invalid GPU index: {}", parts[0]))?,
            name: parts[1].to_string(),
            vram_total_mb: parts[2]
                .parse()
                .with_context(|| format!("invalid VRAM: {}", parts[2]))?,
            compute_capability: parts[3].to_string(),
        });
    }
    Ok(devices)
}

/// Extract the driver version from nvidia-smi discovery output.
/// Takes the driver_version field from the first GPU line.
pub fn parse_driver_version(csv_output: &str) -> Option<String> {
    let line = csv_output.lines().find(|l| !l.trim().is_empty())?;
    let parts: Vec<&str> = line.splitn(5, ',').map(|s| s.trim()).collect();
    if parts.len() >= 5 {
        Some(parts[4].to_string())
    } else {
        None
    }
}

/// Parse the CUDA version from `nvcc --version` output.
///
/// Expected line: `Cuda compilation tools, release 12.8, V12.8.93`
pub fn parse_cuda_version(nvcc_output: &str) -> Option<String> {
    for line in nvcc_output.lines() {
        if line.contains("release") {
            // Extract "12.8" from "release 12.8,"
            let after_release = line.split("release").nth(1)?;
            let version = after_release.trim().split(',').next()?.trim();
            if !version.is_empty() {
                return Some(version.to_string());
            }
        }
    }
    None
}

/// Parse nvidia-smi CSV output for health metrics.
///
/// Expected input format (one line per GPU):
/// ```text
/// 0, 8192, 24372, 45, 62
/// ```
pub fn parse_health_info(csv_output: &str) -> Result<Vec<DeviceHealth>> {
    let mut devices = Vec::new();
    for line in csv_output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.splitn(5, ',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            anyhow::bail!("malformed nvidia-smi health line (expected 5 fields): {line}");
        }
        devices.push(DeviceHealth {
            index: parts[0].parse().with_context(|| "invalid index")?,
            vram_used_mb: parts[1].parse().with_context(|| "invalid vram_used")?,
            vram_free_mb: parts[2].parse().with_context(|| "invalid vram_free")?,
            utilization_pct: parts[3].parse().with_context(|| "invalid utilization")?,
            temp_c: parts[4].parse().with_context(|| "invalid temp")?,
        });
    }
    Ok(devices)
}

// ── Command execution wrappers ──────────────────────────────────────

async fn run_command(cmd: &str, args: &[&str]) -> Result<String> {
    let output = tokio::process::Command::new(cmd)
        .args(args)
        .output()
        .await
        .with_context(|| format!("failed to execute {cmd}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("{cmd} failed: {stderr}");
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

async fn run_command_optional(cmd: &str, args: &[&str]) -> Option<String> {
    run_command(cmd, args).await.ok()
}

/// Discover the full system: hostname, OS, kernel, GPUs, CUDA version.
/// Handles nvidia-smi not found gracefully (returns empty devices).
pub async fn discover_system() -> Result<DiscoveryResponse> {
    let hostname = run_command("uname", &["-n"])
        .await
        .unwrap_or_else(|_| "unknown".into())
        .trim()
        .to_string();
    let os = run_command("uname", &["-s"])
        .await
        .unwrap_or_else(|_| "unknown".into())
        .trim()
        .to_string();
    let kernel = run_command("uname", &["-r"])
        .await
        .unwrap_or_else(|_| "unknown".into())
        .trim()
        .to_string();

    let (devices, driver_version) = match run_command_optional(
        "nvidia-smi",
        &[
            &format!("--query-gpu={NVIDIA_SMI_DISCOVERY_QUERY}"),
            "--format=csv,noheader,nounits",
        ],
    )
    .await
    {
        Some(output) => {
            let devs = parse_gpu_info(&output).unwrap_or_default();
            let driver = parse_driver_version(&output);
            (devs, driver)
        }
        None => {
            tracing::info!("nvidia-smi not found — no GPU devices discovered");
            (vec![], None)
        }
    };

    let cuda_version = match run_command_optional("nvcc", &["--version"]).await {
        Some(output) => parse_cuda_version(&output),
        None => None,
    };

    Ok(DiscoveryResponse {
        hostname,
        os,
        kernel,
        cuda_version,
        driver_version,
        devices,
        harnesses: vec![], // populated by harness registry in Phase 8
    })
}

/// Run nvidia-smi health query and parse the output.
pub async fn query_health() -> Result<Vec<DeviceHealth>> {
    let output = run_command(
        "nvidia-smi",
        &[
            &format!("--query-gpu={NVIDIA_SMI_HEALTH_QUERY}"),
            "--format=csv,noheader,nounits",
        ],
    )
    .await?;
    parse_health_info(&output)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_info_single_gpu() {
        let csv = "0, NVIDIA GeForce RTX 4090, 24564, 8.9, 570.86.16\n";
        let devices = parse_gpu_info(csv).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devices[0].vram_total_mb, 24564);
        assert_eq!(devices[0].compute_capability, "8.9");
    }

    #[test]
    fn test_parse_gpu_info_multi_gpu() {
        let csv = "\
            0, NVIDIA GeForce RTX 5090, 32614, 12.0, 570.86.16\n\
            1, NVIDIA GeForce RTX 5090, 32614, 12.0, 570.86.16\n";
        let devices = parse_gpu_info(csv).unwrap();
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].index, 0);
        assert_eq!(devices[1].index, 1);
        assert_eq!(devices[0].vram_total_mb, 32614);
    }

    #[test]
    fn test_parse_gpu_info_empty() {
        let devices = parse_gpu_info("").unwrap();
        assert!(devices.is_empty());
    }

    #[test]
    fn test_parse_gpu_info_malformed() {
        let result = parse_gpu_info("garbage data");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_driver_version() {
        let csv = "0, NVIDIA GeForce RTX 4090, 24564, 8.9, 570.86.16\n";
        assert_eq!(parse_driver_version(csv), Some("570.86.16".to_string()));
    }

    #[test]
    fn test_parse_cuda_version() {
        let nvcc = "\
            nvcc: NVIDIA (R) Cuda compiler driver\n\
            Copyright (c) 2005-2024 NVIDIA Corporation\n\
            Built on Thu_Sep_12_02:18:05_PDT_2024\n\
            Cuda compilation tools, release 12.8, V12.8.93\n";
        assert_eq!(parse_cuda_version(nvcc), Some("12.8".to_string()));
    }

    #[test]
    fn test_parse_cuda_version_missing() {
        assert_eq!(parse_cuda_version("unrelated output"), None);
    }

    #[test]
    fn test_parse_health_info() {
        let csv = "0, 8192, 16372, 45, 62\n";
        let health = parse_health_info(csv).unwrap();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].index, 0);
        assert_eq!(health[0].vram_used_mb, 8192);
        assert_eq!(health[0].vram_free_mb, 16372);
        assert_eq!(health[0].utilization_pct, 45);
        assert_eq!(health[0].temp_c, 62);
    }

    #[test]
    fn test_parse_health_info_multi_gpu() {
        let csv = "\
            0, 8192, 24372, 45, 62\n\
            1, 4096, 28468, 30, 58\n";
        let health = parse_health_info(csv).unwrap();
        assert_eq!(health.len(), 2);
        assert_eq!(health[1].vram_used_mb, 4096);
        assert_eq!(health[1].temp_c, 58);
    }
}
