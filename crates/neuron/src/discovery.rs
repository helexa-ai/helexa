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

// ── Driver/library mismatch preflight (#19) ─────────────────────────

/// Classify a failed nvidia-smi invocation: is it the classic
/// "Driver/library version mismatch" (userspace libs updated, kernel
/// module not reloaded — every CUDA call on the host is dead until a
/// reboot)? Returns the userspace NVML library version when the
/// message carries one ("NVML library version: 580.159"), or
/// `Some("unknown")` for a mismatch without a parsable version.
/// `None` for any other failure — other errors (no devices, perms)
/// are NOT the mismatch and must not trigger the loud diagnosis.
pub fn classify_driver_mismatch(combined_output: &str) -> Option<String> {
    if !combined_output.contains("Driver/library version mismatch") {
        return None;
    }
    let userspace = combined_output
        .lines()
        .find_map(|l| l.trim().strip_prefix("NVML library version:"))
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    Some(userspace)
}

/// Extract the loaded kernel module's driver version from
/// `/proc/driver/nvidia/version` contents. Typical first line:
///
/// ```text
/// NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.159.03  Release Build  (...)
/// ```
pub fn parse_kernel_module_version(proc_contents: &str) -> Option<String> {
    let is_numeric = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit());
    let line = proc_contents
        .lines()
        .find(|l| l.starts_with("NVRM version:"))?;
    line.split_whitespace()
        .find(|tok| {
            let mut parts = tok.split('.');
            parts.next().is_some_and(is_numeric) && parts.next().is_some_and(is_numeric)
        })
        .map(|s| s.to_string())
}

/// Render the operator-actionable mismatch description carried in
/// `DiscoveryResponse::cuda_unavailable_reason` and logged at startup.
pub fn mismatch_reason(userspace: &str, kernel_module: Option<&str>) -> String {
    format!(
        "host NVIDIA driver/library mismatch (userspace NVML {userspace} vs loaded kernel \
         module {}) — reboot the host to reload the kernel module; all CUDA inference is \
         unavailable until then",
        kernel_module.unwrap_or("unknown")
    )
}

/// Outcome of an nvidia-smi invocation, distinguishing "binary not
/// present" (CPU-only host, not an error) from "present but failing"
/// (possible driver mismatch — worth classifying).
enum SmiOutcome {
    Ok(String),
    Failed(String),
    Absent,
}

async fn run_nvidia_smi(args: &[&str]) -> SmiOutcome {
    match tokio::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .await
    {
        Err(_) => SmiOutcome::Absent,
        Ok(out) if out.status.success() => {
            SmiOutcome::Ok(String::from_utf8_lossy(&out.stdout).to_string())
        }
        Ok(out) => {
            let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
            combined.push('\n');
            combined.push_str(&String::from_utf8_lossy(&out.stderr));
            SmiOutcome::Failed(combined)
        }
    }
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

    let (devices, driver_version, cuda_unavailable_reason) = match run_nvidia_smi(&[
        &format!("--query-gpu={NVIDIA_SMI_DISCOVERY_QUERY}"),
        "--format=csv,noheader,nounits",
    ])
    .await
    {
        SmiOutcome::Ok(output) => {
            let devs = parse_gpu_info(&output).unwrap_or_default();
            let driver = parse_driver_version(&output);
            (devs, driver, None)
        }
        SmiOutcome::Absent => {
            tracing::info!("nvidia-smi not found — no GPU devices discovered");
            (vec![], None, None)
        }
        SmiOutcome::Failed(combined) => {
            // nvidia-smi exists but can't talk to the driver. The case
            // worth diagnosing precisely is the userspace↔kernel-module
            // version skew after an un-rebooted driver update (#19) —
            // every CUDA call on the host fails until a reboot, and
            // without this classification it surfaces as a cryptic
            // NCCL/cuInit error deep inside the first model load.
            let reason = classify_driver_mismatch(&combined).map(|userspace| {
                let kmod = std::fs::read_to_string("/proc/driver/nvidia/version")
                    .ok()
                    .as_deref()
                    .and_then(parse_kernel_module_version);
                mismatch_reason(&userspace, kmod.as_deref())
            });
            if reason.is_none() {
                tracing::warn!(
                    output = %combined.trim(),
                    "nvidia-smi present but failing — no GPU devices discovered"
                );
            }
            (vec![], None, reason)
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
        cuda_unavailable_reason,
        max_prompt_tokens: crate::harness::candle::max_prompt_tokens() as u64,
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

    // ── #19 driver/library mismatch preflight ────────────────────────

    #[test]
    fn classify_driver_mismatch_detects_and_extracts_nvml_version() {
        // Verbatim shape of nvidia-smi's failure output on a host
        // whose userspace libs were updated without a reboot.
        let out = "Failed to initialize NVML: Driver/library version mismatch\n\
                   NVML library version: 580.159\n";
        assert_eq!(classify_driver_mismatch(out).as_deref(), Some("580.159"));
    }

    #[test]
    fn classify_driver_mismatch_without_version_line() {
        let out = "Failed to initialize NVML: Driver/library version mismatch\n";
        assert_eq!(classify_driver_mismatch(out).as_deref(), Some("unknown"));
    }

    #[test]
    fn classify_driver_mismatch_ignores_other_failures() {
        // Other nvidia-smi failures must NOT be diagnosed as the
        // mismatch (no false positives on healthy or odd hosts).
        for out in [
            "No devices were found\n",
            "Failed to initialize NVML: Insufficient Permissions\n",
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n",
            "",
        ] {
            assert_eq!(
                classify_driver_mismatch(out),
                None,
                "false positive on: {out:?}"
            );
        }
    }

    #[test]
    fn parse_kernel_module_version_from_proc() {
        let proc = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.159.03  Release Build  (dvs-builder@U22-I3-AE24-12-2)  Tue May 12 21:03:35 UTC 2026\n\
                    GCC version:  gcc version 15.2.1 20251022 (Red Hat 15.2.1-3) (GCC)\n";
        assert_eq!(
            parse_kernel_module_version(proc).as_deref(),
            Some("580.159.03")
        );
    }

    #[test]
    fn parse_kernel_module_version_absent() {
        assert_eq!(parse_kernel_module_version(""), None);
        assert_eq!(parse_kernel_module_version("GCC version: gcc 15\n"), None);
    }

    #[test]
    fn mismatch_reason_is_operator_actionable() {
        let reason = mismatch_reason("580.159", Some("580.159.03"));
        assert!(reason.contains("580.159"));
        assert!(reason.contains("580.159.03"));
        assert!(reason.contains("reboot"));
    }
}
