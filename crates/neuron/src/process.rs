// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::collections::HashMap;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use thiserror::Error;
use tracing::{info, warn};

/// Opaque identifier for a backend process managed by [`ProcessManager`].
///
/// In the neuron context, each worker process typically corresponds to a
/// single model backend (e.g. a vLLM or llama.cpp instance exposing an
/// OpenAI-compatible HTTP API). The handle records both the logical model
/// identifier and the OS process identifier so that workers can be grouped
/// and evicted by model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkerHandle {
    /// Opaque model identifier; this should usually match `protocol::ModelId`.
    pub model_id: String,
    /// OS process identifier for the backend worker.
    pub pid: u32,
}

/// Errors that can occur when managing backend processes.
#[derive(Debug, Error)]
pub enum ProcessError {
    /// Failure to spawn a new worker process.
    #[error("failed to spawn process: {0}")]
    Spawn(#[from] std::io::Error),
}

/// Simple process manager for neuron backend workers.
///
/// This module is intentionally minimal and process-focused:
///
/// - It knows how to spawn command-line workers (e.g. vLLM or llama.cpp).
/// - It tracks workers by PID and by model identifier.
/// - It exposes termination operations for individual workers and
///   all workers associated with a given model.
///
/// Higher-level concerns such as health checks, HTTP readiness probes,
/// and log streaming should be implemented in other modules using the
/// tracking information provided here.
#[derive(Debug, Default)]
pub struct ProcessManager {
    /// Map of worker PIDs to the corresponding child handles.
    workers: Mutex<HashMap<u32, Child>>,
    /// Map from model identifier to the set of worker PIDs serving it.
    ///
    /// This allows higher layers (e.g. control-plane directive handlers)
    /// to evict or restart all workers for a given model.
    by_model: Mutex<HashMap<String, Vec<u32>>>,
}

impl ProcessManager {
    /// Create a new, empty process manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn a new worker process with the given command and arguments.
    ///
    /// On success, the worker is tracked internally by PID and model so that
    /// it can be terminated or inspected later. The caller receives a
    /// [`WorkerHandle`] that can be used with other [`ProcessManager`] APIs.
    ///
    /// The `model_id` should match the protocol's notion of a model
    /// identifier, typically the opaque slug string.
    ///
    /// Stdout and stderr are configured as piped so that higher layers can
    /// attach readers and expose log streams (for example, over WebSockets)
    /// without having to re-spawn the process.
    pub fn spawn_worker(
        &self,
        cmd: &str,
        args: &[&str],
        model_id: &str,
    ) -> Result<WorkerHandle, ProcessError> {
        self.spawn_worker_with_env(cmd, args, model_id, &[])
    }

    /// Spawn a new worker process with the given command, args, and
    /// additional environment variables.
    ///
    /// This behaves like [`ProcessManager::spawn_worker`], but allows callers
    /// to specify extra environment entries (for example, to extend PATH or
    /// LD_LIBRARY_PATH for user-local binaries and libraries).
    ///
    /// The `extra_env` slice is applied on top of the inherited environment
    /// of the current process; entries with the same key override inherited
    /// values.
    pub fn spawn_worker_with_env(
        &self,
        cmd: &str,
        args: &[&str],
        model_id: &str,
        extra_env: &[(String, String)],
    ) -> Result<WorkerHandle, ProcessError> {
        info!(
            "neuron::process: spawning worker for model_id={} -> {} {:?}",
            model_id, cmd, args
        );

        let mut command = Command::new(cmd);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Apply caller-provided environment overrides.
        for (k, v) in extra_env {
            command.env(k, v);
        }

        let child = command.spawn()?;
        let pid = child.id();

        // Track the worker by PID.
        if let Ok(mut map) = self.workers.lock() {
            map.insert(pid, child);
        } else {
            warn!(
                "neuron::process: workers map lock poisoned when tracking pid={}",
                pid
            );
        }

        // Track the worker under its model identifier so that we can evict
        // all workers for a given model later.
        let model_id_string = model_id.to_string();
        if let Ok(mut map) = self.by_model.lock() {
            map.entry(model_id_string)
                .and_modify(|pids| pids.push(pid))
                .or_insert_with(|| vec![pid]);
        } else {
            warn!(
                "neuron::process: by_model map lock poisoned when tracking pid={} for model_id={}",
                pid, model_id
            );
        }

        Ok(WorkerHandle {
            model_id: model_id.to_string(),
            pid,
        })
    }

    /// Attempt to terminate a worker process gracefully by PID.
    ///
    /// If the PID is not known, this is a no-op.
    pub fn terminate_worker_by_pid(&self, pid: u32) {
        // Remove from PID → Child map and attempt to kill.
        if let Ok(mut map) = self.workers.lock() {
            if let Some(mut child) = map.remove(&pid) {
                info!("neuron::process: terminating worker pid={}", pid);
                if let Err(e) = child.kill() {
                    warn!("neuron::process: failed to kill worker pid={}: {e}", pid);
                }
            } else {
                info!(
                    "neuron::process: no tracked worker for pid={}; nothing to terminate",
                    pid
                );
            }
        } else {
            warn!(
                "neuron::process: workers map lock poisoned when terminating pid={}",
                pid
            );
        }

        // Also remove the PID from any model index entries.
        if let Ok(mut by_model) = self.by_model.lock() {
            by_model.retain(|_model, pids| {
                pids.retain(|p| *p != pid);
                !pids.is_empty()
            });
        } else {
            warn!(
                "neuron::process: by_model map lock poisoned when cleaning up pid={}",
                pid
            );
        }
    }

    /// Attempt to terminate a worker process gracefully using a [`WorkerHandle`].
    ///
    /// This is a convenience wrapper around [`ProcessManager::terminate_worker_by_pid`].
    pub fn terminate_worker_by_handle(&self, handle: WorkerHandle) {
        self.terminate_worker_by_pid(handle.pid);
    }

    /// Terminate all workers associated with a given `model_id`.
    ///
    /// This looks up the set of PIDs serving the model, attempts to kill each
    /// of them, and removes them from the internal maps. Unknown models are
    /// treated as a no-op.
    pub fn terminate_workers_for_model(&self, model_id: &str) {
        let pids = if let Ok(map) = self.by_model.lock() {
            map.get(model_id).cloned().unwrap_or_default()
        } else {
            warn!(
                "neuron::process: by_model map lock poisoned when terminating workers for model_id={}",
                model_id
            );
            Vec::new()
        };

        if pids.is_empty() {
            info!(
                "neuron::process: no tracked workers found for model_id={}; nothing to terminate",
                model_id
            );
            return;
        }

        info!(
            "neuron::process: terminating {} worker(s) for model_id={}: {:?}",
            pids.len(),
            model_id,
            pids
        );

        for pid in pids {
            self.terminate_worker_by_pid(pid);
        }
    }

    /// Attempt to terminate a worker process gracefully using an existing
    /// [`Child`] handle.
    ///
    /// This method is retained for compatibility with patterns where a
    /// `Child` is obtained externally, but new code should prefer
    /// [`ProcessManager::terminate_worker_by_pid`] and rely on the internal
    /// tracking map where possible.
    pub fn terminate_worker(&self, child: &mut Child) {
        let id = child.id();
        info!("neuron::process: terminating worker pid={}", id);
        if let Err(e) = child.kill() {
            warn!("neuron::process: failed to kill worker pid={}: {e}", id);
        }

        // Best-effort cleanup of the tracking map.
        if let Ok(mut map) = self.workers.lock() {
            map.remove(&id);
        }

        // Also remove the PID from any model index entries.
        if let Ok(mut by_model) = self.by_model.lock() {
            by_model.retain(|_model, pids| {
                pids.retain(|p| *p != id);
                !pids.is_empty()
            });
        }
    }
}
