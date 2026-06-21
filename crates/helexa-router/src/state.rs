use crate::config::{CortexEndpoint, RouterConfig};

/// Shared router state.
///
/// The skeleton (#70) holds only the static downstream cortex list from
/// config. Live multi-operator topology (per-cortex capacity + catalogue)
/// is added by the poller (#72), at which point this grows an
/// `Arc<RwLock<...>>` topology map alongside the static endpoints.
#[derive(Debug)]
pub struct RouterState {
    /// Downstream cortex endpoints, as configured.
    pub cortexes: Vec<CortexEndpoint>,
}

impl RouterState {
    pub fn from_config(config: &RouterConfig) -> Self {
        Self {
            cortexes: config.cortexes.clone(),
        }
    }
}
