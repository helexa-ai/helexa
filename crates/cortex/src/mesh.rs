// SPDX-License-Identifier: PolyForm-Shield-1.0

use anyhow::Result;
use mesh::MeshHandle;
use tracing::info;

pub async fn start_mesh(node_id: Option<String>) -> Result<MeshHandle> {
    let id = node_id.unwrap_or_else(|| "anonymous-cortex".to_string());
    info!("joining mesh as {}", &id);
    let handle = mesh::MeshHandle::new(id);
    // TODO: real mesh join logic
    Ok(handle)
}
