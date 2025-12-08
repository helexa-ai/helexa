use std::net::SocketAddr;

use mesh::MeshHandle;
use tracing::info;

pub fn spawn(addr: SocketAddr, _mesh: MeshHandle) {
    info!("starting portal role on {}", addr);

    // TODO: http server for web ui + billing hooks
    tokio::spawn(async move {
        // placeholder
    });
}
