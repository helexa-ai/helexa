use tokio::signal;
use tracing::info;

pub async fn wait_for_signal() {
    info!("waiting for shutdown signal");
    let _ = signal::ctrl_c().await;
}
