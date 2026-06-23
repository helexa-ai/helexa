//! Shared application state.

use crate::config::UpstreamConfig;
use sqlx::postgres::PgPool;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub config: Arc<UpstreamConfig>,
}

impl AppState {
    pub fn new(pool: PgPool, config: UpstreamConfig) -> Self {
        Self {
            pool,
            config: Arc::new(config),
        }
    }
}
