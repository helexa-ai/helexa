//! Shared application state.

use crate::config::UpstreamConfig;
use crate::email::EmailSender;
use sqlx::postgres::PgPool;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub config: Arc<UpstreamConfig>,
    pub email: EmailSender,
}

impl AppState {
    pub fn new(pool: PgPool, config: UpstreamConfig, email: EmailSender) -> Self {
        Self {
            pool,
            config: Arc::new(config),
            email,
        }
    }
}
