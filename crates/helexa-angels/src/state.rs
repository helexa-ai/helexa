//! Shared application state.

use crate::config::AngelsConfig;
use crate::content::Content;
use crate::notify::Notifier;
use sqlx::postgres::PgPool;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub pool: PgPool,
    pub config: Arc<AngelsConfig>,
    pub http: reqwest::Client,
    pub content: Content,
    pub notifier: Notifier,
}

impl AppState {
    pub fn new(pool: PgPool, config: AngelsConfig, notifier: Notifier) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.upstream.timeout_secs))
            .build()
            .unwrap_or_default();
        let content = Content::new(config.content.dir.clone());
        Self {
            pool,
            config: Arc::new(config),
            http,
            content,
            notifier,
        }
    }

    /// Context every template needs: branding, the contracting entity, and
    /// the contact address. Kept in one place so a page cannot accidentally
    /// render without the confidentiality footer.
    pub fn base_context(&self) -> Vec<(&'static str, String)> {
        vec![
            ("site_name", "helexa investor portal".to_string()),
            ("site_tagline", "investor portal".to_string()),
            ("brand_name", crate::BRAND_NAME.to_string()),
            ("entity_name", crate::ENTITY_NAME.to_string()),
            ("entity_note", crate::ENTITY_NOTE.to_string()),
            ("contact_email", self.config.site.contact_email.clone()),
        ]
    }
}
