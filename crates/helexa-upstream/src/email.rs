//! Transactional email for verification + password-reset links.
//!
//! Two transports: `Log` (dev — writes the link to the log so flows are
//! testable without a relay) and `Smtp` (lettre over rustls). Built from
//! `[email]` config.

use crate::config::EmailSettings;
use anyhow::{Context, Result};
use lettre::message::Mailbox;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

#[derive(Clone)]
pub enum EmailSender {
    /// Dev: log the message instead of sending.
    Log { from: String },
    Smtp {
        from: String,
        transport: AsyncSmtpTransport<Tokio1Executor>,
    },
}

impl EmailSender {
    pub fn from_config(cfg: &EmailSettings) -> Result<Self> {
        match cfg.provider.as_str() {
            "smtp" => {
                let url = cfg
                    .smtp_url
                    .as_deref()
                    .context("[email].smtp_url required when provider = \"smtp\"")?;
                let transport = AsyncSmtpTransport::<Tokio1Executor>::from_url(url)
                    .context("parsing [email].smtp_url")?
                    .build();
                Ok(EmailSender::Smtp {
                    from: cfg.from_addr.clone(),
                    transport,
                })
            }
            _ => Ok(EmailSender::Log {
                from: cfg.from_addr.clone(),
            }),
        }
    }

    /// Send a plaintext email. Errors are returned but the caller treats
    /// send failures as non-fatal to the request (the user can re-request).
    pub async fn send(&self, to: &str, subject: &str, body: &str) -> Result<()> {
        match self {
            EmailSender::Log { from } => {
                tracing::info!(%from, %to, %subject, body, "EMAIL (log transport)");
                Ok(())
            }
            EmailSender::Smtp { from, transport } => {
                let msg = Message::builder()
                    .from(from.parse::<Mailbox>().context("parsing from_addr")?)
                    .to(to.parse::<Mailbox>().context("parsing recipient")?)
                    .subject(subject)
                    .body(body.to_string())
                    .context("building message")?;
                transport.send(msg).await.context("sending email")?;
                Ok(())
            }
        }
    }
}
