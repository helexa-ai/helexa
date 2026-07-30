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
            "log" => Ok(EmailSender::Log {
                from: cfg.from_addr.clone(),
            }),
            other => anyhow::bail!(
                "[email].provider must be \"log\" or \"smtp\", got \"{other}\" — refusing \
                 to start rather than silently swallowing verification emails"
            ),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(provider: &str, url: Option<&str>) -> EmailSettings {
        EmailSettings {
            provider: provider.into(),
            smtp_url: url.map(String::from),
            from_addr: "helexa <no-reply@helexa.ai>".into(),
        }
    }

    #[test]
    fn log_provider_builds() {
        assert!(matches!(
            EmailSender::from_config(&settings("log", None)).unwrap(),
            EmailSender::Log { .. }
        ));
    }

    #[test]
    fn unknown_provider_is_refused() {
        // A typo must fail startup, not silently write links to the log.
        assert!(EmailSender::from_config(&settings("stmp", None)).is_err());
    }

    #[test]
    fn smtp_requires_url() {
        assert!(EmailSender::from_config(&settings("smtp", None)).is_err());
    }

    #[test]
    fn smtps_url_with_encoded_user_parses() {
        // The production shape: implicit-TLS submission with the mailbox
        // as the login, `@` percent-encoded per RFC 3986.
        let cfg = settings(
            "smtp",
            Some("smtps://no-reply%40helexa.ai:secret@mail.l4ir.net:465"),
        );
        assert!(matches!(
            EmailSender::from_config(&cfg).unwrap(),
            EmailSender::Smtp { .. }
        ));
    }
}
