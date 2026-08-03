//! Outbound notification for expressions of interest.
//!
//! Mirrors `helexa_upstream::email` — same lettre transport, same
//! `log`/`smtp` split — rather than sharing code across the crates,
//! because the two have different failure postures. Upstream's mail
//! carries a verification link the user is *waiting for*; this mail is a
//! notification to the operator, and its loss must never look to the
//! investor like their submission failed.
//!
//! So: the interest row is written first and the mail is best-effort. If
//! the relay is down the operator sees the submission via
//! `helexa-angels interest`, and nobody is told their message vanished.

use crate::config::EmailSettings;
use anyhow::{Context, Result};
use lettre::message::Mailbox;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

#[derive(Clone)]
pub enum Notifier {
    /// Dev, and a perfectly reasonable production choice while volumes
    /// are a handful of people: the submission is in the database either
    /// way, this only decides whether a mail also goes out.
    Log { from: String },
    Smtp {
        from: String,
        transport: Box<AsyncSmtpTransport<Tokio1Executor>>,
    },
}

impl Notifier {
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
                Ok(Notifier::Smtp {
                    from: cfg.from_addr.clone(),
                    transport: Box::new(transport),
                })
            }
            "log" => Ok(Notifier::Log {
                from: cfg.from_addr.clone(),
            }),
            other => anyhow::bail!(
                "[email].provider must be \"log\" or \"smtp\", got \"{other}\" — refusing to \
                 start rather than silently swallowing operator notifications"
            ),
        }
    }

    pub async fn send(&self, to: &str, subject: &str, body: &str) -> Result<()> {
        match self {
            Notifier::Log { from } => {
                tracing::info!(%from, %to, %subject, body, "NOTIFICATION (log transport)");
                Ok(())
            }
            Notifier::Smtp { from, transport } => {
                let msg = Message::builder()
                    .from(from.parse::<Mailbox>().context("parsing from_addr")?)
                    .to(to.parse::<Mailbox>().context("parsing recipient")?)
                    .subject(subject)
                    .body(body.to_string())
                    .context("building message")?;
                transport.send(msg).await.context("sending notification")?;
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
            notify_to: "angels@helexa.ai".into(),
        }
    }

    #[test]
    fn log_provider_builds() {
        assert!(matches!(
            Notifier::from_config(&settings("log", None)).unwrap(),
            Notifier::Log { .. }
        ));
    }

    #[test]
    fn unknown_provider_refuses_to_start() {
        // Failing closed matters: a typo that silently degraded to "no
        // mail" would lose investor submissions with no signal at all.
        assert!(Notifier::from_config(&settings("sendgrid", None)).is_err());
    }

    #[test]
    fn smtp_without_url_refuses_to_start() {
        assert!(Notifier::from_config(&settings("smtp", None)).is_err());
    }
}
