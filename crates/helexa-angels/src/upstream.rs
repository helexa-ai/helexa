//! Client for helexa-upstream.
//!
//! Registration is delegated rather than reimplemented. helexa-upstream
//! already owns password policy, argon2 parameters, the verification email
//! (via Stalwart, from `no-reply@helexa.ai`), the unverified-signup reaper
//! and registration fingerprinting. Two implementations writing one
//! `users` table would drift, and the first symptom would be an account
//! that works on one property and not the other.
//!
//! The practical consequence is that a new investor confirms their address
//! through the ordinary helexa flow and then signs in here — which is also
//! correct, because it is one account for both properties by design.

use crate::state::AppState;
use anyhow::{Context, Result, bail};
use serde_json::json;

/// Register a new helexa account.
///
/// Upstream deliberately does not distinguish "created" from "address
/// already in use" (it no-ops on the unique-email conflict) so that this
/// endpoint cannot be used to enumerate accounts. We inherit that.
pub async fn register(state: &AppState, email: &str, password: &str) -> Result<()> {
    let url = format!(
        "{}/web/v1/register",
        state.config.upstream.base_url.trim_end_matches('/')
    );
    let resp = state
        .http
        .post(&url)
        // Tell upstream which front end this signup came from, so the
        // confirmation mail sends the investor back to the portal rather
        // than to helexa.ai, where the material they were invited to read
        // is nowhere to be seen. Upstream allowlists this.
        .json(&json!({
            "email": email,
            "password": password,
            "origin": state.config.site.base_url,
        }))
        .send()
        .await
        .with_context(|| format!("calling {url}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("upstream register returned {status}: {body}");
    }
    Ok(())
}

// The evaluation allocation that ships with a grant needs an endpoint
// upstream does not have yet (its allocation paths are either the signup
// grant or a redeemed top-up code). Added in A4 alongside the matching
// upstream handler, rather than left here calling a route that would 404
// on every grant.
