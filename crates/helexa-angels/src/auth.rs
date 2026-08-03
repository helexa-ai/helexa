//! Sessions and credential verification.
//!
//! Credentials are shared with helexa.ai (D2): a person has one helexa
//! account, and the same email and password work on both properties. That
//! is what lets an investor evaluate the platform they are being asked to
//! back.
//!
//! Sessions are **not** shared. helexa-upstream's `sessions` back the
//! public SPA, whose token lives in `localStorage` in an application that
//! renders markdown, runs a chat loop and fetches remote pages on the
//! user's behalf. Honouring those tokens here would put confidential
//! documents one script injection away from disclosure. The cookie issued
//! here is `HttpOnly` (unreadable from JS at all), `Secure`, `SameSite=Lax`
//! and — importantly — **host-only**: no `Domain` attribute, so it is
//! never sent to helexa.ai.

use crate::config::SessionSettings;
use crate::crypto;
use crate::error::{AngelsError, Result};
use axum::http::HeaderMap;
use axum::http::header::{COOKIE, SET_COOKIE};
use chrono::{DateTime, Utc};
use sqlx::Row;
use sqlx::postgres::PgPool;
use uuid::Uuid;

/// An authenticated visitor.
#[derive(Debug, Clone)]
pub struct Session {
    pub user_id: Uuid,
    pub email: String,
}

/// A password hash to verify against when the account does not exist.
///
/// Without this, "unknown address" returns in microseconds while "known
/// address, wrong password" takes argon2's deliberate ~100 ms — which
/// turns the sign-in form into an oracle for testing whether a given
/// person has an account here. Since holding an account on *this* portal
/// implies being an invited investor, that leak is more sensitive than
/// usual. Verifying against a real hash equalises the timing.
const DUMMY_PHC: &str = "$argon2id$v=19$m=19456,t=2,p=1$c29tZXNhbHR2YWx1ZQ$\
                         DdIDPWMLmtGh1jFkRLRcOEjbvRcOUJ8yqIrxCDQGLnU";

/// Verify an email/password pair against the shared `public.users` table.
///
/// Returns `BadCredentials` for a wrong password, an unknown address, and
/// an unverified address alike — the caller must not be able to tell them
/// apart. Unverified accounts are refused because a grant attaches to
/// whoever holds the address: allowing sign-in before the address is
/// confirmed would let someone claim an invitation sent to another person
/// simply by registering their email.
pub async fn verify_credentials(pool: &PgPool, email: &str, password: &str) -> Result<Session> {
    let row = sqlx::query(
        "SELECT id, email::text AS email, password_hash, email_verified \
         FROM public.users WHERE email = $1",
    )
    .bind(email)
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        // Burn the same time an existing account would have cost.
        let _ = crypto::verify_password(password, DUMMY_PHC);
        return Err(AngelsError::BadCredentials);
    };

    let phc: String = row.get("password_hash");
    if !crypto::verify_password(password, &phc) {
        return Err(AngelsError::BadCredentials);
    }
    if !row.get::<bool, _>("email_verified") {
        return Err(AngelsError::BadCredentials);
    }

    Ok(Session {
        user_id: row.get("id"),
        email: row.get("email"),
    })
}

/// Mint a session and return the raw cookie value. Only its sha256 is
/// stored, so a database disclosure does not yield usable sessions.
pub async fn issue_session(
    pool: &PgPool,
    user_id: Uuid,
    ttl_secs: u64,
    ip: Option<&str>,
    user_agent: Option<&str>,
) -> Result<String> {
    let raw = crypto::random_token();
    let expires: DateTime<Utc> = Utc::now() + chrono::Duration::seconds(ttl_secs as i64);
    sqlx::query(
        "INSERT INTO sessions (token_hash, user_id, expires_at, ip, user_agent) \
         VALUES ($1, $2, $3, $4, $5)",
    )
    .bind(crypto::sha256(&raw))
    .bind(user_id)
    .bind(expires)
    .bind(ip)
    .bind(user_agent)
    .execute(pool)
    .await?;
    Ok(raw)
}

/// Resolve a request's cookie to a live session, enforcing both the
/// absolute expiry and the idle timeout, and touching `last_seen_at`.
pub async fn resolve_session(
    pool: &PgPool,
    headers: &HeaderMap,
    settings: &SessionSettings,
) -> Option<Session> {
    let raw = cookie_value(headers, &settings.cookie_name)?;
    let hash = crypto::sha256(&raw);

    let row = sqlx::query(
        "SELECT s.user_id, u.email::text AS email \
         FROM sessions s JOIN public.users u ON u.id = s.user_id \
         WHERE s.token_hash = $1 \
           AND s.expires_at > now() \
           AND s.last_seen_at > now() - make_interval(secs => $2)",
    )
    .bind(&hash)
    .bind(settings.idle_secs as f64)
    .fetch_optional(pool)
    .await
    .ok()
    .flatten()?;

    // Touch last_seen so an active reader's session does not idle out
    // mid-read. Failure here is not fatal to the request.
    let _ = sqlx::query("UPDATE sessions SET last_seen_at = now() WHERE token_hash = $1")
        .bind(&hash)
        .execute(pool)
        .await;

    Some(Session {
        user_id: row.get("user_id"),
        email: row.get("email"),
    })
}

/// Drop a session server-side (sign out). Revocation is real, not just a
/// cleared cookie.
pub async fn destroy_session(pool: &PgPool, headers: &HeaderMap, cookie_name: &str) {
    if let Some(raw) = cookie_value(headers, cookie_name) {
        let _ = sqlx::query("DELETE FROM sessions WHERE token_hash = $1")
            .bind(crypto::sha256(&raw))
            .execute(pool)
            .await;
    }
}

/// Build the `Set-Cookie` header value.
///
/// No `Domain` attribute: that makes the cookie host-only, so it is never
/// transmitted to helexa.ai or any other subdomain. `SameSite=Lax` keeps
/// it off cross-site requests while still surviving the ordinary case of
/// following a link from an email.
pub fn session_cookie(name: &str, value: &str, max_age: u64, secure: bool) -> String {
    let mut c = format!("{name}={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}");
    if secure {
        c.push_str("; Secure");
    }
    c
}

/// The cookie that clears a session.
pub fn clearing_cookie(name: &str, secure: bool) -> String {
    let mut c = format!("{name}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0");
    if secure {
        c.push_str("; Secure");
    }
    c
}

/// A short-lived cookie carrying a pending invite across the sign-in or
/// registration detour, so the visitor is not asked to paste the code
/// again after confirming their email.
pub fn pending_invite_cookie(value: &str, secure: bool) -> String {
    let mut c = format!(
        "angels_invite={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age={}",
        3600 * 24
    );
    if secure {
        c.push_str("; Secure");
    }
    c
}

/// Read one cookie out of a request's `Cookie` headers.
pub fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    for hv in headers.get_all(COOKIE).iter() {
        let Ok(s) = hv.to_str() else { continue };
        for part in s.split(';') {
            let part = part.trim();
            if let Some(rest) = part.strip_prefix(name)
                && let Some(v) = rest.strip_prefix('=')
            {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Client IP, honouring the edge proxy's `X-Forwarded-For`.
pub fn client_ip(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn user_agent(headers: &HeaderMap) -> Option<String> {
    headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.chars().take(400).collect())
}

/// Helper for handlers that need to set a cookie on a redirect.
pub fn with_cookie(mut headers: HeaderMap, cookie: String) -> HeaderMap {
    if let Ok(v) = cookie.parse() {
        headers.append(SET_COOKIE, v);
    }
    headers
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn headers_with(cookie: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(COOKIE, HeaderValue::from_str(cookie).unwrap());
        h
    }

    #[test]
    fn reads_a_cookie_among_several() {
        let h = headers_with("foo=1; angels_session=abc123; bar=2");
        assert_eq!(
            cookie_value(&h, "angels_session").as_deref(),
            Some("abc123")
        );
        assert_eq!(cookie_value(&h, "missing"), None);
    }

    #[test]
    fn does_not_match_a_cookie_by_prefix() {
        // "angels_session_other" must not satisfy a lookup for
        // "angels_session" — a prefix match here would let an attacker
        // who can set any cookie shadow the session name.
        let h = headers_with("angels_session_other=nope");
        assert_eq!(cookie_value(&h, "angels_session"), None);
    }

    #[test]
    fn session_cookie_is_httponly_samesite_and_host_only() {
        let c = session_cookie("angels_session", "tok", 3600, true);
        assert!(c.contains("HttpOnly"), "{c}");
        assert!(c.contains("Secure"), "{c}");
        assert!(c.contains("SameSite=Lax"), "{c}");
        // The absence of Domain is the whole point: a Domain=.helexa.ai
        // cookie would travel to the public SPA.
        assert!(!c.contains("Domain"), "cookie must be host-only: {c}");
    }

    #[test]
    fn insecure_cookie_only_when_explicitly_asked() {
        let c = session_cookie("angels_session", "tok", 60, false);
        assert!(!c.contains("Secure"), "{c}");
    }

    #[test]
    fn clearing_cookie_expires_immediately() {
        let c = clearing_cookie("angels_session", true);
        assert!(c.contains("Max-Age=0"), "{c}");
        assert!(c.contains("HttpOnly"), "{c}");
    }

    #[test]
    fn forwarded_for_takes_the_first_hop() {
        let mut h = HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            HeaderValue::from_static("1.2.3.4, 10.0.0.1"),
        );
        assert_eq!(client_ip(&h).as_deref(), Some("1.2.3.4"));
    }

    #[test]
    fn user_agent_is_bounded() {
        let mut h = HeaderMap::new();
        let long = "x".repeat(2000);
        h.insert("user-agent", HeaderValue::from_str(&long).unwrap());
        assert_eq!(user_agent(&h).map(|s| s.len()), Some(400));
    }

    #[test]
    fn dummy_hash_is_a_valid_argon2_phc() {
        // If this ever stopped parsing, the timing-equalisation branch
        // would return instantly and reintroduce the enumeration oracle
        // it exists to close.
        assert!(!crypto::verify_password("anything", DUMMY_PHC));
        assert!(
            argon2::password_hash::PasswordHash::new(DUMMY_PHC).is_ok(),
            "DUMMY_PHC must parse as a real argon2 hash"
        );
    }
}
