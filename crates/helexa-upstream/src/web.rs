//! `/web/v1` — the human-facing account API the helexa.ai frontend (#F4)
//! consumes: email+password auth (register / verify / login / reset),
//! API-key CRUD with per-key limits, and the account balance. Web sessions
//! are JWTs, **distinct** from inference API keys.
//!
//! Errors use a plain JSON shape `{ "error": { "message", "code" } }` (web
//! clients, not OpenAI clients — the #63 envelope is the authz surface).
//!
//! Silent fingerprint abuse (no clue to the abuser): registration captures
//! the browser fingerprint and always succeeds; when ≥ threshold accounts
//! share one fingerprint, all are silently `deactivated` (keys then resolve
//! as ordinary `401`s at the authz surface — never a "banned" signal).

use crate::crypto::{generate_api_key, hash_password, random_token, sha256, verify_password};
use crate::state::AppState;
use axum::extract::{Path, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::{Extension, Router};
use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::Row;
use uuid::Uuid;

pub fn router(state: &AppState) -> Router<AppState> {
    // Session-only: identity management and the full account view.
    //
    // Key management stays here deliberately (#264). If an inference key
    // could mint keys, one leaked credential would become a foothold that
    // SURVIVES revoking the original — a spending problem turning into an
    // eviction problem. `/account` stays too: it carries `angel_access`,
    // and whatever field is added to it next would otherwise be exposed to
    // inference credentials by default.
    let session_only = Router::new()
        .route("/web/v1/account", get(account))
        .route("/web/v1/keys", get(list_keys).post(create_key))
        .route("/web/v1/keys/{id}/archive", post(archive_key))
        .route(
            "/web/v1/keys/{id}/limit",
            axum::routing::patch(update_key_limit),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_session,
        ));

    // Session **or** API key: reading and refilling the allocation.
    //
    // A service that watches its own balance should not have to hold its
    // owner's password — the credential that can mint and revoke keys — to
    // read a number it is already entitled to spend against.
    let account_auth = Router::new()
        .route("/web/v1/allocation", get(allocation))
        .route("/web/v1/redeem", post(redeem))
        .route("/web/v1/topup/request", post(request_topup))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_account,
        ));

    Router::new()
        .route("/web/v1/features", get(features))
        .route("/web/v1/register", post(register))
        .route("/web/v1/verify", post(verify))
        .route("/web/v1/login", post(login))
        .route("/web/v1/password-reset/request", post(reset_request))
        .route("/web/v1/password-reset/confirm", post(reset_confirm))
        .merge(session_only)
        .merge(account_auth)
}

// ── errors ──────────────────────────────────────────────────────────

enum WebError {
    BadRequest(&'static str),
    Unauthorized,
    Internal,
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            WebError::BadRequest(m) => (StatusCode::BAD_REQUEST, "bad_request", m),
            WebError::Unauthorized => (StatusCode::UNAUTHORIZED, "unauthorized", "unauthorized"),
            WebError::Internal => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "internal error",
            ),
        };
        (
            status,
            Json(json!({"error": {"message": message, "code": code}})),
        )
            .into_response()
    }
}

impl From<sqlx::Error> for WebError {
    fn from(e: sqlx::Error) -> Self {
        tracing::error!(error = %e, "web db error");
        WebError::Internal
    }
}

type WebResult<T> = Result<T, WebError>;

// ── sessions (JWT) ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String, // user id
    exp: usize,
}

fn mint_session(state: &AppState, user_id: Uuid) -> WebResult<String> {
    let exp = (Utc::now() + Duration::seconds(state.config.auth.session_ttl_secs as i64))
        .timestamp() as usize;
    let claims = Claims {
        sub: user_id.to_string(),
        exp,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(state.config.auth.jwt_secret.as_bytes()),
    )
    .map_err(|_| WebError::Internal)
}

/// Authenticated user id, injected by [`require_session`].
#[derive(Clone)]
struct AuthUser(Uuid);

async fn require_session(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let Some(token) = token else {
        return WebError::Unauthorized.into_response();
    };
    let decoded = decode::<Claims>(
        token,
        &DecodingKey::from_secret(state.config.auth.jwt_secret.as_bytes()),
        &Validation::default(),
    );
    match decoded
        .ok()
        .and_then(|d| Uuid::parse_str(&d.claims.sub).ok())
    {
        Some(uid) => {
            req.extensions_mut().insert(AuthUser(uid));
            next.run(req).await
        }
        None => WebError::Unauthorized.into_response(),
    }
}

/// Authenticated **account**, injected by [`require_account`]. Unlike
/// [`AuthUser`] this says nothing about *who* is calling — only which
/// account the call bills to — because an API key identifies an account
/// without identifying a person.
#[derive(Clone, Copy)]
struct AuthAccount(Uuid);

/// Accept **either** a web session or an inference API key, and resolve
/// both to an account id (#264).
///
/// Routes behind this may read and refill an allocation. They may not
/// touch keys: a service watching its own balance should not have to
/// hold the credential that can mint and revoke credentials. See the
/// router for why key management stays session-only.
///
/// The two token shapes are unambiguous — API keys carry the
/// `sk-helexa-` prefix (`crypto::generate_api_key`) and JWTs cannot — so
/// this dispatches on the prefix rather than trying to decode a session
/// and falling back, which would log a decode failure for every
/// perfectly valid key.
async fn require_account(State(state): State<AppState>, mut req: Request, next: Next) -> Response {
    let Some(token) = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .map(str::to_string)
    else {
        return WebError::Unauthorized.into_response();
    };

    let account = if token.starts_with("sk-helexa-") {
        // `resolve_key` already requires both key and account to be
        // `active`, so an archived key or a deactivated account
        // authenticates nothing — no extra check needed here.
        match crate::ledger::resolve_key(&state.pool, &sha256(&token)).await {
            Ok(Some(principal)) => principal.account_id,
            Ok(None) => return WebError::Unauthorized.into_response(),
            Err(e) => {
                tracing::error!(error = %e, "key resolve failed on account route");
                return WebError::Internal.into_response();
            }
        }
    } else {
        let uid = decode::<Claims>(
            &token,
            &DecodingKey::from_secret(state.config.auth.jwt_secret.as_bytes()),
            &Validation::default(),
        )
        .ok()
        .and_then(|d| Uuid::parse_str(&d.claims.sub).ok());
        let Some(uid) = uid else {
            return WebError::Unauthorized.into_response();
        };
        match account_id_for(&state, uid).await {
            Ok(a) => a,
            Err(e) => return e.into_response(),
        }
    };

    req.extensions_mut().insert(AuthAccount(account));
    next.run(req).await
}

/// The caller's single account id.
async fn account_id_for(state: &AppState, user_id: Uuid) -> WebResult<Uuid> {
    let row = sqlx::query("SELECT id FROM accounts WHERE owner_user_id = $1")
        .bind(user_id)
        .fetch_optional(&state.pool)
        .await?;
    row.map(|r| r.get::<Uuid, _>("id"))
        .ok_or(WebError::Internal)
}

// ── feature gates ───────────────────────────────────────────────────

/// Public, unauthenticated: the product feature gates the SPA reads at
/// chat load. Anonymous grounding (#191) is gated here so an operator
/// can kill it with a config edit + restart instead of a site rebuild.
/// The SPA fails closed for anonymous sessions when this endpoint is
/// unreachable.
async fn features(State(state): State<AppState>) -> Response {
    Json(json!({
        "anon_web_search": state.config.features.anon_web_search,
    }))
    .into_response()
}

// ── auth lifecycle ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct RegisterReq {
    email: String,
    password: String,
    #[serde(default)]
    fingerprint: Option<String>,
    /// Front end the signup came from, so the verification mail sends the
    /// person back where they started. Allowlisted — see
    /// [`resolve_origin`].
    #[serde(default)]
    origin: Option<String>,
}

/// Resolve a caller-supplied origin against the configured allowlist.
///
/// The result is embedded in an email we send and sign, so an unchecked
/// value would turn signup into a phishing-link generator carrying our
/// domain's reputation. Unknown or absent → the default front end.
fn resolve_origin(state: &AppState, requested: Option<&str>) -> String {
    resolve_origin_with(&state.config.auth, requested)
}

/// Split out from [`resolve_origin`] so the allowlist can be tested without
/// standing up an `AppState` — this is the check that keeps /register from
/// becoming a phishing-link generator, so it needs to be exercised.
fn resolve_origin_with(auth: &crate::config::AuthSettings, requested: Option<&str>) -> String {
    let default = auth.app_base_url.clone();
    let Some(req) = requested else { return default };
    let req = req.trim_end_matches('/');
    if req == default.trim_end_matches('/')
        || auth
            .additional_app_origins
            .iter()
            .any(|o| o.trim_end_matches('/') == req)
    {
        req.to_string()
    } else {
        tracing::warn!(origin = %req, "signup origin not allowlisted — using the default");
        default
    }
}

/// `POST /web/v1/register` — always returns `202`, regardless of whether the
/// email was new, already taken, or fingerprint-flagged (no enumeration, no
/// abuse clue).
async fn register(State(state): State<AppState>, Json(req): Json<RegisterReq>) -> Response {
    match register_inner(&state, req).await {
        Ok(()) | Err(WebError::BadRequest(_)) => {}
        Err(e) => return e.into_response(),
    }
    // Generic 202 whatever happened above (except hard server errors).
    StatusCode::ACCEPTED.into_response()
}

async fn register_inner(state: &AppState, req: RegisterReq) -> WebResult<()> {
    if !req.email.contains('@') {
        return Err(WebError::BadRequest("invalid email"));
    }
    if req.password.len() < 8 {
        return Err(WebError::BadRequest("password too short (min 8)"));
    }
    let phc = hash_password(&req.password).map_err(|_| WebError::Internal)?;

    // Insert the user; a duplicate email silently no-ops (no enumeration).
    let user_id: Option<Uuid> = sqlx::query(
        "INSERT INTO users (email, password_hash, registration_fingerprint) \
         VALUES ($1, $2, $3) ON CONFLICT (email) DO NOTHING RETURNING id",
    )
    .bind(&req.email)
    .bind(&phc)
    .bind(&req.fingerprint)
    .fetch_optional(&state.pool)
    .await?
    .map(|r| r.get("id"));

    let Some(user_id) = user_id else {
        // Address already registered. If it is still UNVERIFIED, this is
        // someone retrying a signup they never completed — re-send a fresh
        // link (and adopt the password they just chose) so they are not
        // stuck waiting for a mail that would otherwise never come: the
        // old code returned 202 here having sent nothing at all.
        //
        // Only ever for unverified accounts. Doing this for a verified one
        // would let anyone reset a stranger's password by "registering"
        // their address.
        return resend_verification_if_unverified(state, &req.email, &phc).await;
    };

    // Account with the flat free grant.
    sqlx::query("INSERT INTO accounts (owner_user_id, allocation_total) VALUES ($1, $2)")
        .bind(user_id)
        .bind(state.config.grant.free_token_grant)
        .execute(&state.pool)
        .await?;

    // Silent fingerprint abuse handling.
    if let Some(fp) = req.fingerprint.as_deref().filter(|f| !f.is_empty()) {
        apply_fingerprint_policy(state, fp).await?;
    }

    // Email verification link.
    let token = random_token();
    let expires: DateTime<Utc> =
        Utc::now() + Duration::seconds(state.config.auth.email_token_ttl_secs as i64);
    sqlx::query(
        "INSERT INTO email_tokens (token_hash, user_id, kind, expires_at) \
         VALUES ($1, $2, 'verify', $3)",
    )
    .bind(sha256(&token))
    .bind(user_id)
    .bind(expires)
    .execute(&state.pool)
    .await?;

    let origin = resolve_origin(state, req.origin.as_deref());
    let link = format!("{origin}/verify?token={token}");
    let _ = state
        .email
        .send(
            &req.email,
            "Confirm your helexa email address",
            &format!(
                "Welcome to helexa.\n\n\
                 Confirm this address to finish setting up your account:\n\n\
                 {link}\n\n\
                 The link is valid for 24 hours. Once confirmed, sign in at \
                 {app} and you're ready to go.\n\n\
                 If you didn't create a helexa account, you can ignore this \
                 message — nothing has been set up and the address will be \
                 released.\n\n\
                 helexa\n",
                app = origin,
            ),
        )
        .await;
    Ok(())
}

/// Count accounts sharing `fp`; flag them, and silently deactivate all once
/// the count reaches the configured threshold. No response difference — the
/// abuser gets no signal.
async fn apply_fingerprint_policy(state: &AppState, fp: &str) -> WebResult<()> {
    let count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM users WHERE registration_fingerprint = $1")
            .bind(fp)
            .fetch_one(&state.pool)
            .await?;
    if count > 1 {
        sqlx::query(
            "UPDATE accounts SET fingerprint_flagged = true \
             WHERE owner_user_id IN (SELECT id FROM users WHERE registration_fingerprint = $1)",
        )
        .bind(fp)
        .execute(&state.pool)
        .await?;
    }
    if count >= state.config.abuse.fingerprint_account_threshold {
        let res = sqlx::query(
            "UPDATE accounts SET status = 'deactivated' \
             WHERE owner_user_id IN (SELECT id FROM users WHERE registration_fingerprint = $1)",
        )
        .bind(fp)
        .execute(&state.pool)
        .await?;
        tracing::warn!(
            fingerprint = fp,
            accounts = res.rows_affected(),
            "silently deactivated fingerprint-abusing accounts"
        );
    }
    Ok(())
}

#[derive(Deserialize)]
struct TokenReq {
    token: String,
}

/// Re-send a verification link for an existing **unverified** account,
/// adopting the password supplied by this signup attempt. No-op (and no
/// signal to the caller) for verified accounts or unknown addresses, so
/// the endpoint stays non-enumerating.
async fn resend_verification_if_unverified(
    state: &AppState,
    email: &str,
    password_hash: &str,
) -> WebResult<()> {
    let row = sqlx::query("SELECT id, email_verified FROM users WHERE email = $1")
        .bind(email)
        .fetch_optional(&state.pool)
        .await?;
    let Some(row) = row else { return Ok(()) };
    let verified: bool = row.get("email_verified");
    if verified {
        return Ok(());
    }
    let user_id: Uuid = row.get("id");

    // Adopt the new password: the person proving they want this address is
    // the one who will verify it. Harmless while unverified — the account
    // cannot be signed into until the link is clicked.
    sqlx::query("UPDATE users SET password_hash = $1 WHERE id = $2")
        .bind(password_hash)
        .bind(user_id)
        .execute(&state.pool)
        .await?;

    // Retire any outstanding verify tokens so only the newest link works.
    sqlx::query(
        "UPDATE email_tokens SET consumed_at = now() \
         WHERE user_id = $1 AND kind = 'verify' AND consumed_at IS NULL",
    )
    .bind(user_id)
    .execute(&state.pool)
    .await?;

    let token = random_token();
    let expires: DateTime<Utc> =
        Utc::now() + Duration::seconds(state.config.auth.email_token_ttl_secs as i64);
    sqlx::query(
        "INSERT INTO email_tokens (token_hash, user_id, kind, expires_at) \
         VALUES ($1, $2, 'verify', $3)",
    )
    .bind(sha256(&token))
    .bind(user_id)
    .bind(expires)
    .execute(&state.pool)
    .await?;

    let link = format!("{}/verify?token={token}", state.config.auth.app_base_url);
    let _ = state
        .email
        .send(
            email,
            "Confirm your helexa email address",
            &format!(
                "Here is a fresh confirmation link.\n\n\
                 Confirm this address to finish setting up your account:\n\n\
                 {link}\n\n\
                 The link is valid for 24 hours. Once confirmed, sign in at \
                 {app} and you're ready to go.\n\n\
                 If you didn't create a helexa account, you can ignore this \
                 message — nothing has been set up and the address will be \
                 released.\n\n\
                 helexa\n",
                app = state.config.auth.app_base_url,
            ),
        )
        .await;
    Ok(())
}

/// `POST /web/v1/verify` — consume a verification token, mark verified.
async fn verify(State(state): State<AppState>, Json(req): Json<TokenReq>) -> WebResult<Response> {
    let row = sqlx::query(
        "UPDATE email_tokens SET consumed_at = now() \
         WHERE token_hash = $1 AND kind = 'verify' AND consumed_at IS NULL AND expires_at > now() \
         RETURNING user_id",
    )
    .bind(sha256(&req.token))
    .fetch_optional(&state.pool)
    .await?;
    let Some(row) = row else {
        return Err(WebError::BadRequest("invalid or expired token"));
    };
    let user_id: Uuid = row.get("user_id");
    sqlx::query("UPDATE users SET email_verified = true WHERE id = $1")
        .bind(user_id)
        .execute(&state.pool)
        .await?;
    Ok(StatusCode::OK.into_response())
}

#[derive(Deserialize)]
struct LoginReq {
    email: String,
    password: String,
}

/// `POST /web/v1/login` — verify password + email-verified → session JWT.
async fn login(State(state): State<AppState>, Json(req): Json<LoginReq>) -> WebResult<Response> {
    let row = sqlx::query("SELECT id, password_hash, email_verified FROM users WHERE email = $1")
        .bind(&req.email)
        .fetch_optional(&state.pool)
        .await?;
    // Generic 401 for every failure mode (no enumeration).
    let Some(row) = row else {
        return Err(WebError::Unauthorized);
    };
    let phc: String = row.get("password_hash");
    let verified: bool = row.get("email_verified");
    if !verify_password(&req.password, &phc) || !verified {
        return Err(WebError::Unauthorized);
    }
    let user_id: Uuid = row.get("id");
    let token = mint_session(&state, user_id)?;
    Ok(Json(json!({
        "token": token,
        "expires_in": state.config.auth.session_ttl_secs,
    }))
    .into_response())
}

#[derive(Deserialize)]
struct EmailReq {
    email: String,
}

/// Delete unverified signups that can no longer be completed, freeing the
/// address for a fresh attempt (#lockout).
///
/// Without this an abandoned or mistyped signup holds its email address
/// forever: `register` no-ops on the unique-email conflict, login refuses
/// an unverified account, and a password reset rotates the hash without
/// setting `email_verified` — so the address becomes permanently
/// unusable.
///
/// Two conditions, both required:
///  - older than `unverified_grace_secs`, and
///  - holding **no live verification token** (unconsumed and unexpired).
///
/// The second is what makes this race-free: an account whose link is
/// still clickable is never reaped, even at the instant the grace period
/// lapses, and a re-sent link extends the reprieve without any extra
/// bookkeeping. Verified accounts are never touched. `email_tokens`,
/// `accounts` and `sessions` cascade on delete.
pub async fn reap_unverified(pool: &sqlx::PgPool, grace_secs: i64) -> Result<u64, sqlx::Error> {
    let result = sqlx::query(
        "DELETE FROM users u \
         WHERE NOT u.email_verified \
           AND u.created_at < now() - make_interval(secs => $1) \
           AND NOT EXISTS ( \
                 SELECT 1 FROM email_tokens t \
                 WHERE t.user_id = u.id AND t.kind = 'verify' \
                   AND t.consumed_at IS NULL AND t.expires_at > now() \
           )",
    )
    .bind(grace_secs as f64)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// `POST /web/v1/password-reset/request` — always `202` (no enumeration);
/// mints + emails a reset token only if the account exists.
async fn reset_request(State(state): State<AppState>, Json(req): Json<EmailReq>) -> Response {
    // The inner only ever yields `Internal` (DB failure); a missing email is
    // Ok(()) so there's no enumeration. Surface 500 on a real error, else 202.
    match reset_request_inner(&state, &req.email).await {
        Ok(()) => StatusCode::ACCEPTED.into_response(),
        Err(e) => e.into_response(),
    }
}

async fn reset_request_inner(state: &AppState, email: &str) -> WebResult<()> {
    let row = sqlx::query("SELECT id FROM users WHERE email = $1")
        .bind(email)
        .fetch_optional(&state.pool)
        .await?;
    let Some(row) = row else { return Ok(()) };
    let user_id: Uuid = row.get("id");
    let token = random_token();
    let expires: DateTime<Utc> =
        Utc::now() + Duration::seconds(state.config.auth.email_token_ttl_secs as i64);
    sqlx::query(
        "INSERT INTO email_tokens (token_hash, user_id, kind, expires_at) \
         VALUES ($1, $2, 'reset', $3)",
    )
    .bind(sha256(&token))
    .bind(user_id)
    .bind(expires)
    .execute(&state.pool)
    .await?;
    let link = format!("{}/reset?token={token}", state.config.auth.app_base_url);
    let _ = state
        .email
        .send(
            email,
            "Reset your helexa password",
            &format!(
                "Someone asked to reset the password for this helexa \
                 account.\n\n\
                 {link}\n\n\
                 The link is valid for 24 hours and can be used once.\n\n\
                 If that wasn't you, ignore this message — your password has \
                 not changed and nobody can use this link without access to \
                 your inbox.\n\n\
                 helexa\n"
            ),
        )
        .await;
    Ok(())
}

#[derive(Deserialize)]
struct ResetConfirmReq {
    token: String,
    new_password: String,
}

/// `POST /web/v1/password-reset/confirm` — consume reset token, rotate hash.
async fn reset_confirm(
    State(state): State<AppState>,
    Json(req): Json<ResetConfirmReq>,
) -> WebResult<Response> {
    if req.new_password.len() < 8 {
        return Err(WebError::BadRequest("password too short (min 8)"));
    }
    let row = sqlx::query(
        "UPDATE email_tokens SET consumed_at = now() \
         WHERE token_hash = $1 AND kind = 'reset' AND consumed_at IS NULL AND expires_at > now() \
         RETURNING user_id",
    )
    .bind(sha256(&req.token))
    .fetch_optional(&state.pool)
    .await?;
    let Some(row) = row else {
        return Err(WebError::BadRequest("invalid or expired token"));
    };
    let user_id: Uuid = row.get("user_id");
    let phc = hash_password(&req.new_password).map_err(|_| WebError::Internal)?;
    sqlx::query("UPDATE users SET password_hash = $1 WHERE id = $2")
        .bind(phc)
        .bind(user_id)
        .execute(&state.pool)
        .await?;
    Ok(StatusCode::OK.into_response())
}

// ── account + keys (protected) ──────────────────────────────────────

async fn account(
    State(state): State<AppState>,
    Extension(user): Extension<AuthUser>,
) -> WebResult<Response> {
    let acct = account_id_for(&state, user.0).await?;
    let row = sqlx::query(
        "SELECT allocation_total, allocation_spent, allocation_reserved FROM accounts WHERE id = $1",
    )
    .bind(acct)
    .fetch_one(&state.pool)
    .await?;
    // Whether a self-service top-up is available right now, so the
    // dashboard can show the offer (and why it is unavailable) without a
    // second round trip. The grant path re-checks, so this is display
    // only — a stale page cannot conjure an extra top-up.
    let (topup_available, topup_reason) =
        match crate::topup::auto_eligibility(&state.pool, acct).await {
            Ok(Ok(())) => (true, None),
            Ok(Err(reason)) => (false, Some(reason.to_string())),
            Err(e) => {
                tracing::warn!(error = %e, "top-up eligibility check failed");
                (false, None)
            }
        };
    Ok(Json(json!({
        "account_id": acct.to_string(),
        "allocation_total": row.get::<i64, _>("allocation_total"),
        "allocation_spent": row.get::<i64, _>("allocation_spent"),
        "allocation_reserved": row.get::<i64, _>("allocation_reserved"),
        "topup_available": topup_available,
        "topup_reason": topup_reason,
        "angel_access": has_angel_access(&state, user.0).await,
    }))
    .into_response())
}

/// `GET /web/v1/allocation` — the account's token balance, readable with
/// **either** a session or an API key (#264).
///
/// Deliberately narrower than [`account`]: balance and top-up
/// availability, nothing else. `/account` also returns `angel_access`,
/// and whatever field is added to it next would otherwise be handed to
/// every inference key by default. A separate projection means widening
/// the account view can never widen what a key can see.
async fn allocation(
    State(state): State<AppState>,
    Extension(AuthAccount(acct)): Extension<AuthAccount>,
) -> WebResult<Response> {
    let row = sqlx::query(
        "SELECT allocation_total, allocation_spent, allocation_reserved FROM accounts WHERE id = $1",
    )
    .bind(acct)
    .fetch_one(&state.pool)
    .await?;
    // Same display-only eligibility the dashboard uses, so an automated
    // caller can check before asking and avoid provoking a 409. The grant
    // path re-checks, so a stale read cannot conjure a top-up.
    let (topup_available, topup_reason) =
        match crate::topup::auto_eligibility(&state.pool, acct).await {
            Ok(Ok(())) => (true, None),
            Ok(Err(reason)) => (false, Some(reason.to_string())),
            Err(e) => {
                tracing::warn!(error = %e, "top-up eligibility check failed");
                (false, None)
            }
        };
    Ok(Json(json!({
        "allocation_total": row.get::<i64, _>("allocation_total"),
        "allocation_spent": row.get::<i64, _>("allocation_spent"),
        "allocation_reserved": row.get::<i64, _>("allocation_reserved"),
        "topup_available": topup_available,
        "topup_reason": topup_reason,
    }))
    .into_response())
}

/// Whether this user holds any investor-portal grant.
///
/// A **boolean, and deliberately nothing more**. helexa.ai is a static
/// bundle served to everyone, so anything it receives is effectively
/// public: a flag saying "this account has something" is safe there,
/// whereas a list of round names would publish which programmes exist and
/// what they are called. The portal keeps that server-side.
///
/// helexa-angels owns these tables (schema `angels`); this is a read-only
/// peek across the schema boundary, guarded with `to_regclass` so a
/// database where angels has never run — a fresh deployment, or upstream
/// running ahead of it — returns false instead of erroring. Never fails
/// the account request: worst case the header link is missing, and the
/// investor still has the link they were sent.
async fn has_angel_access(state: &AppState, user_id: Uuid) -> bool {
    let res = sqlx::query(
        "SELECT EXISTS ( \
             SELECT 1 FROM angels.grants g JOIN angels.rounds r ON r.slug = g.round_slug \
             WHERE g.user_id = $1 AND g.state = 'active' AND r.status <> 'draft' \
         ) AS ok \
         WHERE to_regclass('angels.grants') IS NOT NULL",
    )
    .bind(user_id)
    .fetch_optional(&state.pool)
    .await;

    match res {
        Ok(Some(row)) => row.get::<bool, _>("ok"),
        Ok(None) => false,
        Err(e) => {
            tracing::debug!(error = %e, "angel access check unavailable");
            false
        }
    }
}

/// `POST /web/v1/topup/request` — self-service top-up.
///
/// Lets an account that is running low grant itself more allocation, so
/// somebody evaluating helexa is not blocked waiting on the operator.
/// Eligibility (threshold, per-account ceiling, cooldown) is enforced
/// server-side from `app_config`; a refusal explains itself, since the
/// caller is authenticated and asking about their own account.
async fn request_topup(
    State(state): State<AppState>,
    Extension(AuthAccount(acct)): Extension<AuthAccount>,
) -> WebResult<Response> {
    use crate::topup::AutoTopUpError;
    match crate::topup::auto_grant(&state.pool, acct).await {
        Ok(grant) => {
            tracing::info!(
                account = %acct,
                value = grant.value,
                used = grant.used_count,
                max = grant.max_count,
                "self-service top-up granted"
            );
            Ok(Json(json!({
                "value": grant.value,
                "allocation_total": grant.allocation_total,
                "used_count": grant.used_count,
                "max_count": grant.max_count,
            }))
            .into_response())
        }
        Err(AutoTopUpError::Db(e)) => Err(WebError::from(e)),
        // Every other variant is the caller's own account state, not a
        // server fault: 409 with the reason so the UI can say why.
        Err(reason) => Ok((
            StatusCode::CONFLICT,
            Json(json!({ "error": reason.to_string() })),
        )
            .into_response()),
    }
}

async fn list_keys(
    State(state): State<AppState>,
    Extension(user): Extension<AuthUser>,
) -> WebResult<Response> {
    let acct = account_id_for(&state, user.0).await?;
    let rows = sqlx::query(
        "SELECT id, key_prefix, label, status, limit_kind, limit_value, key_spent, key_reserved, \
                created_at \
         FROM api_keys WHERE account_id = $1 ORDER BY created_at DESC",
    )
    .bind(acct)
    .fetch_all(&state.pool)
    .await?;
    let keys: Vec<_> = rows
        .iter()
        .map(|r| {
            json!({
                "id": r.get::<Uuid, _>("id").to_string(),
                "prefix": r.get::<String, _>("key_prefix"),
                "label": r.get::<String, _>("label"),
                "status": r.get::<String, _>("status"),
                "limit_kind": r.get::<String, _>("limit_kind"),
                "limit_value": r.get::<i64, _>("limit_value"),
                "spent": r.get::<i64, _>("key_spent"),
                "reserved": r.get::<i64, _>("key_reserved"),
                "created_at": r.get::<DateTime<Utc>, _>("created_at").to_rfc3339(),
            })
        })
        .collect();
    Ok(Json(json!({ "keys": keys })).into_response())
}

#[derive(Deserialize)]
struct CreateKeyReq {
    #[serde(default)]
    label: String,
    /// "percent" | "hardcap" (default percent=100 → full allocation).
    #[serde(default)]
    limit_kind: Option<String>,
    #[serde(default)]
    limit_value: Option<i64>,
}

async fn create_key(
    State(state): State<AppState>,
    Extension(user): Extension<AuthUser>,
    Json(req): Json<CreateKeyReq>,
) -> WebResult<Response> {
    let acct = account_id_for(&state, user.0).await?;
    let limit_kind = match req.limit_kind.as_deref() {
        Some("hardcap") => "hardcap",
        _ => "percent",
    };
    let limit_value = req.limit_value.unwrap_or(100).max(0);
    let (raw, prefix) = generate_api_key();
    let id: Uuid = sqlx::query(
        "INSERT INTO api_keys (account_id, key_hash, key_prefix, label, limit_kind, limit_value) \
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
    )
    .bind(acct)
    .bind(sha256(&raw))
    .bind(&prefix)
    .bind(&req.label)
    .bind(limit_kind)
    .bind(limit_value)
    .fetch_one(&state.pool)
    .await?
    .get("id");
    // The raw key is shown exactly once.
    Ok((
        StatusCode::CREATED,
        Json(json!({
            "id": id.to_string(),
            "key": raw,
            "prefix": prefix,
            "limit_kind": limit_kind,
            "limit_value": limit_value,
        })),
    )
        .into_response())
}

async fn archive_key(
    State(state): State<AppState>,
    Extension(user): Extension<AuthUser>,
    Path(id): Path<Uuid>,
) -> WebResult<Response> {
    let acct = account_id_for(&state, user.0).await?;
    let res = sqlx::query(
        "UPDATE api_keys SET status = 'archived' WHERE id = $1 AND account_id = $2 AND status = 'active'",
    )
    .bind(id)
    .bind(acct)
    .execute(&state.pool)
    .await?;
    if res.rows_affected() == 0 {
        return Err(WebError::BadRequest("no such active key"));
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

#[derive(Deserialize)]
struct UpdateLimitReq {
    limit_kind: String,
    limit_value: i64,
}

async fn update_key_limit(
    State(state): State<AppState>,
    Extension(user): Extension<AuthUser>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateLimitReq>,
) -> WebResult<Response> {
    if req.limit_kind != "percent" && req.limit_kind != "hardcap" {
        return Err(WebError::BadRequest(
            "limit_kind must be percent or hardcap",
        ));
    }
    if req.limit_value < 0 {
        return Err(WebError::BadRequest("limit_value must be >= 0"));
    }
    let acct = account_id_for(&state, user.0).await?;
    let res = sqlx::query(
        "UPDATE api_keys SET limit_kind = $1, limit_value = $2 WHERE id = $3 AND account_id = $4",
    )
    .bind(&req.limit_kind)
    .bind(req.limit_value)
    .bind(id)
    .bind(acct)
    .execute(&state.pool)
    .await?;
    if res.rows_affected() == 0 {
        return Err(WebError::BadRequest("no such key"));
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

#[derive(Deserialize)]
struct RedeemReq {
    code: String,
}

/// `POST /web/v1/redeem` — redeem a single-use top-up code, raising the
/// account's allocation. Returns the new total. Generic 400 for an invalid
/// or already-redeemed code (no oracle).
async fn redeem(
    State(state): State<AppState>,
    Extension(AuthAccount(acct)): Extension<AuthAccount>,
    Json(req): Json<RedeemReq>,
) -> WebResult<Response> {
    match crate::topup::redeem(&state.pool, acct, &req.code).await {
        Ok(new_total) => Ok(Json(json!({ "allocation_total": new_total })).into_response()),
        Err(crate::topup::TopUpError::Invalid) => {
            Err(WebError::BadRequest("invalid or already-redeemed code"))
        }
        Err(crate::topup::TopUpError::Db(e)) => {
            tracing::error!(error = %e, "redeem db error");
            Err(WebError::Internal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AuthSettings;

    #[test]
    fn signup_origin_must_be_allowlisted() {
        // The resolved origin is embedded in an email we send and DKIM-sign.
        // If a caller could supply any value, /register would become a
        // phishing-link generator carrying our domain's reputation.
        let cfg = AuthSettings {
            app_base_url: "https://helexa.ai".into(),
            additional_app_origins: vec!["https://angels.helexa.ai".into()],
            ..Default::default()
        };
        let origin = |req: Option<&str>| resolve_origin_with(&cfg, req);

        assert_eq!(origin(None), "https://helexa.ai");
        assert_eq!(
            origin(Some("https://angels.helexa.ai")),
            "https://angels.helexa.ai"
        );
        // A trailing slash is the same origin, not a bypass.
        assert_eq!(
            origin(Some("https://angels.helexa.ai/")),
            "https://angels.helexa.ai"
        );
        // Everything else falls back rather than being echoed into a mail.
        for hostile in [
            "https://evil.example",
            "https://angels.helexa.ai.evil.example",
            "https://helexa.ai.evil.example",
            "javascript:alert(1)",
            "//evil.example",
        ] {
            assert_eq!(
                origin(Some(hostile)),
                "https://helexa.ai",
                "leaked: {hostile}"
            );
        }
    }
}
