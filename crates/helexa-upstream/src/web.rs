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
    let protected = Router::new()
        .route("/web/v1/account", get(account))
        .route("/web/v1/keys", get(list_keys).post(create_key))
        .route("/web/v1/keys/{id}/archive", post(archive_key))
        .route(
            "/web/v1/keys/{id}/limit",
            axum::routing::patch(update_key_limit),
        )
        .route("/web/v1/redeem", post(redeem))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            require_session,
        ));

    Router::new()
        .route("/web/v1/register", post(register))
        .route("/web/v1/verify", post(verify))
        .route("/web/v1/login", post(login))
        .route("/web/v1/password-reset/request", post(reset_request))
        .route("/web/v1/password-reset/confirm", post(reset_confirm))
        .merge(protected)
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

/// The caller's single account id.
async fn account_id_for(state: &AppState, user_id: Uuid) -> WebResult<Uuid> {
    let row = sqlx::query("SELECT id FROM accounts WHERE owner_user_id = $1")
        .bind(user_id)
        .fetch_optional(&state.pool)
        .await?;
    row.map(|r| r.get::<Uuid, _>("id"))
        .ok_or(WebError::Internal)
}

// ── auth lifecycle ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct RegisterReq {
    email: String,
    password: String,
    #[serde(default)]
    fingerprint: Option<String>,
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
        return Ok(()); // email already registered — say nothing
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

    let link = format!("{}/verify?token={token}", state.config.auth.app_base_url);
    let _ = state
        .email
        .send(
            &req.email,
            "Verify your helexa account",
            &format!("Welcome to helexa. Verify your email:\n\n{link}\n"),
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
            &format!("Reset your password:\n\n{link}\n"),
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
    Ok(Json(json!({
        "account_id": acct.to_string(),
        "allocation_total": row.get::<i64, _>("allocation_total"),
        "allocation_spent": row.get::<i64, _>("allocation_spent"),
        "allocation_reserved": row.get::<i64, _>("allocation_reserved"),
    }))
    .into_response())
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
    Extension(user): Extension<AuthUser>,
    Json(req): Json<RedeemReq>,
) -> WebResult<Response> {
    let acct = account_id_for(&state, user.0).await?;
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
