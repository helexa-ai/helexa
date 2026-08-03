//! Error type and its HTTP rendering.
//!
//! The portal renders HTML, not JSON: an error here is seen by a person,
//! so it becomes a page. Two rules shape the mapping —
//!
//! 1. **Never confirm existence to an unauthenticated caller.** A bad
//!    invite code, a code for a round that does not exist, and a revoked
//!    code all render identically. Otherwise the portal becomes an oracle
//!    for enumerating rounds and codes.
//! 2. **Never leak internals.** Database and template failures render a
//!    generic page; the detail goes to the log.

use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};

#[derive(Debug, thiserror::Error)]
pub enum AngelsError {
    #[error("not found")]
    NotFound,

    /// Authenticated, but not entitled to this round. Distinct from
    /// `NotFound` internally so the access log can record a denial, but
    /// it renders as an ordinary "no access" page.
    #[error("no access to this round")]
    Forbidden,

    /// Not signed in. Triggers a redirect to the sign-in page.
    #[error("authentication required")]
    Unauthenticated,

    #[error("invalid credentials")]
    BadCredentials,

    #[error("{0}")]
    BadRequest(String),

    #[error(transparent)]
    Db(#[from] sqlx::Error),

    #[error(transparent)]
    Template(#[from] minijinja::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AngelsError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unauthenticated => StatusCode::UNAUTHORIZED,
            Self::BadCredentials => StatusCode::UNAUTHORIZED,
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Db(_) | Self::Template(_) | Self::Other(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// What the visitor is told. Internal failures are deliberately vague.
    pub fn public_message(&self) -> String {
        match self {
            Self::NotFound => "That link doesn't lead anywhere.".into(),
            Self::Forbidden => "This material isn't available to your account.".into(),
            Self::Unauthenticated => "Please sign in to continue.".into(),
            Self::BadCredentials => "That email address and password don't match.".into(),
            Self::BadRequest(m) => m.clone(),
            Self::Db(_) | Self::Template(_) | Self::Other(_) => {
                "Something went wrong at our end. It has been logged.".into()
            }
        }
    }
}

impl IntoResponse for AngelsError {
    fn into_response(self) -> Response {
        // Log the real cause for the 5xx family; the visitor never sees it.
        if self.status() == StatusCode::INTERNAL_SERVER_ERROR {
            tracing::error!(error = %self, "angels request failed");
        }
        let body = crate::templates::render_error(self.status(), &self.public_message());
        (self.status(), Html(body)).into_response()
    }
}

pub type Result<T> = std::result::Result<T, AngelsError>;
