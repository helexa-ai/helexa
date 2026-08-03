//! Server-side rendering.
//!
//! Templates are compiled into the binary with `include_str!` rather than
//! read from `/usr/share`: they are application chrome, not content, so
//! embedding them means the RPM ships one file and a template can never
//! drift out of sync with the code that fills it. Round *content* is a
//! different matter and does live on disk — see [`crate::content`].

use axum::http::StatusCode;
use minijinja::{Environment, context};
use std::sync::LazyLock;

static ENV: LazyLock<Environment<'static>> = LazyLock::new(|| {
    let mut env = Environment::new();
    // Auto-escape everything. Round content is the one place we render
    // pre-sanitised HTML, and it goes through an explicit `|safe`.
    env.set_auto_escape_callback(|_| minijinja::AutoEscape::Html);
    env.add_template("base.html", include_str!("../templates/base.html"))
        .expect("base.html is embedded and must compile");
    env.add_template("signin.html", include_str!("../templates/signin.html"))
        .expect("signin.html is embedded and must compile");
    env.add_template("error.html", include_str!("../templates/error.html"))
        .expect("error.html is embedded and must compile");
    env.add_template("round.html", include_str!("../templates/round.html"))
        .expect("round.html is embedded and must compile");
    env.add_template("document.html", include_str!("../templates/document.html"))
        .expect("document.html is embedded and must compile");
    env.add_template("portal.html", include_str!("../templates/portal.html"))
        .expect("portal.html is embedded and must compile");
    env.add_template("account.html", include_str!("../templates/account.html"))
        .expect("account.html is embedded and must compile");
    env.add_template("privacy.html", include_str!("../templates/privacy.html"))
        .expect("privacy.html is embedded and must compile");
    env
});

/// Render a named template with the given context.
pub fn render(name: &str, ctx: minijinja::value::Value) -> Result<String, minijinja::Error> {
    ENV.get_template(name)?.render(ctx)
}

/// The error page. Infallible by construction — an error while rendering
/// the error page would otherwise recurse, so this falls back to plain
/// text rather than propagating.
pub fn render_error(code: StatusCode, message: &str) -> String {
    ENV.get_template("error.html")
        .and_then(|t| t.render(context! { code => code.as_u16(), message => message }))
        .unwrap_or_else(|_| format!("<!doctype html><title>{code}</title><p>{message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_template_compiles() {
        // Touching ENV forces the LazyLock, which panics on a bad
        // template. This is the test that catches a syntax error before
        // it reaches a request path.
        let _ = ENV.get_template("base.html").expect("base");
        let _ = ENV.get_template("signin.html").expect("signin");
        let _ = ENV.get_template("round.html").expect("round");
        let _ = ENV.get_template("document.html").expect("document");
        let _ = ENV.get_template("portal.html").expect("portal");
        let _ = ENV.get_template("account.html").expect("account");
        let _ = ENV.get_template("privacy.html").expect("privacy");
    }

    #[test]
    fn error_page_renders_and_escapes() {
        let html = render_error(StatusCode::NOT_FOUND, "nothing here");
        assert!(html.contains("404"));
        assert!(html.contains("nothing here"));
        assert!(html.contains("noindex"));
    }

    #[test]
    fn error_page_escapes_hostile_input() {
        let html = render_error(StatusCode::BAD_REQUEST, "<script>alert(1)</script>");
        assert!(
            !html.contains("<script>alert"),
            "error message was not escaped: {html}"
        );
    }
}
