//! Audio job lookups through the router (#382): a job id carries no hint of
//! its cortex, so job calls are asked of each cortex until one answers with
//! something other than 404, and listings merge every cortex's jobs.

use axum::extract::{Path, RawQuery};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::state::RouterState;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::net::TcpListener;

/// A cortex that holds exactly one job, `job_id`, created at `created`.
async fn spawn_cortex(job_id: &'static str, created: u64) -> String {
    let job = move || {
        json!({"id": job_id, "object": "audio.generation", "status": "succeeded",
               "model": "m", "created_at": created})
    };
    let app = Router::new()
        .route(
            "/v1/audio/generations",
            get(move |headers: HeaderMap| async move {
                // Only the bearer's own jobs are listed.
                if headers.get("authorization").is_none() {
                    return Json(json!({"object": "list", "data": []})).into_response();
                }
                Json(json!({"object": "list", "data": [job()]})).into_response()
            }),
        )
        .route(
            "/v1/audio/generations/{id}",
            get(move |Path(id): Path<String>| async move {
                if id == job_id {
                    Json(job()).into_response()
                } else {
                    not_found()
                }
            })
            .delete(move |Path(id): Path<String>| async move {
                if id == job_id {
                    let mut j = job();
                    j["status"] = json!("cancelled");
                    Json(j).into_response()
                } else {
                    not_found()
                }
            }),
        )
        .route(
            "/v1/audio/generations/{id}/content",
            get(
                move |Path(id): Path<String>, RawQuery(q): RawQuery| async move {
                    if id == job_id {
                        // Echo the query so the test can see it was forwarded.
                        let (ct, suffix) = match q.as_deref() {
                            Some("format=mp3") => ("audio/mpeg", ".mp3"),
                            _ => ("audio/flac", ""),
                        };
                        (
                            StatusCode::OK,
                            [("content-type", ct)],
                            format!("audio-of-{job_id}{suffix}"),
                        )
                            .into_response()
                    } else {
                        not_found()
                    }
                },
            ),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({"error": {"type": "invalid_request_error", "code": "job_not_found", "message": "no"}})),
    )
        .into_response()
}

async fn router_over(cortexes: Vec<(&str, String)>) -> String {
    let cfg = RouterConfig {
        cortexes: cortexes
            .into_iter()
            .map(|(name, endpoint)| CortexEndpoint {
                name: name.into(),
                endpoint,
                region: None,
                tls_ca: None,
            })
            .collect(),
        ..Default::default()
    };
    let state = Arc::new(RouterState::from_config(&cfg));
    let app = helexa_router::handlers::api_routes().with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

#[tokio::test]
async fn job_calls_find_the_cortex_that_holds_the_job() {
    let a = spawn_cortex("audiogen_a", 100).await;
    let b = spawn_cortex("audiogen_b", 200).await;
    // An unreachable cortex in the list must be skipped, not fatal.
    let router = router_over(vec![
        ("dead", "http://127.0.0.1:1".into()),
        ("a", a),
        ("b", b),
    ])
    .await;
    let http = reqwest::Client::new();

    let job: Value = http
        .get(format!("{router}/v1/audio/generations/audiogen_b"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(job["id"], "audiogen_b");

    let content = http
        .get(format!("{router}/v1/audio/generations/audiogen_b/content"))
        .send()
        .await
        .unwrap();
    assert_eq!(content.status(), 200);
    assert_eq!(content.headers()["content-type"], "audio/flac");
    assert_eq!(content.text().await.unwrap(), "audio-of-audiogen_b");

    // The download format in the query reaches the cortex.
    let mp3 = http
        .get(format!(
            "{router}/v1/audio/generations/audiogen_b/content?format=mp3"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(mp3.headers()["content-type"], "audio/mpeg");
    assert_eq!(mp3.text().await.unwrap(), "audio-of-audiogen_b.mp3");

    let cancelled: Value = http
        .delete(format!("{router}/v1/audio/generations/audiogen_a"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(cancelled["status"], "cancelled");

    let missing = http
        .get(format!("{router}/v1/audio/generations/audiogen_nope"))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 404);
    let body: Value = missing.json().await.unwrap();
    assert_eq!(body["error"]["code"], "job_not_found");
}

#[tokio::test]
async fn listing_merges_cortexes_newest_first_and_forwards_the_bearer() {
    let a = spawn_cortex("audiogen_a", 100).await;
    let b = spawn_cortex("audiogen_b", 200).await;
    let router = router_over(vec![("a", a), ("b", b)]).await;
    let http = reqwest::Client::new();

    let list: Value = http
        .get(format!("{router}/v1/audio/generations"))
        .bearer_auth("k")
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = list["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| j["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["audiogen_b", "audiogen_a"]);

    let anon: Value = http
        .get(format!("{router}/v1/audio/generations"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(anon["data"].as_array().unwrap().len(), 0);
}
