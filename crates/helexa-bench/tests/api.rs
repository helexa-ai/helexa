//! Read-API tests: seed a temp store, serve the router, assert JSON.

use helexa_bench::api;
use helexa_bench::store::{RunRecord, Store};
use serde_json::Value;

#[allow(clippy::too_many_arguments)]
fn rec(
    host: &str,
    sha: &str,
    build_ts: Option<&str>,
    model: &str,
    scenario: &str,
    ttft: f64,
    ok: bool,
) -> RunRecord {
    RunRecord {
        ts: "2026-06-13T00:00:00Z".into(),
        target_name: host.into(),
        target_kind: "neuron".into(),
        endpoint: format!("http://{host}:13131"),
        hostname: Some(host.into()),
        driver_version: Some("580.159".into()),
        cuda_version: Some("13.0".into()),
        gpus_json: Some("[]".into()),
        git_sha: sha.into(),
        git_sha_long: None,
        package_version: "0.1.16".into(),
        git_dirty: false,
        build_timestamp: build_ts.map(|s| s.to_string()),
        rustc_version: None,
        profile: Some("release".into()),
        features_json: "[\"cuda\"]".into(),
        candle_version: Some("0.10.2".into()),
        bench_version: "0.1.16".into(),
        bench_sha: "deadbee".into(),
        model_id: model.into(),
        harness: "candle".into(),
        capabilities_json: "[\"text\"]".into(),
        devices_json: "[0]".into(),
        scenario_id: scenario.into(),
        prompt_size_approx: 128,
        prompt_tokens_actual: Some(130),
        max_tokens: 64,
        ttft_s: if ok { Some(ttft) } else { None },
        decode_tps: if ok { Some(30.0) } else { None },
        total_s: if ok { Some(2.0) } else { None },
        completion_tokens: if ok { Some(60) } else { None },
        prefill_ms: if ok { Some(150) } else { None },
        decode_ms: if ok { Some(1800) } else { None },
        prefill_tokens: if ok { Some(130) } else { None },
        ok,
        error: if ok { None } else { Some("boom".into()) },
    }
}

/// Seed a temp db, return its path.
fn seed(tag: &str) -> String {
    let path = std::env::temp_dir().join(format!("hb-api-{}-{tag}.sqlite", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let p = path.to_string_lossy().to_string();
    let store = Store::open(&p).unwrap();
    // beast / m / chat:128 across two builds (old then new).
    store
        .insert_run(&rec(
            "beast",
            "old",
            Some("2026-06-01T00:00:00Z"),
            "m",
            "chat:128",
            0.20,
            true,
        ))
        .unwrap();
    store
        .insert_run(&rec(
            "beast",
            "new",
            Some("2026-06-10T00:00:00Z"),
            "m",
            "chat:128",
            0.10,
            true,
        ))
        .unwrap();
    store
        .insert_run(&rec(
            "beast",
            "new",
            Some("2026-06-10T00:00:00Z"),
            "m",
            "chat:128",
            0.12,
            true,
        ))
        .unwrap();
    // a failed row (must not count in series/summary medians)
    store
        .insert_run(&rec(
            "beast",
            "new",
            Some("2026-06-10T00:00:00Z"),
            "m",
            "chat:128",
            0.0,
            false,
        ))
        .unwrap();
    // a different host for the runs filter
    store
        .insert_run(&rec(
            "benjy",
            "new",
            Some("2026-06-10T00:00:00Z"),
            "n",
            "chat:128",
            0.15,
            true,
        ))
        .unwrap();
    p
}

async fn spawn(db: &str) -> String {
    let state = api::open_state(db).unwrap();
    let app = api::api_routes(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn get(base: &str, path: &str) -> Value {
    reqwest::get(format!("{base}{path}"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

#[tokio::test]
async fn health_reports_run_count() {
    let base = spawn(&seed("health")).await;
    let v = get(&base, "/api/health").await;
    assert_eq!(v["status"], "ok");
    assert_eq!(v["run_count"], 5);
}

#[tokio::test]
async fn dimensions_lists_distinct_values_and_builds_chronologically() {
    let base = spawn(&seed("dims")).await;
    let v = get(&base, "/api/dimensions").await;
    let hosts: Vec<&str> = v["hosts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap())
        .collect();
    assert_eq!(hosts, vec!["beast", "benjy"]);
    assert_eq!(v["models"].as_array().unwrap().len(), 2);
    // builds ordered by earliest build_timestamp: old before new
    let builds = v["builds"].as_array().unwrap();
    assert_eq!(builds[0]["git_sha"], "old");
    assert_eq!(builds[1]["git_sha"], "new");
}

#[tokio::test]
async fn summary_uses_latest_sha_and_ignores_failures() {
    let base = spawn(&seed("summary")).await;
    let v = get(&base, "/api/summary").await;
    let rows = v.as_array().unwrap();
    let beast = rows
        .iter()
        .find(|r| r["target_name"] == "beast" && r["scenario_id"] == "chat:128")
        .unwrap();
    assert_eq!(beast["git_sha"], "new");
    assert_eq!(beast["samples"], 2); // two ok rows on "new"; failure excluded
    // median of 0.10 and 0.12
    assert!((beast["ttft_s_median"].as_f64().unwrap() - 0.11).abs() < 1e-9);
}

#[tokio::test]
async fn series_is_chronological_per_build() {
    let base = spawn(&seed("series")).await;
    let v = get(&base, "/api/series?host=beast&model=m&scenario=chat:128").await;
    let pts = v.as_array().unwrap();
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0]["git_sha"], "old");
    assert_eq!(pts[1]["git_sha"], "new");
    assert_eq!(pts[0]["samples"], 1);
    assert_eq!(pts[1]["samples"], 2);
}

#[tokio::test]
async fn series_resolves_host_when_omitted() {
    // The public UI selects by model alone; the store resolves the host.
    let base = spawn(&seed("series-nohost")).await;
    let v = get(&base, "/api/series?model=m&scenario=chat:128").await;
    let pts = v.as_array().unwrap();
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0]["git_sha"], "old");
    assert_eq!(pts[1]["git_sha"], "new");
}

#[tokio::test]
async fn runs_filters_by_host() {
    let base = spawn(&seed("runs")).await;
    let all = get(&base, "/api/runs").await;
    assert_eq!(all.as_array().unwrap().len(), 5);
    let beast = get(&base, "/api/runs?host=beast").await;
    let rows = beast.as_array().unwrap();
    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r["host"] == "beast"));
    // failed row carries its error + ok=false
    assert!(
        rows.iter()
            .any(|r| r["ok"] == false && r["error"] == "boom")
    );
}
