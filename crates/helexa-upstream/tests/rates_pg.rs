//! `model_rates` (#386) against a real PostgreSQL. Gated on
//! `UPSTREAM_TEST_DATABASE_URL`, like the ledger tests: unset, every test
//! logs a skip and passes.

use helexa_upstream::db::connect_and_migrate;
use helexa_upstream::rates;
use sqlx::postgres::PgPool;

async fn pool_or_skip(test: &str) -> Option<PgPool> {
    let Ok(url) = std::env::var("UPSTREAM_TEST_DATABASE_URL") else {
        eprintln!("skipping {test}: UPSTREAM_TEST_DATABASE_URL not set");
        return None;
    };
    Some(
        connect_and_migrate(&url, 4)
            .await
            .expect("connect + migrate"),
    )
}

#[tokio::test]
async fn default_audio_rate_is_seeded_and_model_rows_are_listed() {
    let Some(pool) = pool_or_skip("default_audio_rate_is_seeded").await else {
        return;
    };
    let model = format!("test/model-{}", uuid::Uuid::new_v4());
    sqlx::query(
        "INSERT INTO model_rates (model_id, unit, tokens_per_unit) VALUES ($1, 'audio_second', 2500)",
    )
    .bind(&model)
    .execute(&pool)
    .await
    .unwrap();

    let all = rates::list(&pool).await.unwrap();
    assert!(
        all.iter()
            .any(|r| r.model_id == "*" && r.unit == "audio_second" && r.tokens_per_unit > 0),
        "seeded default missing: {all:?}"
    );
    assert!(
        all.iter()
            .any(|r| r.model_id == model && r.tokens_per_unit == 2500)
    );

    sqlx::query("DELETE FROM model_rates WHERE model_id = $1")
        .bind(&model)
        .execute(&pool)
        .await
        .unwrap();
}

#[tokio::test]
async fn rates_must_be_positive_and_of_a_known_unit() {
    let Some(pool) = pool_or_skip("rates_must_be_positive").await else {
        return;
    };
    let insert = |unit: &'static str, tokens: i64| {
        let pool = pool.clone();
        async move {
            sqlx::query(
                "INSERT INTO model_rates (model_id, unit, tokens_per_unit) VALUES ('test/bad', $1, $2)",
            )
            .bind(unit)
            .bind(tokens)
            .execute(&pool)
            .await
        }
    };
    assert!(insert("audio_second", 0).await.is_err());
    assert!(insert("audio_second", -5).await.is_err());
    assert!(insert("furlongs", 10).await.is_err());
}
