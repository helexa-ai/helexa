//! Outbound TLS pinning tests for #74.
//!
//! Proves the router, as a TLS client to cortexes, reaches a cortex
//! presenting its **enrolled** cert and rejects one presenting an
//! unexpected (or untrusted) cert — and that a rejected handshake flows
//! through the existing reachability path (#72) to exclude the cortex.
//!
//! A minimal `tokio-rustls` HTTPS server presents a self-signed cert; the
//! router's `reqwest` client (native-tls) validates against the PEM anchor
//! enrolled in config. Server (rustls) and client (native-tls) interoperate
//! at the protocol level — what matters is the trust decision.

use helexa_router::config::{CortexEndpoint, RouterConfig};
use helexa_router::poller::poll_once;
use helexa_router::state::{RouterState, build_client};
use std::io::Write;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// A self-signed cert: PEM (for the reqwest pin file) + DER cert/key (for
/// the rustls server).
struct TestCert {
    cert_pem: String,
    cert_der: rustls::pki_types::CertificateDer<'static>,
    key_der: Vec<u8>,
}

fn make_cert() -> TestCert {
    let key = rcgen::generate_simple_self_signed(vec!["127.0.0.1".to_string()]).unwrap();
    TestCert {
        cert_pem: key.cert.pem(),
        cert_der: key.cert.der().clone(),
        key_der: key.key_pair.serialize_der(),
    }
}

/// Write a cert PEM to a unique temp file (named by `tag`) and return the
/// path. `tag` is caller-unique (we use the bound port), so no randomness.
fn write_pem(tag: &str, pem: &str) -> String {
    let path = std::env::temp_dir().join(format!("helexa-router-tls-{tag}.pem"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(pem.as_bytes()).unwrap();
    path.to_string_lossy().into_owned()
}

/// Spawn a minimal HTTPS server presenting `cert`, answering every request
/// with a canned `/v1/models`-shaped 200. Returns its `https://` base URL.
async fn spawn_https(cert: &TestCert) -> String {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let key = rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
        cert.key_der.clone(),
    ));
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert_der.clone()], key)
        .unwrap();
    let acceptor = TlsAcceptor::from(Arc::new(config));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                continue;
            };
            let acceptor = acceptor.clone();
            tokio::spawn(async move {
                if let Ok(mut tls) = acceptor.accept(stream).await {
                    let mut buf = [0u8; 2048];
                    let _ = tls.read(&mut buf).await; // consume request line/headers
                    let body = "{\"object\":\"list\",\"data\":[]}";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = tls.write_all(resp.as_bytes()).await;
                    let _ = tls.shutdown().await;
                }
            });
        }
    });
    format!("https://{addr}")
}

fn tag_for(url: &str) -> String {
    url.rsplit(':').next().unwrap_or("0").to_string()
}

#[tokio::test]
async fn pinned_client_accepts_enrolled_cert_and_rejects_others() {
    let server_cert = make_cert();
    let other_cert = make_cert();
    let url = spawn_https(&server_cert).await;
    let tag = tag_for(&url);

    let good_pin = write_pem(&format!("{tag}-good"), &server_cert.cert_pem);
    let bad_pin = write_pem(&format!("{tag}-bad"), &other_cert.cert_pem);

    // Enrolled with the server's own cert → handshake trusted → 200.
    let good = build_client(Some(&good_pin)).unwrap();
    let resp = good.get(format!("{url}/v1/models")).send().await;
    assert!(resp.is_ok(), "enrolled cert must be accepted: {resp:?}");
    assert_eq!(resp.unwrap().status(), 200);

    // Enrolled with a different cert → server's cert is unexpected → reject.
    let bad = build_client(Some(&bad_pin)).unwrap();
    assert!(
        bad.get(format!("{url}/v1/models")).send().await.is_err(),
        "unexpected cert must be rejected"
    );

    // No enrollment (default platform roots) → self-signed cert untrusted.
    let default = build_client(None).unwrap();
    assert!(
        default
            .get(format!("{url}/v1/models"))
            .send()
            .await
            .is_err(),
        "un-enrolled self-signed cert must be rejected by default roots"
    );
}

#[tokio::test]
async fn poller_excludes_cortex_with_unexpected_cert() {
    let server_cert = make_cert();
    let other_cert = make_cert();
    let url = spawn_https(&server_cert).await;
    let tag = tag_for(&url);

    let good_pin = write_pem(&format!("{tag}-pgood"), &server_cert.cert_pem);
    let bad_pin = write_pem(&format!("{tag}-pbad"), &other_cert.cert_pem);

    // Cortex A enrolled correctly → reachable. Cortex B enrolled with the
    // wrong cert → TLS handshake fails → excluded.
    let cfg = RouterConfig {
        cortexes: vec![
            CortexEndpoint {
                name: "good".into(),
                endpoint: url.clone(),
                region: None,
                tls_ca: Some(good_pin),
            },
            CortexEndpoint {
                name: "bad".into(),
                endpoint: url.clone(),
                region: None,
                tls_ca: Some(bad_pin),
            },
        ],
        ..Default::default()
    };
    let state = RouterState::from_config(&cfg);
    poll_once(&state).await;

    let topo = state.topology.read().await;
    assert!(
        topo["good"].reachable,
        "correctly-enrolled cortex reachable"
    );
    assert!(
        !topo["bad"].reachable,
        "cortex presenting an unexpected cert is excluded"
    );
}

#[tokio::test]
async fn misconfigured_pin_disables_cortex_fail_closed() {
    // A `tls_ca` pointing at a nonexistent file must NOT fall back to an
    // unpinned client — the cortex is disabled entirely.
    let cfg = RouterConfig {
        cortexes: vec![
            CortexEndpoint {
                name: "broken".into(),
                endpoint: "https://127.0.0.1:1".into(),
                region: None,
                tls_ca: Some("/no/such/anchor.pem".into()),
            },
            CortexEndpoint {
                name: "plain".into(),
                endpoint: "http://127.0.0.1:1".into(),
                region: None,
                tls_ca: None,
            },
        ],
        ..Default::default()
    };
    let state = RouterState::from_config(&cfg);
    assert!(
        state.client_for("broken").is_none(),
        "a cortex with an unloadable pin is disabled (fail closed)"
    );
    assert!(
        state.client_for("plain").is_some(),
        "an un-pinned cortex still gets a client"
    );
}

#[test]
fn build_client_rejects_garbage_pem() {
    let path = write_pem(
        "garbage",
        "-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----",
    );
    assert!(build_client(Some(&path)).is_err());
}
