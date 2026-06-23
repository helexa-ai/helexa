//! Hashing helpers. API keys and top-up codes are stored only as their
//! sha256 (they are high-entropy secrets; sha256 is the fast, sufficient
//! choice — argon2 is reserved for low-entropy passwords).

use sha2::{Digest, Sha256};

/// sha256 of `input`, as raw bytes (matches the `BYTEA` columns).
pub fn sha256(input: &str) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    h.finalize().to_vec()
}
