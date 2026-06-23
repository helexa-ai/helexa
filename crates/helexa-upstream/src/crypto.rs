//! Hashing + secret-generation helpers.
//!
//! - **Passwords** (low-entropy) → argon2id PHC strings.
//! - **API keys / top-up codes / email + session tokens** (high-entropy
//!   secrets minted here) → stored only as their sha256; sha256 is the fast,
//!   sufficient choice for high-entropy material.

use argon2::Argon2;
use argon2::password_hash::rand_core::OsRng as ArgonOsRng;
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use rand::RngCore;
use sha2::{Digest, Sha256};

/// sha256 of `input`, as raw bytes (matches the `BYTEA` columns).
pub fn sha256(input: &str) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    h.finalize().to_vec()
}

/// Hash a password with argon2id, returning a PHC string for storage.
pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut ArgonOsRng);
    Ok(Argon2::default()
        .hash_password(password.as_bytes(), &salt)?
        .to_string())
}

/// Verify a password against a stored PHC hash. `false` on any mismatch or
/// malformed hash (never panics).
pub fn verify_password(password: &str, phc: &str) -> bool {
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// A fresh URL-safe high-entropy secret (256 bits) for email/session/reset
/// tokens. The caller stores only `sha256` of this and emails/returns the
/// raw value.
pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base62(&bytes)
}

/// Mint a new API key: `(raw, prefix)`. `raw` is shown to the user once;
/// only `sha256(raw)` is stored. The prefix is a non-secret display tag.
pub fn generate_api_key() -> (String, String) {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let raw = format!("sk-helexa-{}", base62(&bytes));
    // Non-secret prefix for the dashboard list (scheme + first few chars).
    let prefix: String = raw.chars().take(14).collect();
    (raw, prefix)
}

/// base62 encode (0-9A-Za-z) — URL/clipboard friendly, no padding.
fn base62(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    // Treat the bytes as a big-endian integer and base62 it. 32 bytes → ~43
    // chars. Simple repeated-division over a big-uint built from the bytes.
    let mut digits: Vec<u8> = vec![0];
    for &byte in bytes {
        let mut carry = byte as u32;
        for d in digits.iter_mut() {
            let v = (*d as u32) * 256 + carry;
            *d = (v % 62) as u8;
            carry = v / 62;
        }
        while carry > 0 {
            digits.push((carry % 62) as u8);
            carry /= 62;
        }
    }
    digits
        .iter()
        .rev()
        .map(|&d| ALPHABET[d as usize] as char)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_round_trips_and_rejects_wrong() {
        let phc = hash_password("correct horse").unwrap();
        assert!(verify_password("correct horse", &phc));
        assert!(!verify_password("wrong", &phc));
        assert!(!verify_password("correct horse", "not-a-phc-string"));
    }

    #[test]
    fn api_key_has_scheme_prefix_and_unique_body() {
        let (raw, prefix) = generate_api_key();
        assert!(raw.starts_with("sk-helexa-"));
        assert!(prefix.starts_with("sk-helexa-"));
        let (raw2, _) = generate_api_key();
        assert_ne!(raw, raw2, "keys are unique");
    }

    #[test]
    fn random_tokens_are_unique_and_nonempty() {
        let a = random_token();
        let b = random_token();
        assert!(!a.is_empty());
        assert_ne!(a, b);
    }
}
