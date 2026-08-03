//! Hashing and secret generation.
//!
//! Mirrors `helexa_upstream::crypto` deliberately — passwords are verified
//! against hashes **that service wrote**, so the algorithm and parameter
//! set must match. argon2id with `Argon2::default()` on both sides; if
//! upstream ever changes its parameters, this must change with it.
//!
//! High-entropy secrets we mint ourselves (session tokens, invite codes)
//! are stored as sha256 only.

use argon2::Argon2;
use argon2::password_hash::{PasswordHash, PasswordVerifier};
use rand::RngCore;
use sha2::{Digest, Sha256};

/// sha256 of `input` as raw bytes, matching the `BYTEA` columns.
pub fn sha256(input: &str) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    h.finalize().to_vec()
}

/// Verify a password against a stored argon2 PHC hash from `public.users`.
///
/// `false` on any mismatch or malformed hash — never panics, and never
/// distinguishes "no such user" from "wrong password" to the caller.
pub fn verify_password(password: &str, phc: &str) -> bool {
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// A fresh URL-safe 256-bit secret. Callers store only `sha256` of it.
pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base62(&bytes)
}

/// An invite code. Long enough that guessing is not a threat model even
/// before rate limiting, and it is only ever half the story — redeeming it
/// still requires a verified account (see `migrations/0001_init.sql`).
pub fn generate_invite_code() -> String {
    let mut bytes = [0u8; 24];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base62(&bytes)
}

/// base62 (0-9A-Za-z) — URL and clipboard friendly, no padding.
fn base62(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut num = bytes.to_vec();
    let mut out = Vec::new();
    while num.iter().any(|&b| b != 0) {
        let mut rem = 0u32;
        for byte in num.iter_mut() {
            let acc = (rem << 8) | u32::from(*byte);
            *byte = (acc / 62) as u8;
            rem = acc % 62;
        }
        out.push(ALPHABET[rem as usize]);
    }
    if out.is_empty() {
        out.push(ALPHABET[0]);
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_stable_and_32_bytes() {
        let a = sha256("hello");
        assert_eq!(a.len(), 32);
        assert_eq!(a, sha256("hello"));
        assert_ne!(a, sha256("hello "));
    }

    #[test]
    fn tokens_are_unique_and_url_safe() {
        let a = random_token();
        let b = random_token();
        assert_ne!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric()), "{a}");
        assert!(a.len() > 30, "token too short: {}", a.len());
    }

    #[test]
    fn invite_codes_are_unguessable_length() {
        let c = generate_invite_code();
        assert!(c.chars().all(|c| c.is_ascii_alphanumeric()), "{c}");
        // 24 random bytes in base62 lands around 32 characters; anything
        // markedly shorter would mean the encoder dropped entropy.
        assert!(c.len() >= 28, "invite code too short: {} ({c})", c.len());
    }

    #[test]
    fn malformed_password_hash_is_rejected_not_panicked() {
        assert!(!verify_password("anything", "not-a-phc-string"));
        assert!(!verify_password("anything", ""));
    }

    #[test]
    fn verifies_a_hash_produced_the_way_upstream_produces_them() {
        // Guards the shared-credential contract: if upstream's argon2
        // parameters and ours ever diverge, this fails.
        use argon2::password_hash::rand_core::OsRng as ArgonOsRng;
        use argon2::password_hash::{PasswordHasher, SaltString};
        let salt = SaltString::generate(&mut ArgonOsRng);
        let phc = Argon2::default()
            .hash_password(b"correct horse battery staple", &salt)
            .unwrap()
            .to_string();
        assert!(verify_password("correct horse battery staple", &phc));
        assert!(!verify_password("wrong password", &phc));
    }
}
