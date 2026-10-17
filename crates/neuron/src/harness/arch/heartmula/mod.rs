//! HeartMuLa text-to-song (#379): the MuLa language model ([`lm`], #391),
//! which turns tags and lyrics into 12.5 Hz frames of eight RVQ codes, and
//! the HeartCodec decoder ([`codec`], #392), which turns those codes into
//! 48 kHz stereo audio.
//!
//! Ported from heartlib @ 5858ca8f (Apache-2.0). The MuQ-MuLan reference-
//! audio encoder (CC-BY-NC) is never used: text-to-song conditions on a
//! zero reference embedding, as heartlib does (#389).
//!
//! Parity with heartlib on the CPU in f32 is checked by the ignored tests
//! in `tests.rs` against the #390 traces in `testdata/heartmula`.

pub mod codec;
pub mod lm;

#[cfg(test)]
mod tests;
