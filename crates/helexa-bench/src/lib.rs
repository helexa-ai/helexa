//! helexa-bench — a continuous, version-aware benchmark harness for the
//! neuron fleet. It hits each neuron directly, exercises an extensible
//! scenario suite against every warm model, and records each run with
//! full build/version provenance into SQLite so improvements can be
//! tracked automatically across neuron implementation updates.

pub mod api;
pub mod client;
pub mod config;
pub mod report;
pub mod scenario;
pub mod store;
pub mod sweep;
