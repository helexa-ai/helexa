//! NCCL state held by both the worker process and the leader's pool.
//!
//! Split into its own module so the worker (`tp/worker.rs`) and the
//! leader (`tp/mod.rs`) share the same hex-encoding/decoding code and
//! the same shape of `Option<Comm>` state machine.
//!
//! When the `cuda` feature is off, `NcclState` is a zero-sized
//! placeholder that returns `Error{kind="cuda_feature_not_enabled"}`
//! from every operation. When it's on, the same struct holds the
//! actual `cudarc::nccl::Comm`.

use super::rpc::WorkerResponse;
use super::worker::WorkerConfig;

/// Encode bytes as lowercase hex. Used for ferrying NCCL `Id::internal()`
/// across the leader→worker RPC boundary inside a JSON string.
pub fn encode_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Decode lowercase-or-uppercase hex into bytes. Errors on odd length
/// or non-hex characters; the caller bubbles those up via the RPC's
/// `Error{kind="bad_request"}` variant.
pub fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string has odd length {}", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("bad hex byte at {i}: {e}"))
        })
        .collect()
}

#[cfg(not(feature = "cuda"))]
pub struct NcclState;

#[cfg(not(feature = "cuda"))]
impl Default for NcclState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(feature = "cuda"))]
impl NcclState {
    pub fn new() -> Self {
        Self
    }

    pub fn init(&mut self, _cfg: WorkerConfig, _comm_id_hex: &str) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "this neuron binary was built without --features cuda; \
                 NCCL Init requires CUDA"
                .into(),
        }
    }

    pub fn sanity_check(&mut self) -> WorkerResponse {
        WorkerResponse::Error {
            kind: "cuda_feature_not_enabled".into(),
            message: "NCCL sanity check requires --features cuda".into(),
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::driver::CudaContext;
    use cudarc::nccl::{Comm, Id, ReduceOp};
    use std::sync::Arc;

    /// Number of bytes in NCCL's unique-id type; matches `Id::internal()`'s
    /// `[c_char; 128]`. Wire-encoded as 256 lowercase hex chars.
    const NCCL_ID_BYTES: usize = 128;

    pub struct NcclState {
        comm: Option<Comm>,
        /// Held alongside the Comm so the device isn't dropped
        /// underneath the NCCL handle.
        #[allow(dead_code)]
        ctx: Option<Arc<CudaContext>>,
    }

    impl Default for NcclState {
        fn default() -> Self {
            Self::new()
        }
    }

    impl NcclState {
        pub fn new() -> Self {
            Self {
                comm: None,
                ctx: None,
            }
        }
    }

    // SAFETY: `cudarc::nccl::Comm` contains a raw `ncclComm_t` pointer
    // (libnccl-allocated state). NCCL requires that operations against
    // one Comm be issued one at a time; we serialise access by storing
    // NcclState behind a Mutex in `WorkerPool`. The Comm itself is
    // move-safe — NCCL doesn't track the calling OS thread, only the
    // stream the operations are dispatched against.
    unsafe impl Send for NcclState {}
    unsafe impl Sync for NcclState {}

    /// Generate a fresh NCCL `Id` and return it hex-encoded. Used by
    /// the leader to mint the shared communicator id which is then
    /// broadcast to every worker via the RPC `Init` message.
    pub fn generate_comm_id_hex() -> Result<String, String> {
        let id = Id::new().map_err(|e| format!("Id::new(): {e}"))?;
        let bytes_u8: [u8; NCCL_ID_BYTES] = std::array::from_fn(|i| id.internal()[i] as u8);
        Ok(encode_hex(&bytes_u8))
    }

    impl NcclState {
        pub fn init(&mut self, cfg: WorkerConfig, comm_id_hex: &str) -> WorkerResponse {
            match try_init(self, cfg, comm_id_hex) {
                Ok(()) => WorkerResponse::InitOk,
                Err(msg) => WorkerResponse::Error {
                    kind: "nccl_init_failed".into(),
                    message: msg,
                },
            }
        }

        pub fn sanity_check(&mut self) -> WorkerResponse {
            let Some(comm) = self.comm.as_ref() else {
                return WorkerResponse::Error {
                    kind: "nccl_not_initialised".into(),
                    message: "sanity_check requires Init to have completed first".into(),
                };
            };
            match try_sanity_check(comm) {
                Ok(sum) => WorkerResponse::NcclSanityResult { observed_sum: sum },
                Err(msg) => WorkerResponse::Error {
                    kind: "nccl_sanity_failed".into(),
                    message: msg,
                },
            }
        }
    }

    fn try_init(state: &mut NcclState, cfg: WorkerConfig, comm_id_hex: &str) -> Result<(), String> {
        let bytes = decode_hex(comm_id_hex)?;
        if bytes.len() != NCCL_ID_BYTES {
            return Err(format!(
                "comm_id is {} bytes, expected {NCCL_ID_BYTES}",
                bytes.len()
            ));
        }
        let id_bytes: [std::ffi::c_char; NCCL_ID_BYTES] =
            std::array::from_fn(|i| bytes[i] as std::ffi::c_char);
        let id = Id::uninit(id_bytes);

        let ctx = CudaContext::new(cfg.cuda_device as usize)
            .map_err(|e| format!("CudaContext::new({}) failed: {e}", cfg.cuda_device))?;
        let stream = ctx.default_stream();
        let comm = Comm::from_rank(stream, cfg.rank as usize, cfg.world_size as usize, id)
            .map_err(|e| {
                format!(
                    "Comm::from_rank(rank={}, world={}) failed: {e}",
                    cfg.rank, cfg.world_size
                )
            })?;

        state.ctx = Some(ctx);
        state.comm = Some(comm);
        Ok(())
    }

    fn try_sanity_check(comm: &Comm) -> Result<u32, String> {
        let stream = comm.stream().clone();
        let input = stream
            .memcpy_stod(&[1u32])
            .map_err(|e| format!("htod sentinel: {e}"))?;
        let mut output = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| format!("alloc output: {e}"))?;
        comm.all_reduce(&input, &mut output, &ReduceOp::Sum)
            .map_err(|e| format!("all_reduce: {e}"))?;
        let result = stream
            .memcpy_dtov(&output)
            .map_err(|e| format!("dtoh result: {e}"))?;
        Ok(result[0])
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{NcclState, generate_comm_id_hex};

/// Non-cuda stub for the leader: returns a clear marker error rather
/// than letting `init_nccl` succeed vacuously.
#[cfg(not(feature = "cuda"))]
pub fn generate_comm_id_hex() -> Result<String, String> {
    Err("cuda_feature_not_enabled: build with --features cuda".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let original: Vec<u8> = (0u8..=255).collect();
        let encoded = encode_hex(&original);
        assert_eq!(encoded.len(), 512);
        let decoded = decode_hex(&encoded).expect("decode");
        assert_eq!(decoded, original);
    }

    #[test]
    fn hex_decode_rejects_odd_length() {
        assert!(decode_hex("a").is_err());
        assert!(decode_hex("abc").is_err());
    }

    #[test]
    fn hex_decode_rejects_non_hex() {
        assert!(decode_hex("zz").is_err());
        assert!(decode_hex("ab_d").is_err());
    }

    #[test]
    fn hex_encode_is_lowercase_padded() {
        assert_eq!(encode_hex(&[0x0a, 0xff]), "0aff");
    }
}
