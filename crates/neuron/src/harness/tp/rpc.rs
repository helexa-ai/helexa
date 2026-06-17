//! Wire protocol between the neuron leader process and its
//! `--worker` subprocesses.
//!
//! Every frame is one newline-delimited JSON object on the worker's
//! stdin (request) or stdout (response). Both directions are tagged
//! sum types from the start so new ops in Stage 7b/7c slot in without
//! breaking compatibility — no "14 message types and a version field"
//! drift later. Adding a new variant is the canonical way to evolve
//! the protocol; existing peers that don't recognise an op return
//! `WorkerResponse::Error { kind: "unknown_op", .. }`.
//!
//! The serialised shape uses `tag = "op"` so a request looks like:
//!     {"op":"ping"}
//!     {"op":"init","comm_id":"a1b2..."}
//! and a response:
//!     {"op":"pong","rank":0,"world_size":2,"cuda_device":0}
//!     {"op":"error","kind":"nccl_init_failed","message":"..."}

use serde::{Deserialize, Serialize};

/// Leader → worker. Worker handles one at a time; replies with exactly
/// one `WorkerResponse` per request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkerRequest {
    /// Liveness probe. Worker replies with `Pong` containing its own
    /// identity. Used by the leader to confirm the subprocess is up
    /// and ready before kicking off any heavier work.
    Ping,

    /// One-shot NCCL communicator setup. The leader generates the
    /// `comm_id` once (rank 0 of NCCL), broadcasts it to every worker
    /// via this message, then every rank (leader included) calls
    /// `Comm::from_rank` with the same id — NCCL blocks until all
    /// `world_size` ranks check in. The hex-encoded bytes are the
    /// canonical `cudarc::nccl::Id::internal()` content.
    Init {
        /// Hex-encoded NCCL id bytes (128 bytes → 256 hex chars).
        comm_id: String,
    },

    /// Sanity check: after Init, every rank runs an `all_reduce` over
    /// a sentinel value (`1u32`). The expected sum is `world_size`.
    /// Worker replies with the observed value so the leader can verify
    /// the NCCL handshake is genuinely live, not just configured.
    NcclSanityCheck,

    /// Load this rank's shard of a dense Qwen3 model from mmaped
    /// safetensors. The same `safetensors_paths` list is sent to every
    /// rank — the ShardedVarBuilder reads only the rank-local slice of
    /// each tensor at materialisation time, so the worker's VRAM
    /// footprint is `1 / world_size` of the full model (plus replicated
    /// embedding/norm/lm_head).
    LoadDenseShard {
        /// Caller-supplied id for later `GenerateStep` / `UnloadModel`
        /// lookups. Typically the HF model id verbatim.
        model_id: String,
        /// JSON-serialised `candle_transformers::models::qwen3::Config`
        /// — the same blob the leader parsed from the HF cache's
        /// `config.json`. Threaded through verbatim so the worker uses
        /// identical hyperparameters.
        config_json: String,
        /// Absolute paths the worker should mmap. The same set on every
        /// rank; ShardedVarBuilder slices into them per rank.
        safetensors_paths: Vec<String>,
        /// Optional in-situ quantization dtype (e.g. "q5k", "q8_0",
        /// "q6k"). When set, each linear-layer weight is quantized
        /// at load time to the named ggml format — saves ~3-5x vs
        /// bf16/f16 at the cost of some accuracy. `None` keeps the
        /// weights in the on-disk dtype (typically bf16).
        #[serde(default)]
        quant: Option<String>,
    },

    /// Run one forward step on this rank's loaded model. The worker
    /// reaches into its NCCL Comm for the row-parallel `AllReduce`s
    /// inside the model — and so blocks on every other rank issuing the
    /// same op. The leader does *not* receive logits back over RPC; it
    /// runs its own rank-0 forward in parallel and uses its own logits
    /// for sampling.
    GenerateStep {
        model_id: String,
        /// Input token ids for this step. For prefill, the whole prompt;
        /// for decode, a single token. Identical on every rank.
        tokens: Vec<u32>,
        /// KV cache offset (count of tokens already in the cache before
        /// this step).
        offset: usize,
    },

    /// Like `GenerateStep` but the prefill carries image content. Every
    /// rank preprocesses the same `image_data_uris` through its
    /// *replicated* vision tower, splices the resulting patch embeddings
    /// at `image_token_id` positions, and runs the forward — the
    /// row-parallel `AllReduce`s still synchronise every rank. Because
    /// the tower is replicated and `preprocess_data_uri` is
    /// deterministic, the spliced hidden state is identical on every
    /// rank, so no embedding broadcast is needed. Sent only for the
    /// (single-shot) image-bearing prefill; decode steps use plain
    /// `GenerateStep`. Worker replies with the same `GenerateStepOk`.
    GenerateStepWithImages {
        model_id: String,
        tokens: Vec<u32>,
        offset: usize,
        /// `<|image_pad|>` sentinel id (248056 for Qwen3.6); splice
        /// target in the expanded token stream.
        image_token_id: u32,
        /// Source image data URIs (`data:image/...;base64,...`), one per
        /// image in prompt order. Each rank decodes + preprocesses these
        /// identically; tens of KB each, so cheap over the stdin pipe.
        image_data_uris: Vec<String>,
        /// Prefill chunk size (tokens). Sent explicitly so every rank
        /// walks the prompt in identical windows and the per-chunk
        /// row-parallel collectives stay paired across ranks.
        chunk_size: usize,
    },

    /// Reset the KV cache for this model on this rank. Sent at the
    /// start of every inference so a fresh request doesn't accidentally
    /// attend over the previous one's tokens.
    ClearKvCache { model_id: String },

    /// Capture this rank's live cache state as a prefix snapshot
    /// (#11), stored in-process under `snapshot_id`. The id is minted
    /// by the leader's pool and broadcast so every rank keys the same
    /// snapshot identically; all ranks are at the same token boundary
    /// because step fan-out is synchronous. Worker replies
    /// `KvSnapshotStored { bytes }` with this rank's snapshot size.
    SnapshotKvCache { model_id: String, snapshot_id: u64 },

    /// Replace this rank's live cache state with the stored snapshot,
    /// instead of `ClearKvCache`, so prefill resumes at the snapshot's
    /// token boundary. The snapshot remains stored.
    RestoreKvCache { model_id: String, snapshot_id: u64 },

    /// Drop one stored snapshot on this rank (prefix-cache eviction).
    /// Idempotent — replies `KvSnapshotDropped` whether or not the id
    /// was present.
    DropKvSnapshot { model_id: String, snapshot_id: u64 },

    /// Query this rank's live device VRAM as `(free_mb, total_mb)`.
    /// Non-mutating; replies `VramInfo`. Used to derive the context
    /// limit (#67) against the tightest card across ranks — a non-leader
    /// card is often tighter than the leader's.
    QueryVram,

    /// Drop this rank's shard for the given model. Releases the VRAM
    /// the shard's weights occupied; subsequent `GenerateStep` calls
    /// against the same `model_id` return an `Error`.
    UnloadModel { model_id: String },

    /// Worker should release resources and exit. Worker replies `Bye`
    /// and then closes stdout / exits zero. The leader reaps the
    /// child via the `tokio::process::Child` it kept.
    Shutdown,
}

/// Worker → leader. Always exactly one of these per `WorkerRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkerResponse {
    /// Reply to `Ping`. Carries enough identity for the leader to log
    /// what it actually got back.
    Pong {
        rank: u32,
        world_size: u32,
        cuda_device: u32,
    },

    /// Reply to `Init`. Empty payload — success is the absence of
    /// `Error`. NCCL's internal blocking handshake means by the time
    /// this comes back, every other rank has also reached
    /// `Comm::from_rank`.
    InitOk,

    /// Reply to `NcclSanityCheck`. The observed sum after a single
    /// `all_reduce(SUM, 1u32)` across all ranks. The leader checks
    /// this matches `world_size`.
    NcclSanityResult { observed_sum: u32 },

    /// Reply to `LoadDenseShard`. Empty payload — success is the
    /// absence of `Error`. By the time this comes back, the rank's
    /// `TpQwen3ForCausalLM` is constructed in memory and ready for
    /// `GenerateStep`.
    LoadDenseShardOk,

    /// Reply to `GenerateStep`. Empty payload — workers don't ship
    /// logits over the wire. The leader uses its own rank-0 logits;
    /// workers only need to confirm the collective completed.
    GenerateStepOk,

    /// Reply to `ClearKvCache`. Empty payload.
    KvCacheCleared,

    /// Reply to `QueryVram`. This rank's device VRAM in MiB.
    VramInfo { free_mb: u64, total_mb: u64 },

    /// Reply to `SnapshotKvCache`. Carries this rank's snapshot size
    /// in bytes so the leader can budget-account the whole fleet's
    /// footprint (shards are symmetric, so leader bytes × world_size
    /// is also a fine estimate; the explicit number keeps it honest).
    KvSnapshotStored { bytes: u64 },

    /// Reply to `RestoreKvCache`. Empty payload.
    KvCacheRestored,

    /// Reply to `DropKvSnapshot`. Empty payload.
    KvSnapshotDropped,

    /// Reply to `UnloadModel`. Empty payload. The named model is no
    /// longer present on this rank.
    Unloaded,

    /// Reply to `Shutdown`. Worker exits immediately after writing this.
    Bye,

    /// Any request can produce this instead of its dedicated success
    /// variant. `kind` is a machine-readable category so the leader
    /// can branch on failure mode without string-matching `message`.
    Error {
        /// Short tag — `nccl_init_failed`, `unknown_op`, etc.
        kind: String,
        /// Human-readable detail for logs.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T>(value: &T) -> T
    where
        T: Serialize + for<'de> Deserialize<'de>,
    {
        serde_json::from_str(&serde_json::to_string(value).expect("serialise"))
            .expect("deserialise")
    }

    #[test]
    fn request_ping_round_trip() {
        let req = WorkerRequest::Ping;
        let wire = serde_json::to_string(&req).unwrap();
        assert_eq!(wire, r#"{"op":"ping"}"#);
        match roundtrip(&req) {
            WorkerRequest::Ping => {}
            other => panic!("expected Ping, got {other:?}"),
        }
    }

    #[test]
    fn request_init_carries_hex_id() {
        let req = WorkerRequest::Init {
            comm_id: "deadbeef".into(),
        };
        let wire = serde_json::to_string(&req).unwrap();
        assert_eq!(wire, r#"{"op":"init","comm_id":"deadbeef"}"#);
    }

    #[test]
    fn request_generate_step_with_images_round_trip() {
        let req = WorkerRequest::GenerateStepWithImages {
            model_id: "Qwen/Qwen3.6-27B".into(),
            tokens: vec![1, 2, 248056, 3],
            offset: 0,
            image_token_id: 248056,
            image_data_uris: vec!["data:image/png;base64,AAA=".into()],
            chunk_size: 512,
        };
        let wire = serde_json::to_string(&req).unwrap();
        assert!(wire.contains(r#""op":"generate_step_with_images""#));
        match roundtrip(&req) {
            WorkerRequest::GenerateStepWithImages {
                tokens,
                image_token_id,
                image_data_uris,
                ..
            } => {
                assert_eq!(tokens, vec![1, 2, 248056, 3]);
                assert_eq!(image_token_id, 248056);
                assert_eq!(image_data_uris.len(), 1);
            }
            other => panic!("expected GenerateStepWithImages, got {other:?}"),
        }
    }

    #[test]
    fn request_shutdown_round_trip() {
        assert_eq!(
            serde_json::to_string(&WorkerRequest::Shutdown).unwrap(),
            r#"{"op":"shutdown"}"#
        );
    }

    #[test]
    fn response_pong_round_trip() {
        let resp = WorkerResponse::Pong {
            rank: 1,
            world_size: 4,
            cuda_device: 1,
        };
        let wire = serde_json::to_string(&resp).unwrap();
        assert!(wire.contains(r#""op":"pong""#));
        assert!(wire.contains(r#""rank":1"#));
        assert!(wire.contains(r#""world_size":4"#));
        match roundtrip(&resp) {
            WorkerResponse::Pong {
                rank,
                world_size,
                cuda_device,
            } => {
                assert_eq!(rank, 1);
                assert_eq!(world_size, 4);
                assert_eq!(cuda_device, 1);
            }
            other => panic!("expected Pong, got {other:?}"),
        }
    }

    #[test]
    fn response_error_carries_kind_and_message() {
        let resp = WorkerResponse::Error {
            kind: "nccl_init_failed".into(),
            message: "could not bind device".into(),
        };
        let wire = serde_json::to_string(&resp).unwrap();
        assert!(wire.contains(r#""op":"error""#));
        assert!(wire.contains(r#""kind":"nccl_init_failed""#));
    }

    #[test]
    fn response_sanity_result_round_trip() {
        let resp = WorkerResponse::NcclSanityResult { observed_sum: 4 };
        match roundtrip(&resp) {
            WorkerResponse::NcclSanityResult { observed_sum } => {
                assert_eq!(observed_sum, 4);
            }
            other => panic!("expected NcclSanityResult, got {other:?}"),
        }
    }

    /// Unknown ops on the wire deserialise to an error rather than
    /// silently mis-matching — confirms our `serde(tag = "op")`
    /// configuration rejects unknowns instead of doing fuzzy matching.
    #[test]
    fn unknown_op_fails_to_parse() {
        let result: Result<WorkerRequest, _> = serde_json::from_str(r#"{"op":"explode"}"#);
        assert!(result.is_err(), "should reject unknown op, got {result:?}");
    }
}
