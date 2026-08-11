//! Model catalogue — profiles describing how to serve each model.

use crate::discovery::DeviceInfo;
use crate::harness::{ModelCost, ModelLimit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A model serving profile loaded from models.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub harness: String,
    #[serde(default)]
    pub quant: Option<String>,
    /// Estimated VRAM usage in MB when loaded.
    #[serde(default)]
    pub vram_mb: Option<u64>,
    /// Minimum number of GPU devices required.
    #[serde(default = "default_min_devices")]
    pub min_devices: u32,
    /// Minimum VRAM per device in MB.
    #[serde(default)]
    pub min_device_vram_mb: Option<u64>,
    /// Neurons this model is allowed to run on. Empty = anywhere its
    /// device constraints are satisfied.
    ///
    /// This is an *affinity* constraint — where the model may be placed.
    /// It says nothing about whether the model may be evicted once
    /// resident; that is [`ModelProfile::residency_priority`]. The two
    /// were a single field once, which made "run only here" and "never
    /// evict here" impossible to ask for separately.
    #[serde(default)]
    pub pinned_on: Vec<String>,
    /// Which residency class this model belongs to when a node runs out
    /// of VRAM. A model may displace residents of its own class or
    /// below, and none above it — so models that should take turns on a
    /// node share a number, and a model is protected by being ranked
    /// *above* whatever must not evict it.
    ///
    /// Unset means [`DEFAULT_RESIDENCY_PRIORITY`], except for profiles
    /// carrying `pinned_on`, which default to
    /// [`PINNED_RESIDENCY_PRIORITY`] — before these were separate
    /// fields, `pinned_on` implied immunity from eviction, and a
    /// catalogue written against that meaning must not silently start
    /// allowing its flagship to be evicted.
    #[serde(default)]
    pub residency_priority: Option<u32>,
    /// Source scheme this profile's weights come from. When set, the
    /// router prefixes `id` with `scheme:` before forwarding the load
    /// request to neuron, ensuring the daemon fetches from the right
    /// registry regardless of which entry happens to match `id`.
    ///
    /// `None` lets neuron substitute its own `default_source` (typically
    /// `huggingface`). Set to `"helexa"` when the model is hosted in
    /// the helexa registry — operator-procurement-grade audit relies
    /// on this being explicit per model rather than implicit.
    #[serde(default)]
    pub source: Option<String>,

    // ── Enrichment (issue #62) ────────────────────────────────
    /// Per-model token budget. When present, advertised in `/v1/models`
    /// so clients can size and compact their context automatically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<ModelLimit>,
    /// Operator-set pricing (USD per 1M tokens). `0.0` for self-hosted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,
    /// Static capability flags the operator wants to advertise even
    /// before the model is loaded on any neuron (e.g. `"reasoning"`,
    /// `"tool_call"`). Runtime-detected capabilities from the harness
    /// are unioned with this set in the gateway's `/v1/models` response.
    #[serde(default)]
    pub capabilities: Vec<String>,
}

fn default_min_devices() -> u32 {
    1
}

/// Residency priority for a model that declares none. Deliberately not
/// zero: an operator needs room to rank something *below* the ordinary
/// case (a scratch or experimental model that should yield to anything)
/// without editing every other entry.
pub const DEFAULT_RESIDENCY_PRIORITY: u32 = 100;

/// Residency priority assumed for a profile that carries `pinned_on` but
/// no explicit priority. High enough that nothing with a default
/// priority can evict it, preserving the immunity `pinned_on` used to
/// grant on its own.
pub const PINNED_RESIDENCY_PRIORITY: u32 = 1000;

/// The full model catalogue.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCatalogue {
    #[serde(default)]
    pub models: Vec<ModelProfile>,
    /// Tier aliases — clients can send a request with `model: "helexa/small"`
    /// and the gateway transparently rewrites + routes to the concrete
    /// model id this maps to. Lets operators define latency/quality
    /// tiers (`small`/`balanced`/`large`, `fast`/`thinking`, etc.)
    /// without imposing knowledge of specific model ids on clients.
    /// Loaded from the `[aliases]` table in models.toml.
    #[serde(default)]
    pub aliases: HashMap<String, String>,
}

impl ModelCatalogue {
    /// Load the catalogue from a TOML file. Returns empty catalogue if file doesn't exist.
    pub fn load(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        if !path.exists() {
            tracing::info!(path = %path.display(), "no model catalogue found, using empty");
            return Self::default();
        }
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(cat) => cat,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "failed to parse model catalogue");
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "failed to read model catalogue");
                Self::default()
            }
        }
    }

    /// How strongly `model_id` holds its place. Models absent from the
    /// catalogue rank at the default — a model can be resident on a
    /// neuron without a profile (loaded directly, or left over from an
    /// earlier catalogue), and treating those as unevictable would let
    /// an unlisted model wedge a node permanently.
    pub fn residency_priority(&self, model_id: &str) -> u32 {
        self.get(model_id)
            .map(|p| {
                p.residency_priority.unwrap_or({
                    if p.pinned_on.is_empty() {
                        DEFAULT_RESIDENCY_PRIORITY
                    } else {
                        PINNED_RESIDENCY_PRIORITY
                    }
                })
            })
            .unwrap_or(DEFAULT_RESIDENCY_PRIORITY)
    }

    /// May `incoming` take VRAM from `resident` when a node cannot hold
    /// both?
    ///
    /// Greater-or-equal, which makes the priority a *class* rather than
    /// a strict order: a model may displace anything in its own class or
    /// below, and nothing above it.
    ///
    /// Equal rank has to permit displacement, because mutual
    /// displacement is what cold-swap *is*. Two models that share a node
    /// and take turns on it — an image generator and a mid-tier text
    /// model, or two generations of the same flagship being compared —
    /// each need to evict the other on demand. Under a strict
    /// greater-than the first one to arrive would win permanently and
    /// the other could never come back, which reads as "the model
    /// vanished after we generated an image".
    ///
    /// Protection therefore comes from ranking a model *above* its
    /// would-be evictor, not from ranking every model differently.
    ///
    /// This governs only *whether* a displacement is permitted, never
    /// whether one is needed. A node with room for both evicts nothing,
    /// however the two rank.
    pub fn may_displace(&self, incoming_id: &str, resident_id: &str) -> bool {
        self.residency_priority(incoming_id) >= self.residency_priority(resident_id)
    }

    /// Find a profile by model id.
    pub fn get(&self, model_id: &str) -> Option<&ModelProfile> {
        self.models.iter().find(|p| p.id == model_id)
    }

    /// Resolve an alias to its concrete model id. Returns `id` verbatim
    /// when it isn't an alias. Aliases never chain — operator config
    /// is treated as flat — so this is a single lookup.
    pub fn resolve_alias<'a>(&'a self, id: &'a str) -> &'a str {
        self.aliases.get(id).map(String::as_str).unwrap_or(id)
    }
}

impl ModelProfile {
    /// True iff this profile's placement constraints can be satisfied
    /// by the named neuron with the given device topology.
    ///
    /// Constraints checked:
    /// - `pinned_on`: non-empty → neuron must be on the list.
    /// - `min_devices`: neuron must have at least this many devices.
    /// - `min_device_vram_mb`: at least `min_devices` of the neuron's
    ///   devices must each meet this VRAM floor.
    pub fn is_feasible_on(&self, neuron_name: &str, devices: &[DeviceInfo]) -> bool {
        if !self.pinned_on.is_empty() && !self.pinned_on.iter().any(|n| n == neuron_name) {
            return false;
        }
        if (devices.len() as u32) < self.min_devices {
            return false;
        }
        if let Some(min_vram) = self.min_device_vram_mb {
            let big_enough = devices
                .iter()
                .filter(|d| d.vram_total_mb >= min_vram)
                .count() as u32;
            if big_enough < self.min_devices {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::DeviceInfo;

    fn device(idx: u32, vram_mb: u64) -> DeviceInfo {
        DeviceInfo {
            index: idx,
            name: format!("DEV-{idx}"),
            vram_total_mb: vram_mb,
            compute_capability: "8.6".into(),
        }
    }

    fn profile() -> ModelProfile {
        ModelProfile {
            id: "Qwen/Qwen3.6-27B".into(),
            harness: "candle".into(),
            quant: None,
            vram_mb: Some(45_000),
            min_devices: 2,
            min_device_vram_mb: Some(24_000),
            pinned_on: vec![],
            residency_priority: None,
            source: None,
            limit: None,
            cost: None,
            capabilities: vec![],
        }
    }

    #[test]
    fn feasible_when_two_devices_meet_vram_floor() {
        let p = profile();
        let devices = [device(0, 32_000), device(1, 32_000)];
        assert!(p.is_feasible_on("beast", &devices));
    }

    #[test]
    fn infeasible_when_only_one_device() {
        let p = profile();
        let devices = [device(0, 64_000)];
        assert!(!p.is_feasible_on("benjy", &devices));
    }

    #[test]
    fn infeasible_when_one_device_underspec() {
        let p = profile();
        let devices = [device(0, 32_000), device(1, 12_000)];
        assert!(!p.is_feasible_on("mixed", &devices));
    }

    #[test]
    fn pinned_on_excludes_other_neurons() {
        let mut p = profile();
        p.pinned_on = vec!["beast".into()];
        let devices = [device(0, 32_000), device(1, 32_000)];
        assert!(p.is_feasible_on("beast", &devices));
        assert!(!p.is_feasible_on("benjy", &devices));
    }

    #[test]
    fn no_vram_floor_just_needs_min_devices() {
        let mut p = profile();
        p.min_device_vram_mb = None;
        let devices = [device(0, 1_000), device(1, 1_000)];
        assert!(p.is_feasible_on("anywhere", &devices));
    }

    /// A catalogue shaped like a real fleet. Two residency classes: the
    /// big-node class holds a flagship and a frontier model that take
    /// turns on the one machine large enough for either; the everyday
    /// class holds an image generator and a mid-tier text model that
    /// take turns on a smaller one. Nothing in the everyday class may
    /// touch the big-node class.
    fn tiered_catalogue() -> ModelCatalogue {
        toml::from_str(
            r#"
[[models]]
id = "flagship"
harness = "candle"
pinned_on = ["big-node"]
residency_priority = 300

[[models]]
id = "frontier"
harness = "candle"
residency_priority = 300

[[models]]
id = "image"
harness = "candle"
residency_priority = 200

[[models]]
id = "mid"
harness = "candle"
residency_priority = 200

[[models]]
id = "tiny"
harness = "candle"
residency_priority = 100
"#,
        )
        .expect("parse tiered catalogue")
    }

    #[test]
    fn image_generation_displaces_the_mid_tier() {
        assert!(tiered_catalogue().may_displace("image", "mid"));
    }

    #[test]
    fn the_mid_tier_comes_back_after_an_image_takes_its_node() {
        // The swap-back half. An image request evicts the text model;
        // the next text request must be able to evict the image model,
        // or the text tier disappears until someone restarts something.
        assert!(tiered_catalogue().may_displace("mid", "image"));
    }

    #[test]
    fn two_generations_of_a_flagship_can_swap_in_both_directions() {
        // Comparing a new flagship against the incumbent needs traffic
        // to move each way on demand. A strict order would let whichever
        // arrived first hold the node permanently.
        let cat = tiered_catalogue();
        assert!(cat.may_displace("frontier", "flagship"));
        assert!(cat.may_displace("flagship", "frontier"));
    }

    #[test]
    fn image_generation_never_displaces_the_flagship() {
        // The image generator's device constraints alone would let it
        // land on the flagship's node, so this is the case priority
        // exists to prevent -- not a hypothetical one.
        assert!(!tiered_catalogue().may_displace("image", "flagship"));
    }

    #[test]
    fn the_frontier_tier_displaces_the_flagship() {
        // The requirement a boolean pin cannot express: the flagship is
        // protected from one model and not from another.
        assert!(tiered_catalogue().may_displace("frontier", "flagship"));
    }

    #[test]
    fn a_lower_class_cannot_displace_a_higher_one() {
        let cat = tiered_catalogue();
        assert!(!cat.may_displace("tiny", "mid"));
        assert!(!cat.may_displace("tiny", "flagship"));
    }

    #[test]
    fn a_higher_class_can_displace_a_lower_one() {
        assert!(tiered_catalogue().may_displace("flagship", "tiny"));
    }

    #[test]
    fn pinned_on_alone_still_protects_a_catalogue_written_before_priorities() {
        // `pinned_on` used to mean "never evict here". A catalogue that
        // predates the split says nothing about priority, and must not
        // silently start allowing its flagship to be evicted.
        let cat: ModelCatalogue = toml::from_str(
            r#"
[[models]]
id = "flagship"
harness = "candle"
pinned_on = ["big-node"]

[[models]]
id = "ordinary"
harness = "candle"
"#,
        )
        .expect("parse legacy catalogue");
        assert!(!cat.may_displace("ordinary", "flagship"));
        assert!(cat.may_displace("flagship", "ordinary"));
    }

    #[test]
    fn an_unlisted_resident_is_displaceable_by_a_ranked_model() {
        // A model can be resident without a profile. Treating it as
        // unevictable would let an unlisted model wedge a node forever.
        let cat = tiered_catalogue();
        assert_eq!(
            cat.residency_priority("never-heard-of-it"),
            DEFAULT_RESIDENCY_PRIORITY
        );
        assert!(cat.may_displace("image", "never-heard-of-it"));
    }

    #[test]
    fn affinity_and_immunity_are_independently_expressible() {
        // The whole point of the split: confine a model to a node
        // without protecting it there, and protect one without
        // confining it anywhere.
        let cat: ModelCatalogue = toml::from_str(
            r#"
[[models]]
id = "confined-but-evictable"
harness = "candle"
pinned_on = ["big-node"]
residency_priority = 50

[[models]]
id = "roaming-but-protected"
harness = "candle"
residency_priority = 900
"#,
        )
        .expect("parse catalogue");
        let devices = [device(0, 32_000)];

        let confined = cat.get("confined-but-evictable").unwrap();
        assert!(confined.is_feasible_on("big-node", &devices));
        assert!(!confined.is_feasible_on("other-node", &devices));

        let roaming = cat.get("roaming-but-protected").unwrap();
        assert!(roaming.is_feasible_on("other-node", &devices));

        assert!(cat.may_displace("roaming-but-protected", "confined-but-evictable"));
        assert!(!cat.may_displace("confined-but-evictable", "roaming-but-protected"));
    }

    #[test]
    fn resolve_alias_returns_target_when_alias_present() {
        let mut cat = ModelCatalogue::default();
        cat.aliases
            .insert("helexa/small".into(), "Qwen/Qwen3-1.7B".into());
        assert_eq!(cat.resolve_alias("helexa/small"), "Qwen/Qwen3-1.7B");
    }

    #[test]
    fn resolve_alias_passes_through_when_not_an_alias() {
        let mut cat = ModelCatalogue::default();
        cat.aliases
            .insert("helexa/small".into(), "Qwen/Qwen3-1.7B".into());
        assert_eq!(cat.resolve_alias("Qwen/Qwen3-8B"), "Qwen/Qwen3-8B");
    }

    #[test]
    fn source_defaults_to_none_when_absent_from_toml() {
        let src = r#"
[[models]]
id = "Qwen/Qwen3-30B"
harness = "candle"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse models table");
        assert!(cat.models[0].source.is_none());
    }

    #[test]
    fn source_round_trips_through_toml() {
        let src = r#"
[[models]]
id = "Helexa/Qwen3.6-27B-Uncensored"
harness = "candle"
source = "helexa"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse models table");
        assert_eq!(cat.models[0].source.as_deref(), Some("helexa"));
    }

    #[test]
    fn aliases_table_round_trips_through_toml() {
        let src = r#"
[aliases]
"helexa/small" = "Qwen/Qwen3-1.7B"
"helexa/large" = "Qwen/Qwen3.6-27B"
"#;
        let cat: ModelCatalogue = toml::from_str(src).expect("parse aliases table");
        assert_eq!(cat.resolve_alias("helexa/small"), "Qwen/Qwen3-1.7B");
        assert_eq!(cat.resolve_alias("helexa/large"), "Qwen/Qwen3.6-27B");
    }
}
