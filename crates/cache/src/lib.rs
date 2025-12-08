/* helexa/crates/cache/src/lib.rs */

// SPDX-License-Identifier: PolyForm-Shield-1.0

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use thiserror::Error;

/// Error type for cache-related operations.
///
/// Most callers can treat this as a thin wrapper around `anyhow::Error`,
/// but it is useful to have a concrete error kind for logging and metrics.
#[derive(Debug, Error)]
pub enum CacheError {
    #[error("failed to determine cache directory")]
    NoCacheDir,
}

/// Returns the base cache directory for helexa under the user's home
/// directory, following the conventional:
///
///   `${HOME}/.cache/<executable>/`
///
/// shape. For systemd units where `WorkingDirectory` is set to the same
/// location as the service user's home (e.g. `/var/lib/helexa`), this means
/// caches will be written under:
///
///   `/var/lib/helexa/.cache/helexa/`
///
/// for the `helexa` binary. For regular users running the binary directly,
/// the same logic yields per-user cache directories under their `$HOME`.
///
/// We intentionally avoid platform-specific cache roots here and instead
/// rely on `$HOME` so that systemd units and interactive users both follow
/// the same convention.
pub fn helexa_cache_root() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or(CacheError::NoCacheDir)?;
    let mut path = PathBuf::from(home);
    path.push(".cache");
    path.push("helexa");
    Ok(path)
}

/// Simple JSON-backed cache store for a single logical value.
///
/// This helper is intentionally minimal and synchronous. It is designed
/// to be embedded inside higher-level components (e.g. neuron model
/// registries or cortex scheduling state) to provide:
///
/// - A well-defined on-disk location under the helexa cache root.
/// - Load-on-start and flush-on-shutdown semantics.
/// - Type-safe, serde-based persistence.
///
/// Example:
///
/// ```ignore
/// #[derive(Serialize, Deserialize)]
/// struct ModelConfigState {
///     configs: HashMap<String, ModelConfig>,
/// }
///
/// let store = JsonStore::new("models")?;
/// let state: ModelConfigState = store.load_or_default()?;
/// // ... mutate state ...
/// store.save(&state)?;
/// ```
pub struct JsonStore {
    /// Full path to the JSON file backing this store.
    path: PathBuf,
}

impl JsonStore {
    /// Create a new JSON store under the helexa cache root with the given
    /// store name.
    ///
    /// The resulting on-disk path will be:
    ///
    /// `helexa_cache_root() / {store_name}.json`
    pub fn new(store_name: &str) -> Result<Self> {
        let root = helexa_cache_root()?;
        Self::with_root(root, store_name)
    }

    /// Create a new JSON store under an explicit root directory.
    ///
    /// This is useful for tests or for callers that want to pin the cache
    /// to a specific working directory rather than the platform default.
    pub fn with_root<P: AsRef<Path>>(root: P, store_name: &str) -> Result<Self> {
        let mut path = root.as_ref().to_path_buf();
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create cache root at {}", path.display()))?;
        path.push(format!("{store_name}.json"));
        Ok(JsonStore { path })
    }

    /// Returns the underlying path of this store.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the value from disk if present, otherwise return `None`.
    ///
    /// This does not create the file on disk. Callers that want a default
    /// value should prefer [`JsonStore::load_or_default`].
    pub fn load_optional<T>(&self) -> Result<Option<T>>
    where
        T: DeserializeOwned,
    {
        if !self.path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&self.path)
            .with_context(|| format!("failed to open cache file {}", self.path.display()))?;
        let mut buf = String::new();
        file.read_to_string(&mut buf)
            .with_context(|| format!("failed to read cache file {}", self.path.display()))?;

        if buf.trim().is_empty() {
            return Ok(None);
        }

        let value = serde_json::from_str(&buf)
            .with_context(|| format!("failed to parse JSON from {}", self.path.display()))?;
        Ok(Some(value))
    }

    /// Load the value from disk if present; otherwise return `T::default()`.
    ///
    /// This is useful for state structures that always have a sensible
    /// empty/default representation.
    pub fn load_or_default<T>(&self) -> Result<T>
    where
        T: DeserializeOwned + Default,
    {
        match self.load_optional()? {
            Some(v) => Ok(v),
            None => Ok(T::default()),
        }
    }

    /// Persist the given value to disk as pretty-printed JSON.
    ///
    /// This performs a best-effort atomic-ish write by:
    /// - serialising to a temporary string,
    /// - writing it to a temporary file next to the target,
    /// - renaming the temporary file into place.
    pub fn save<T>(&self, value: &T) -> Result<()>
    where
        T: Serialize,
    {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create parent dir {} for cache", parent.display())
            })?;
        }

        let json = serde_json::to_string_pretty(value)
            .with_context(|| "failed to serialise value to JSON for cache")?;

        let tmp_path = self.path.with_extension("json.tmp");

        {
            let mut file = fs::File::create(&tmp_path).with_context(|| {
                format!(
                    "failed to create temporary cache file {}",
                    tmp_path.display()
                )
            })?;
            file.write_all(json.as_bytes()).with_context(|| {
                format!(
                    "failed to write temporary cache file {}",
                    tmp_path.display()
                )
            })?;
            file.sync_all().with_context(|| {
                format!("failed to sync temporary cache file {}", tmp_path.display())
            })?;
        }

        fs::rename(&tmp_path, &self.path).with_context(|| {
            format!(
                "failed to rename temporary cache file {} to {}",
                tmp_path.display(),
                self.path.display()
            )
        })?;

        Ok(())
    }

    /// Delete the underlying cache file, if it exists.
    ///
    /// This does not remove the parent directory.
    pub fn clear(&self) -> Result<()> {
        if self.path.exists() {
            fs::remove_file(&self.path)
                .with_context(|| format!("failed to remove cache file {}", self.path.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::env;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
    struct TestState {
        values: HashMap<String, String>,
    }

    fn temp_root() -> PathBuf {
        let mut dir = env::temp_dir();
        // Try to create a reasonably unique directory.
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("helexa-cache-test-{nanos}"));
        dir
    }

    #[test]
    fn roundtrip_save_and_load() {
        let root = temp_root();
        let store = JsonStore::with_root(&root, "state").unwrap();

        let mut state = TestState::default();
        state.values.insert("foo".into(), "bar".into());

        store.save(&state).unwrap();

        let loaded: TestState = store.load_or_default().unwrap();
        assert_eq!(state, loaded);

        // cleanup
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn load_optional_none_for_missing_file() {
        let root = temp_root();
        let store = JsonStore::with_root(&root, "missing").unwrap();

        let loaded: Option<TestState> = store.load_optional().unwrap();
        assert!(loaded.is_none());

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn clear_removes_file() {
        let root = temp_root();
        let store = JsonStore::with_root(&root, "to_clear").unwrap();

        let state = TestState::default();
        store.save(&state).unwrap();
        assert!(store.path().exists());

        store.clear().unwrap();
        assert!(!store.path().exists());

        fs::remove_dir_all(root).ok();
    }
}
